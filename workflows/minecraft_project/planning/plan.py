"""Planner role: propose, repair, or refine the task tree."""
from __future__ import annotations

import json
from contextlib import nullcontext
from pm.memory.snapshot import GhostConflict

from pm_workflows.python_role import RoleContext

from workflows.minecraft_body.lease import BODY_LEASES
from workflows.minecraft_project.schemas import Node
from workflows.minecraft_project.tree_store import TreeStore, digest, validate_tree
from workflows.minecraft_project.evidence_selection import dependency_scope, protect_accepted_specifications, scoped_history
from workflows.minecraft_project.planning.run_context import identity, request, project_intent, scoped_reads, InstructionsChanged
from workflows.minecraft_project.planning.decisions import compact_plan_context
from workflows.minecraft_project.history_encoding import gameplay_history
from workflows.minecraft_project.planning.proposals import initial_candidate, refinement_scope, rejected_context, restart_review, revision_candidate, scoped_rejection
from workflows.minecraft_project.planning.retention import retain
from workflows.minecraft_project.planning.constraints import caller_plan_constraints, project_check_context, record_plan_constraints


def plan(context: RoleContext, store: TreeStore) -> dict:
    """Create or revise a retained tree from fresh observations and feedback."""
    req = request(context)
    store.state.pop("planning_memory_revisions", None)
    current = context.runtime.tools["managed_instructions"]()
    goal = current["instruction"] or req.goal or req.instruction
    with BODY_LEASES.command(context.runtime.tools["minecraft_endpoint"](), context.runtime.stop_event):
        observation = context.runtime.tools["minecraft_call"]("minecraft_observe", {"include_image": False})
        server_info = context.runtime.tools["minecraft_call"]("minecraft_info", {})
    if not observation["ok"]:
        return {"status": "waiting", "summary": "A fresh Minecraft observation is unavailable. Restore the body connection."}
    if not server_info["ok"]:
        return {"status": "waiting", "summary": "Current server rules are unavailable. Restore the server connection before choosing recovery."}
    observation_path = f"planning/observations/{len(store.state['decisions']) + 1:04d}.json"
    store.write(observation_path, observation["result"])
    rules_path = f"planning/server-rules/{len(store.state['decisions']) + 1:04d}.json"
    store.write(rules_path, server_info["result"])
    store.state["server_rules"] = rules_path
    store.state["planning_observation"] = observation_path
    first = "planning/observations/0001.json"
    store.state.setdefault("initial_observation", first if store.path_for(first).is_file() else observation_path)
    store.save()
    retention = retain(context, store, observation["result"])
    if retention["status"] != "unchanged":
        return retention
    # A resumed caller can revise its audit operands. Read the current
    # contract before planning instead of reusing an older review receipt.
    if store.state["revision"]:
        record_plan_constraints(context, store)
    # Re-review the retained tree against fresh facts and current contracts.
    # A historical waiting review may have misread evidence already provided.
    if restart_review(store, context.phase, current["revision"], context.answer or req.answer):
        return {"status": "planned", "summary": "Retained the task tree after fresh restart observation. Review it before another actor."}
    # The supplied planner schema defines omitted defaults. This representation
    # round-trips exactly, while rejected raw output remains intact for repair.
    previous = store.load().model_dump(mode="json", exclude_defaults=True) if store.state["revision"] else None
    repair_base = initial_candidate(store)
    payload = {"identity": identity(context, store), "goal": goal, "observation": observation["result"],
               "project_intent": project_intent(context, store, goal),
               "server_rules": server_info["result"].get("server_rules", ""),
               "planning_mode": "initial_candidate_repair" if repair_base else "initial_full_tree" if previous is None else "retained_revision",
               "project_initial_observation": json.loads(store.path_for(store.state["initial_observation"]).read_text(encoding="utf-8")),
               "previous_tree": previous, "accepted_nodes": list(store.state["acceptances"]),
               "feedback": store.state["feedback"], "refinement_target": store.state.get("refinement_target", ""),
               "persona_context": req.persona_context, "answer": context.answer or req.answer,
               "latest_attempt": store.state["attempts"][-1] if store.state["attempts"] else None}
    payload["latest_project_check"] = project_check_context(store)
    if repair_base:
        source, candidate = repair_base
        payload["initial_repair_base"] = {"original_receipt": source, "original_digest": digest(candidate.model_dump(mode="json")),
                                        "accepted": False, "tree": candidate.model_dump(mode="json", exclude_defaults=True)}
        # A later fragment or malformed patch can have different feedback.
        # Diagnose this exact immutable base before requesting node repairs.
        try:
            validate_tree(candidate)
        except ValueError as error:
            payload["initial_repair_base"]["tree_error"] = str(error)
    payload["current_tool_contracts"] = [row for row in context.runtime.tools["minecraft_tool_schemas"]()
                                         if row["name"] in {"minecraft_recipe_search", "minecraft_smelt_item"}]
    payload["gameplay_history"] = gameplay_history(store)
    candidates = store.acceptance_candidates()
    if candidates:
        from workflows.minecraft_project.tree_store import node_digest
        current_nodes = {node.id: node for node in store.load().nodes}
        payload["retained_acceptance_candidates"] = [
            {"specification": ({"from_node": row["specification"]["id"], "in": "previous_tree"}
                               if row["specification"]["id"] in current_nodes
                               and node_digest(current_nodes[row["specification"]["id"]]) == row["certificate"]["spec_digest"]
                               else Node.model_validate(row["specification"]).model_dump(mode="json", exclude_defaults=True)),
             "archive": row["archive"],
             "certificate_digest": digest(row["certificate"]),
             "required_certificates": row["certificate"]["dependencies"],
             "changed_dependencies": [key for key, expected in row["certificate"]["dependencies"].items()
                                      if key not in store.state["acceptances"] or digest(store.state["acceptances"][key]) != expected],
             "verification": row["certificate"]["verification"]} for row in candidates]
    payload["refinement_target"] = refinement_scope(store, current["revision"], payload["gameplay_history"])
    if payload["gameplay_history"]["recovery"]:
        payload["refinement_scope_reason"] = (
            "A ready recovery leaf already exists. Honor refinement_target and retain that recovery leaf unchanged."
            if payload["refinement_target"] else
            "Active recovery can supersede optional scope when it needs a ready leaf. Follow the latest review and retain useful completed work.")
    latest_review = next((row for row in reversed(store.state["decisions"]) if row["role"] == "reviewer"), None)
    if latest_review:
        review_path = latest_review["path"] + "/output.json"
        if store.path_for(review_path).is_file():
            payload["latest_plan_review"] = {**json.loads(store.path_for(review_path).read_text(encoding="utf-8")),
                                             "original_receipt": review_path}
    latest_verifier = next((row for row in reversed(store.state["decisions"]) if row["role"] == "verifier"), None)
    if latest_verifier:
        decision = json.loads(store.path_for(latest_verifier["path"] + "/input.json").read_text(encoding="utf-8"))
        selected = decision.get("context") or json.loads(decision["messages"][1]["content"])
        relative = selected["original_receipt"]
        proof = json.loads(store.path_for(relative).read_text(encoding="utf-8"))
        payload["latest_verification"] = {"identity": proof["identity"], "checks": proof["checks"],
            "verified": proof["verified"],
            "review": proof["review"], "specification": selected["specification"], "reads": scoped_reads(proof["reads"]),
            "original_receipt": relative, "original_digest": digest(proof)}
    if store.state.get("planning_survey"):
        survey = json.loads(store.path_for(store.state["planning_survey"]).read_text(encoding="utf-8"))
        payload["previous_site_survey"] = {"identity": survey["identity"], "original_receipt": store.state["planning_survey"],
                                           "original_digest": digest(survey)}
        if not payload["refinement_target"]:
            payload["previous_site_survey"]["recipes"] = scoped_reads(survey.get("recipes", []))
            payload["previous_site_survey"]["tool_contracts"] = survey.get("tool_contracts", [])
        # Prefer the latest verifier's exact scope during repair. The earlier
        # site survey stays retained; every revised plan receives a new survey.
        if not latest_verifier:
            payload["previous_site_survey"]["reads"] = scoped_reads(survey["reads"])
    rejected = store.state.get("rejected_proposal")
    if rejected:
        if repair_base:
            document = json.loads(store.path_for(rejected).read_text(encoding="utf-8"))
            payload["rejected_proposal"] = scoped_rejection(repair_base[1], document, rejected)
            payload["rejected_proposal"]["comparison_tree"] = "initial_repair_base.tree (unaccepted candidate)"
        else:
            payload["rejected_proposal"] = rejected_context(store, rejected)
    if store.state.get("requested_survey"):
        relative = store.state["requested_survey"]
        observed = json.loads(store.path_for(relative).read_text(encoding="utf-8"))
        payload["requested_world_survey"] = {**observed, "reads": scoped_reads(observed["reads"]),
                                             "recipes": scoped_reads(observed.get("recipes", [])),
                                             "original_receipt": relative, "original_digest": digest(observed)}
    if store.state.get("planning_constraints"):
        payload["plan_constraints"] = caller_plan_constraints(store)
    selected_scope = None
    if previous:
        tree = store.load()
        target = payload["refinement_target"] or store.state.get("review_target") or store.state.get("selection", {}).get("leaf_id")
        if target:
            selected_scope = dependency_scope(tree, [target])
            full = payload["gameplay_history"]
            relative = f"planning/context/{len(store.state['decisions']) + 1:04d}-history.json"
            store.write(relative, full)
            payload["gameplay_history"] = scoped_history(store, selected_scope, full, {target})
            payload["full_history"] = {"original_receipt": relative, "original_digest": digest(full)}
            # Complete criteria remain on selected nodes. Ancestor integration
            # contracts stay explicit; unrelated siblings appear as topology.
            ancestors, cursor = set(), target
            nodes = {node.id: node for node in tree.nodes}
            while cursor is not None:
                ancestors.add(cursor)
                cursor = nodes[cursor].parent
            payload["previous_tree"] = {**previous, "nodes": [row for row in previous["nodes"]
                if row["id"] in selected_scope | ancestors]}
            payload["tree_topology"] = [{"id": node.id, "parent": node.parent, "children": node.children,
                                          "depends_on": node.depends_on, "kind": node.kind}
                                         for node in tree.nodes]
            payload["decision_scope"] = {"target": target, "node_ids": sorted(selected_scope),
                                         "meaning": "Evidence scope only. Omitted task nodes remain unchanged in the durable tree."}
    restore_nodes = []
    if req.proposed_tree is not None and previous is None and not store.state.get("source_proposal_used"):
        store.state["source_proposal_used"] = True
        store.save()
        proposed = req.proposed_tree
    else:
        try:
            if payload["refinement_target"]:
                payload["replacement_binding"] = {"id": payload["refinement_target"],
                    "parent": next(node.parent for node in store.load().nodes if node.id == payload["refinement_target"]),
                    "conditions": "Copy this node's original criteria from previous_tree exactly. Keep unrelated nodes omitted."}
            from workflows.minecraft_project.planning.native_plan import plan_session
            proposal = plan_session(context, store, compact_plan_context(payload),
                scope=selected_scope | ancestors if selected_scope is not None else None)
        except InstructionsChanged as error:
            store.state.update(feedback=str(error), rejected_proposal="")
            store.save()
            return {"status": "invalid", "summary": str(error)}
        if proposal.status == "survey":
            store.state["pending_survey"] = {"instruction_revision": current["revision"],
                                              "queries": [row.model_dump(mode="json") for row in proposal.queries],
                                              "recipe_queries": proposal.recipe_queries}
            store.save()
        if proposal.status != "planned":
            return {"status": proposal.status, "summary": proposal.summary}
        if (proposal.updates or proposal.restore_nodes) and previous is None and repair_base is None:
            store.state.update(feedback="Initial planning requires a full tree. Node updates need previous_tree.",
                               rejected_proposal=store.state["decisions"][-1]["path"] + "/raw.json")
            store.save()
            return {"status": "invalid", "summary": store.state["feedback"]}
        try:
            proposed = proposal.tree if proposal.tree is not None else revision_candidate(
                repair_base[1] if repair_base else store.load(), proposal)
        except ValueError as error:
            store.state.update(feedback="The node revision was rejected before dispatch: " + str(error),
                               rejected_proposal=store.state["decisions"][-1]["path"] + "/raw.json")
            store.save()
            return {"status": "invalid", "summary": store.state["feedback"]}
        restore_nodes = proposal.restore_nodes
        if selected_scope is not None:
            original_nodes = {node.id: node for node in store.load().nodes}
            from workflows.minecraft_project.tree_store import node_digest
            protected = set(original_nodes) - selected_scope - ancestors
            if any(node.id in protected and node_digest(node) != node_digest(original_nodes[node.id])
                   for node in proposal.updates):
                return {"status": "invalid", "summary": "A scoped repair changed an unrelated existing task. Select its own scope first."}
    target = payload["refinement_target"]
    try:
        # Restoring proof is an explicit native plan operation. The model
        # need not reproduce every unchanged historical field by hand.
        proposed = store.restore_specifications(proposed, restore_nodes)
        if previous:
            protect_accepted_specifications(store, proposed)
        if context.phase == "replan" and previous and store.state.get("require_leaf_change"):
            old_nodes = {node.id: node for node in store.load().nodes}
            from workflows.minecraft_project.tree_store import node_digest
            changed = [node for node in proposed.nodes if node.kind == "leaf"
                       and (node.id not in old_nodes or node_digest(node) != node_digest(old_nodes[node.id]))]
            if not changed:
                raise ValueError("Replanning must add, revise, or defer a concrete leaf. An explanation alone cannot release another attempt.")
        guard = context.runtime.tools.get("memory_guard")
        revisions = store.state.get("planning_memory_revisions", {})
        def commit_plan():
            # Managed control holds its lock before the memory-owner lock.
            # This order matches supervisor updates and avoids a lock cycle.
            with (guard(revisions) if guard is not None else nullcontext()):
                context.runtime.check_cancelled()
                store.revise(proposed, target=target)
        authorize = context.runtime.tools.get("managed_authorize")
        if authorize is not None:
            if not authorize(current["revision"], commit_plan):
                raise InstructionsChanged("Project instructions changed before the plan commit.")
        else:
            if context.runtime.tools["managed_instructions"]()["revision"] != current["revision"]:
                raise InstructionsChanged("Project instructions changed before the plan commit.")
            commit_plan()
    except (ValueError, GhostConflict, InstructionsChanged) as error:
        rejection = store.state.get("rejections", 0) + 1
        relative = f"planning/rejected/{rejection:04d}.json"
        store.write(relative, proposed.model_dump(mode="json"))
        store.state.update(feedback="The proposed tree was rejected before dispatch: " + str(error),
                           rejected_proposal=relative, rejections=rejection)
        store.save()
        return {"status": "invalid", "summary": store.state["feedback"]}
    store.state.update(instruction_revision=current["revision"], refinement_target="", goal=goal, rejected_proposal="")
    store.state.pop("require_leaf_change", None)
    store.save()
    return {"status": "planned", "summary": f"Retained specification revision {store.state['revision']}."}

