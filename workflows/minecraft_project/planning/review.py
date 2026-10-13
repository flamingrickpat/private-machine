"""Review role: independent review of one plan revision."""
from __future__ import annotations

import json

from pm_workflows.python_role import RoleContext

from workflows.minecraft_project.schemas import PlanReview
from workflows.minecraft_project.tree_store import TreeStore, validate_tree
from workflows.minecraft_project.acceptance import observe_conditions
from workflows.minecraft_project.evidence_selection import accepted_inputs, dependency_scope, review_nodes
from workflows.minecraft_project.planning.run_context import boundary, identity, scoped_reads
from workflows.minecraft_project.planning.decisions import reason
from workflows.minecraft_project.history_encoding import gameplay_history
from workflows.minecraft_project.planning.proposals import survey_conditions
from workflows.minecraft_project.planning.survey import recipe_reads
from workflows.minecraft_project.planning.constraints import caller_plan_constraints, project_check_context, record_plan_constraints


def review(context: RoleContext, store: TreeStore) -> dict:
    """Review acceptance coverage independently before selecting any leaf."""
    if not boundary(context, store):
        return {"status": "refine", "summary": "Project instructions changed before plan review."}
    tree = store.load()
    result = record_plan_constraints(context, store)
    if result is not None:
        if result["errors"]:
            store.state.pop("reviewed_revision", None)
            rejected = PlanReview(status="revise", summary="The configured project contract requires repair: " + " ".join(result["errors"]))
            record_review(store, rejected)
            return {"status": rejected.status, "summary": rejected.summary}
        store.save()
    ordered = review_nodes(store)
    survey_path = f"planning/surveys/{len(store.state['decisions']) + 1:04d}.json"
    survey = {"identity": identity(context, store), "reads": observe_conditions(context, survey_conditions(ordered))}
    # These are planning facts, not completion evidence. Query exact requested
    # inventory names so the reviewer does not invent crafting ratios.
    names = sorted({name.removeprefix("minecraft:") for node in ordered for condition in node.criteria
                    if condition.kind == "inventory" for name in condition.names if not any(c in name for c in "*?[" )})
    survey["recipes"] = recipe_reads(context, names)
    survey["tool_contracts"] = [row for row in context.runtime.tools["minecraft_tool_schemas"]()
                                if row["name"] in {"minecraft_recipe_search", "minecraft_smelt_item", "minecraft_eat_best"}]
    store.write(survey_path, survey)
    store.state["planning_survey"] = survey_path
    store.save()
    common = {
        "identity": identity(context, store), "goal": store.state["goal"], "tree": tree.model_dump(mode="json"),
        # The reviewer needs the same caller-owned coverage as the planner.
        # Separate publication audits cannot be inferred from static cells.
        "plan_constraints": caller_plan_constraints(store),
        "latest_project_check": project_check_context(store),
        "server_rules": json.loads(store.path_for(store.state["server_rules"]).read_text(encoding="utf-8")).get("server_rules", "") if store.state.get("server_rules") else "",
        "leaf_prerequisites": {key: sorted(value) for key, value in validate_tree(tree).items()},
        "accepted_nodes": list(store.state["acceptances"]), "active_recovery": gameplay_history(store)["recovery"],
        "project_initial_observation": json.loads(store.path_for(store.state["initial_observation"]).read_text(encoding="utf-8")),
        "original_observation": json.loads(store.path_for(store.state["planning_observation"]).read_text(encoding="utf-8")),
        "observation_path": store.state["planning_observation"],
         "current_site_survey": {"identity": survey["identity"], "reads": scoped_reads(survey["reads"]),
                                 "recipes": scoped_reads(survey["recipes"]), "tool_contracts": survey["tool_contracts"]},
        "survey_path": survey_path}
    topology = [{"id": node.id, "parent": node.parent, "children": node.children,
                 "depends_on": node.depends_on, "kind": node.kind} for node in tree.nodes]
    historical = accepted_inputs(store, {node.id for node in tree.nodes})
    reviews = [{"node_id": key, "status": "accepted", "scope": "Original independently verified historical attainment.",
                "original_receipt": value["certificate"]["verification"]} for key, value in historical.items()]
    for node in ordered:
        if not boundary(context, store):
            return {"status": "refine", "summary": "Project instructions changed during scoped plan review."}
        selected = dependency_scope(tree, [node.id])
        scoped = {**common, "tree": {"goal_id": tree.goal_id, "revision": tree.revision,
                                    "nodes": [node.model_dump(mode="json")]},
                  "tree_topology": topology, "review_scope": {"node_id": node.id, "kind": node.kind},
                  "dependency_specifications": [row.model_dump(mode="json") for row in tree.nodes
                                                if row.id in selected and row.id != node.id and row.kind == "leaf" and not store.valid(row.id)],
                  "accepted_dependencies": accepted_inputs(store, selected), "preceding_reviews": reviews,
                  "current_site_survey": {**common["current_site_survey"],
                      "reads": scoped_reads(observe_conditions(context, survey_conditions([node])))}}
        result = reason(context, store, "reviewer", PlanReview, scoped)
        if not boundary(context, store):
            return {"status": "refine", "summary": "Project instructions changed during scoped plan review."}
        reviews.append({"node_id": node.id, **result.model_dump(mode="json"),
                        "original_receipt": store.state["decisions"][-1]["path"] + "/output.json"})
        if result.status != "accepted":
            if not result.target_id:
                result.target_id = node.id
            store.state["review_target"] = result.target_id
            break
    if not boundary(context, store):
        return {"status": "refine", "summary": "Project instructions changed during plan review."}
    record_review(store, result)
    return {"status": result.status, "summary": result.summary}


def record_review(store: TreeStore, result: PlanReview) -> None:
    """Keep every review outcome available to later explicit recovery."""
    if result.status == "refine":
        if result.target_id not in {node.id for node in store.load().nodes}:
            raise ValueError("The reviewer named an unknown refinement target.")
        store.state.update(feedback=result.summary, refinement_target=result.target_id)
    elif result.status == "accepted":
        store.state.update(reviewed_revision=store.state["revision"], feedback="", refinement_target="")
        store.state.pop("review_target", None)
    else:
        store.state.update(feedback=result.summary, refinement_target="")
    store.save()

