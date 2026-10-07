"""Declared planner, selector, specialist, and verifier workflow roles."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import TypeVar

import yaml
from pm_workflows.python_role import RoleContext
from pydantic import BaseModel, ValidationError

from runtime.body_lease import BODY_LEASES
from runtime.cancellation import LinkedStopEvent
from runtime.workflow_runner import WorkflowRequest, WorkflowRunner, WorkflowStatus
from working_target import FolderEnvironment
from workflows.minecraft.schemas import MinecraftRequest
from workflows.minecraft_tasks.schemas import (
    Condition, Node, PlanProposal, PlanReview, ProjectRequest, VerificationReview,
)
from workflows.minecraft_tasks.tree import TreeStore, digest, validate_tree
from workflows.minecraft_tasks.verification import evaluate, observe_conditions, verified_project

ROOT = Path(__file__).resolve().parents[2]
RESOURCES = Path(__file__).with_name("resources")
Decision = TypeVar("Decision", bound=BaseModel)


class InstructionsChanged(RuntimeError):
    """A leaf became stale before its next action. The world remains intact."""


def request(context: RoleContext) -> ProjectRequest:
    """Read a scoped project request without public transport metadata."""
    payload = yaml.safe_load(context.runtime.environment.read_text("inbox/request.yaml"))
    return ProjectRequest.model_validate({key: value for key, value in payload.items() if not key.startswith("_public_")})


def store_for(context: RoleContext) -> TreeStore:
    """Use the working target's task folder as the sole durable tree owner."""
    return TreeStore(context.runtime.environment._path("."))


def identity(context: RoleContext, store: TreeStore, *, node: Node | None = None, actor_attempt: str = "", verifier_attempt: str = "") -> dict:
    """Bind every decision to the task, specification, and exact attempts."""
    return {"task_id": context.task_id, "tree_revision": store.state["revision"],
            "leaf_id": node.id if node else "", "spec_revision": node.revision if node else 0,
            "actor_attempt": actor_attempt, "verifier_attempt": verifier_attempt}


def reason(context: RoleContext, store: TreeStore, role: str, contract: type[Decision], payload: dict) -> Decision:
    """Run one bounded LLM decision and retain its exact input and output.

    Direct calls receive a scoped snapshot, never an accumulating transcript.
    Oversized inputs fail explicitly. Long execution stays in native pm-coder.
    Cancellation and stale instructions prevent acceptance after inference.
    """
    context.runtime.check_cancelled()
    messages = [{"role": "system", "content": (RESOURCES / f"{role}.md").read_text(encoding="utf-8")
                 + "\nJSON schema:\n" + json.dumps(contract.model_json_schema(), separators=(",", ":"))},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]
    name = f"decisions/{len(store.state['decisions']) + 1:04d}-{role}"
    generation = {"chat_template_kwargs": {"enable_thinking": False}}
    input_characters = sum(len(row["content"]) for row in messages)
    store.write(name + "/input.json", {"identity": payload["identity"], "messages": messages,
                                        "generation": generation, "input_characters": input_characters})
    store.state["decisions"].append({"role": role, "path": name, "identity": payload["identity"]})
    store.save()
    if input_characters > 120000:
        raise ValueError(f"The scoped Minecraft decision has {input_characters} characters; its input budget is 120000. Original input: {name}/input.json")
    complete = context.runtime.tools["minecraft_reason"]
    if complete is None:
        raise ValueError("Minecraft project decisions need the configured normal LLM.")
    # Use the same normal-chat mode as selection and native actor sessions.
    # Retain the setting with the decision; do not cap the output mid-object.
    text = complete(messages, cancel=context.runtime.stop_event, **generation)
    store.write(name + "/raw.json", {"text": text})
    context.runtime.check_cancelled()
    if text.strip().startswith("```"):
        text = "\n".join(text.strip().splitlines()[1:-1])
    result = contract.model_validate_json(text)
    store.write(name + "/output.json", result.model_dump(mode="json"))
    return result


def boundary(context: RoleContext, store: TreeStore) -> bool:
    """Detect a newer project instruction before another role accepts work."""
    current = context.runtime.tools["managed_instructions"]()
    return current["revision"] == store.state["instruction_revision"]


def progress_facts(observation: dict) -> dict:
    """Exclude clock and view changes from the repeated-yield guard."""
    player = observation.get("player", {})
    return {"inventory": observation.get("inventory", []),
            "position": observation.get("camera", {}).get("feet_position"),
            "player": {key: player.get(key) for key in ("health", "food", "oxygen", "dimension")}}


def scoped_reads(reads: list[dict]) -> list[dict]:
    """Retain exact evidence and distinguish known air from an unavailable read."""
    from runtime.minecraft_actor import observed_target
    scoped = []
    for row in reads:
        data = row["result"]
        relevant = {key: data[key] for key in ("position", "block_name", "is_solid", "reason", "blocks", "total_cells", "unseen_cells",
                                              "node_id", "verified", "certificate_digest", "verification",
                                              "source_specification", "source_identity", "source_checks", "source_observed_times",
                                              "pattern", "matches", "total_recipes") if key in data}
        if row["tool"] == "minecraft_observe":
            relevant.update(inventory=data.get("inventory", []), camera=data.get("camera", {}), player=data.get("player", {}))
        available = (observed_target(row, row["arguments"]["position"])
                     if row["tool"] == "minecraft_inspect_block" else bool(row["ok"]))
        scoped.append({**{key: value for key, value in row.items() if key != "result"},
                       "observation_available": available,
                       "result": relevant, "original_digest": digest(data)})
    return scoped


def survey_conditions(nodes: list[Node]) -> list[Condition]:
    """Inspect proposed physical operands once, without requiring completion.

    Survey facts describe the current site. Unmet future conditions are normal.
    Deduplication reduces repeated reads across parent and child criteria.
    """
    queries = {}
    for node in nodes:
        for condition in node.criteria + node.assumptions:
            if condition.kind == "block":
                for point in condition.positions or [condition.position]:
                    key = digest({"kind": "block", "position": point.model_dump(), "dimension": condition.dimension})
                    queries.setdefault(key, condition.model_copy(update={"position": point, "positions": []}))
            elif condition.kind == "region":
                key = digest({"kind": "region", "start": condition.position.model_dump(), "end": condition.end.model_dump(), "dimension": condition.dimension})
                queries.setdefault(key, condition.model_copy())
    return [Condition(id="survey-body", description="Current body state.", kind="alive")] + [
        condition.model_copy(update={"id": f"survey-{index:04d}"}) for index, condition in enumerate(queries.values(), 1)]


def plan(context: RoleContext, store: TreeStore) -> dict:
    """Create or revise a retained tree from fresh observations and feedback."""
    req = request(context)
    current = context.runtime.tools["managed_instructions"]()
    goal = current["instruction"] or req.goal or req.instruction
    with BODY_LEASES.command(context.runtime.tools["minecraft_endpoint"](), context.runtime.stop_event):
        observation = context.runtime.tools["minecraft_call"]("minecraft_observe", {"include_image": False})
    if not observation["ok"]:
        return {"status": "waiting", "summary": "A fresh Minecraft observation is unavailable. Restore the body connection."}
    observation_path = f"planning/observations/{len(store.state['decisions']) + 1:04d}.json"
    store.write(observation_path, observation["result"])
    store.state["planning_observation"] = observation_path
    first = "planning/observations/0001.json"
    store.state.setdefault("initial_observation", first if store.path_for(first).is_file() else observation_path)
    store.save()
    # A replacement kernel starts at observation, not at the old act cursor.
    # Keep the accepted tree when the instruction did not change.
    if context.phase == "plan" and store.state["revision"] and any(
            row["state"] == "running" for row in store.state["attempts"]):
        for attempt in store.state["attempts"]:
            if attempt["state"] == "running":
                attempt.update(state="interrupted", journal=attempt["path"] + "/scratch/actor-journal.jsonl")
        store.state["recovery_observation"] = observation_path
        store.save()
        if current["revision"] == store.state["instruction_revision"]:
            return {"status": "planned", "summary": "Retained the task tree after fresh restart observation. Reconcile the interrupted leaf."}
    # The supplied planner schema defines omitted defaults. This representation
    # round-trips exactly, while rejected raw output remains intact for repair.
    previous = store.load().model_dump(mode="json", exclude_defaults=True) if store.state["revision"] else None
    payload = {"identity": identity(context, store), "goal": goal, "observation": observation["result"],
               "project_initial_observation": json.loads(store.path_for(store.state["initial_observation"]).read_text(encoding="utf-8")),
               "previous_tree": previous, "accepted_nodes": list(store.state["acceptances"]),
               "feedback": store.state["feedback"], "refinement_target": store.state.get("refinement_target", ""),
               "persona_context": req.persona_context, "answer": context.answer or req.answer,
               "latest_attempt": store.state["attempts"][-1] if store.state["attempts"] else None}
    latest_verifier = next((row for row in reversed(store.state["decisions"]) if row["role"] == "verifier"), None)
    if latest_verifier:
        decision = json.loads(store.path_for(latest_verifier["path"] + "/input.json").read_text(encoding="utf-8"))
        selected = json.loads(decision["messages"][1]["content"])
        relative = selected["original_receipt"]
        proof = json.loads(store.path_for(relative).read_text(encoding="utf-8"))
        payload["latest_verification"] = {"identity": proof["identity"], "checks": proof["checks"],
            "review": proof["review"], "specification": selected["specification"], "reads": scoped_reads(proof["reads"]),
            "original_receipt": relative, "original_digest": digest(proof)}
    if store.state.get("planning_survey"):
        survey = json.loads(store.path_for(store.state["planning_survey"]).read_text(encoding="utf-8"))
        payload["previous_site_survey"] = {"identity": survey["identity"], "original_receipt": store.state["planning_survey"],
                                           "original_digest": digest(survey)}
        payload["previous_site_survey"]["recipes"] = scoped_reads(survey.get("recipes", []))
        # Prefer the latest verifier's exact scope during repair. The earlier
        # site survey stays retained; every revised plan receives a new survey.
        if not latest_verifier:
            payload["previous_site_survey"]["reads"] = scoped_reads(survey["reads"])
    rejected = store.state.get("rejected_proposal")
    if rejected:
        payload["rejected_proposal"] = json.loads(store.path_for(rejected).read_text(encoding="utf-8"))
    if store.state.get("requested_survey"):
        relative = store.state["requested_survey"]
        observed = json.loads(store.path_for(relative).read_text(encoding="utf-8"))
        payload["requested_world_survey"] = {**observed, "reads": scoped_reads(observed["reads"]),
                                             "original_receipt": relative, "original_digest": digest(observed)}
    if req.proposed_tree is not None and previous is None and not store.state.get("source_proposal_used"):
        store.state["source_proposal_used"] = True
        store.save()
        proposed = req.proposed_tree
    else:
        try:
            proposal = reason(context, store, "planner", PlanProposal, payload)
        except ValidationError as error:
            store.state.update(feedback="The proposal violates its JSON contract: " + str(error),
                               rejected_proposal=store.state["decisions"][-1]["path"] + "/raw.json")
            store.save()
            return {"status": "invalid", "summary": store.state["feedback"]}
        if proposal.status == "survey":
            store.state["pending_survey"] = {"instruction_revision": current["revision"],
                                              "queries": [row.model_dump(mode="json") for row in proposal.queries]}
            store.save()
        if proposal.status != "planned":
            return {"status": proposal.status, "summary": proposal.summary}
        if proposal.tree is None:
            raise ValueError("A planned result needs a task tree.")
        proposed = proposal.tree
    target = store.state.get("refinement_target", "")
    # A later user instruction supersedes a pending scope-only refinement.
    if store.state["instruction_revision"] != current["revision"]:
        target = ""
    try:
        store.revise(proposed, target=target)
    except ValueError as error:
        rejection = store.state.get("rejections", 0) + 1
        relative = f"planning/rejected/{rejection:04d}.json"
        store.write(relative, proposed.model_dump(mode="json"))
        store.state.update(feedback="The proposed tree was rejected before dispatch: " + str(error),
                           rejected_proposal=relative, rejections=rejection)
        store.save()
        return {"status": "invalid", "summary": store.state["feedback"]}
    store.state.update(instruction_revision=current["revision"], refinement_target="", goal=goal, rejected_proposal="")
    store.save()
    return {"status": "planned", "summary": f"Retained specification revision {store.state['revision']}."}


def survey(context: RoleContext, store: TreeStore) -> dict:
    """Read model-selected world operands without dispatching an actor."""
    pending = store.state["pending_survey"]
    current = context.runtime.tools["managed_instructions"]()
    if pending["instruction_revision"] != current["revision"]:
        store.state.pop("pending_survey")
        store.save()
        return {"status": "replan", "summary": "Instructions changed before the requested world survey."}
    queries = [Condition.model_validate(row) for row in pending["queries"]]
    relative = f"planning/requested-surveys/{len(store.state['decisions']):04d}.json"
    reads = observe_conditions(context, queries)
    store.write(relative, {"identity": identity(context, store), "instruction_revision": current["revision"],
                           "queries": pending["queries"], "reads": reads,
                           "scope": "Planning observations only; these reads do not accept work."})
    store.state.update(requested_survey=relative)
    store.state.pop("pending_survey")
    store.save()
    return {"status": "observed", "summary": "Retained the requested world reads for planning."}


def review(context: RoleContext, store: TreeStore) -> dict:
    """Review acceptance coverage independently before selecting any leaf."""
    if not boundary(context, store):
        return {"status": "refine", "summary": "Project instructions changed before plan review."}
    tree = store.load()
    survey_path = f"planning/surveys/{len(store.state['decisions']) + 1:04d}.json"
    survey = {"identity": identity(context, store), "reads": observe_conditions(context, survey_conditions(tree.nodes))}
    # These are planning facts, not completion evidence. Query exact requested
    # inventory names so the reviewer does not invent crafting ratios.
    names = sorted({name.removeprefix("minecraft:") for node in tree.nodes for condition in node.criteria
                    if condition.kind == "inventory" for name in condition.names if not any(c in name for c in "*?[" )})
    survey["recipes"] = []
    for name in names:
        arguments = {"pattern": name}
        with BODY_LEASES.command(context.runtime.tools["minecraft_endpoint"](), context.runtime.stop_event):
            observed = context.runtime.tools["minecraft_call"]("minecraft_recipe_search", arguments)
        survey["recipes"].append({"tool": "minecraft_recipe_search", "arguments": arguments,
                                  "time": time.time(), "ok": observed["ok"], "result": observed["result"]})
    store.write(survey_path, survey)
    store.state["planning_survey"] = survey_path
    store.save()
    result = reason(context, store, "reviewer", PlanReview, {
        "identity": identity(context, store), "goal": store.state["goal"], "tree": tree.model_dump(mode="json"),
        "leaf_prerequisites": {key: sorted(value) for key, value in validate_tree(tree).items()},
        "accepted_nodes": list(store.state["acceptances"]),
        "project_initial_observation": json.loads(store.path_for(store.state["initial_observation"]).read_text(encoding="utf-8")),
        "original_observation": json.loads(store.path_for(store.state["planning_observation"]).read_text(encoding="utf-8")),
        "observation_path": store.state["planning_observation"],
        "current_site_survey": {"identity": survey["identity"], "reads": scoped_reads(survey["reads"]),
                                "recipes": scoped_reads(survey["recipes"])},
        "survey_path": survey_path})
    if not boundary(context, store):
        return {"status": "refine", "summary": "Project instructions changed during plan review."}
    if result.status == "refine":
        if result.target_id not in {node.id for node in tree.nodes}:
            raise ValueError("The reviewer named an unknown refinement target.")
        store.state.update(feedback=result.summary, refinement_target=result.target_id)
    elif result.status == "accepted":
        store.state["reviewed_revision"] = tree.revision
    elif result.status == "revise":
        store.state.update(feedback=result.summary, refinement_target="")
    store.save()
    return {"status": result.status, "summary": result.summary}


def check_assumptions(context: RoleContext, store: TreeStore, node: Node) -> bool:
    """Retain each current assumption read before it can release world work."""
    reads = observe_conditions(context, node.assumptions)
    checks = evaluate(node.assumptions, reads, certificates=store.state["acceptances"])
    number = store.state.get("assumption_checks", 0) + 1
    store.write(f"planning/assumptions/{number:04d}-{node.id}.json", {
        "identity": identity(context, store, node=node), "reads": reads, "checks": checks})
    store.state["assumption_checks"] = number
    store.save()
    return all(checks.values())


def select(context: RoleContext, store: TreeStore) -> dict:
    """Select only an independently reviewed leaf with valid prerequisites."""
    if not boundary(context, store):
        return {"status": "replan", "summary": "Project instructions changed before leaf selection."}
    if store.state.get("reviewed_revision") != store.state["revision"]:
        raise ValueError("The current tree has no independent scope review.")
    invalid = store.invalidate()
    for node in store.load().nodes:
        if store.valid(node.id) and node.assumptions:
            if not check_assumptions(context, store, node):
                del store.state["acceptances"][node.id]
                invalid.append(node.id)
    invalid.extend(store.invalidate())
    store.save()
    if invalid:
        return {"status": "replan", "summary": "Acceptance evidence or current assumptions changed: " + ", ".join(invalid)}
    aggregates = store.ready_aggregates()
    eligible = store.eligible()
    if aggregates or eligible:
        key = (aggregates or eligible)[0]
        nodes = {node.id: node for node in store.load().nodes}
        node, cursor = nodes[key], key
        while cursor is not None:
            assumptions = nodes[cursor].assumptions
            if assumptions and not check_assumptions(context, store, nodes[cursor]):
                return {"status": "replan", "summary": f"Current assumptions failed for {cursor} before dispatch."}
            cursor = nodes[cursor].parent
        actor_attempt = ""
        if aggregates:
            actor_attempt = next(row["id"] for row in reversed(store.state["attempts"])
                                 if row.get("state") == "complete" and row.get("report"))
        store.state["selection"] = identity(context, store, node=node, actor_attempt=actor_attempt)
        store.save()
        if aggregates:
            return {"status": "aggregate", "summary": f"Verify aggregate {key} before releasing dependent work."}
        return {"status": "selected", "summary": f"Selected verified-prerequisite leaf {key}."}
    if all(store.valid(node.id) for node in store.load().nodes):
        return {"status": "complete", "summary": "All leaf, story, and goal certificates passed."}
    return {"status": "waiting", "summary": "No leaf has valid prerequisite evidence. Inspect the retained task tree."}


def act(context: RoleContext, store: TreeStore) -> dict:
    """Dispatch a fresh native actor child for the exact selected revision."""
    if not boundary(context, store):
        return {"status": "replan", "summary": "Project instructions changed before specialist dispatch."}
    selected = store.state["selection"]
    node = next(node for node in store.load().nodes if node.id == selected["leaf_id"])
    if node.id not in store.eligible():
        raise ValueError("The selected leaf lost its prerequisite acceptance.")
    attempt_id = f"actor-{len(store.state['attempts']) + 1:04d}"
    child_id = context.task_id + "-" + attempt_id
    relative = "attempts/" + attempt_id
    prerequisites = store.requirements(node.id)
    attempt = {**selected, "actor_attempt": attempt_id, "id": attempt_id, "path": relative, "started": time.time(), "state": "running",
               "prerequisites": {key: {"verification": store.state["acceptances"][key]["verification"],
                                       "certificate_digest": digest(store.state["acceptances"][key])}
                                 for key in prerequisites}}
    store.state["attempts"].append(attempt)
    store.state["selection"].update(actor_attempt=attempt_id)
    store.save()
    req = request(context)
    child_environment = FolderEnvironment(store.root / relative, "bash")
    parent_boundary = context.runtime.tools["managed_instructions"]
    parent_revision = store.state["instruction_revision"]
    def instructions(wait=False, close=False):
        current = parent_boundary()
        if current["revision"] != parent_revision:
            raise InstructionsChanged("The project instruction changed. Replan from retained world effects.")
        return {**current, "instruction": node.instruction}
    instructions.authorize = getattr(parent_boundary, "authorize", None)
    instructions.cancelled = getattr(parent_boundary, "cancelled", None)
    endpoint = context.runtime.tools["minecraft_endpoint"]()
    tools = {**context.runtime.tools, "managed_instructions": instructions,
             "minecraft_lease": lambda: BODY_LEASES.acquire(endpoint, child_id)}
    child_stop = LinkedStopEvent(context.runtime.stop_event)
    child_runtime = replace(context.runtime, environment=child_environment, tools=tools, stop_event=child_stop)
    nodes = {row.id: row for row in store.load().nodes}
    prerequisite_context = {}
    for key, certificate in attempt["prerequisites"].items():
        verified = json.loads(store.path_for(certificate["verification"]).read_text(encoding="utf-8"))
        prerequisite_context[key] = {**certificate, "specification": nodes[key].model_dump(mode="json"),
                                     "identity": verified["identity"], "checks": verified["checks"]}
        source_actor = next(row for row in store.state["attempts"] if row["id"] == verified["identity"]["actor_attempt"])
        source_report = yaml.safe_load(store.path_for(source_actor["report"]).read_text(encoding="utf-8"))
        prerequisite_context[key]["actor_summary"] = source_report["summary"]
    ancestors, cursor = [], node.parent
    while cursor is not None:
        ancestors.append(nodes[cursor].model_dump(mode="json"))
        cursor = nodes[cursor].parent
    acceptance = {"criteria": [row.model_dump(mode="json") for row in node.criteria],
                  "prerequisites": prerequisite_context,
                  "project": {"goal": store.state["goal"], "ancestor_specifications": ancestors}}
    earlier = [row for row in store.state["attempts"][:-1] if row["leaf_id"] == node.id]
    if earlier:
        previous = earlier[-1]
        handoff = store.path_for(previous["path"] + "/outbox/handoff.md")
        acceptance["previous_attempt"] = {"identity": previous, "handoff": handoff.read_text(encoding="utf-8") if handoff.is_file() else "",
                                          "journals": [str(store.path_for(row["path"] + "/scratch/actor-journal.jsonl")) for row in earlier],
                                          "scope": "Past work; current specification and fresh observations take precedence."}
    payload = MinecraftRequest(goal=node.instruction, instruction=node.instruction, body_id=req.body_id,
        specialist=node.specialist, persona_context=req.persona_context, include_image=req.include_image,
        acceptance=acceptance,
        task_reference={**selected, "actor_attempt": attempt_id}).model_dump(mode="json")
    runner = WorkflowRunner(ROOT / "workflows/minecraft/minecraft.workflow.md", store.root / "kernel-children", namespace="minecraft_leaf")
    try:
        result = runner.invoke(WorkflowRequest(child_id, payload), child_runtime)
    except InstructionsChanged as error:
        attempt.update(state="stale", summary=str(error))
        store.save()
        return {"status": "replan", "summary": str(error)}
    except (ValidationError, json.JSONDecodeError) as error:
        # Native output is untrusted. Cleanup has finished before invoke raises.
        # Keep the rejected candidate and world calls; never invent missing data.
        attempt.update(state="invalid", summary=f"{type(error).__name__}: {error}",
                       journal=relative + "/scratch/actor-journal.jsonl")
        store.save()
        return {"status": "replan", "summary": "The native completion contract was rejected. "
                "Obtain fresh world state and preserve prior effects before a new attempt. " + attempt["summary"]}
    context.runtime.check_cancelled()
    if result.status != WorkflowStatus.Completed:
        journal_path = relative + "/scratch/actor-journal.jsonl"
        rows = [json.loads(line) for line in store.path_for(journal_path).read_text(encoding="utf-8").splitlines()] if store.path_for(journal_path).is_file() else []
        errors = [row for row in rows if row["event"] == "uncertain"]
        attempt.update(state="uncertain" if errors else "failed", summary=str(result.kernel_summary), journal=journal_path, errors=errors)
        store.save()
        detail = errors[-1]["error"] if errors else str(result.kernel_summary)
        return {"status": "replan", "summary": "The actor stopped without acceptance. Obtain fresh world state before a new method. " + detail}
    report = result.report
    attempt.update(state=report["outcome"]["status"], summary=report["summary"],
                   session=report["agent_session"], report=relative + "/outbox/report.yaml",
                   journal=relative + "/scratch/actor-journal.jsonl")
    final = report["final_observation"]
    attempt["progress_digest"] = digest(progress_facts(final))
    store.save()
    from workflows.minecraft.steps import build_world
    req.goal = store.state["goal"]
    world = build_world(req, {"result": report["final_observation"]}, None, [])
    world["outcome"] = "progress"
    world["episodes"][0].update(goal=node.instruction, outcome="unverified", source_attempt=attempt_id)
    context.runtime.environment.write_text("outbox/world.yaml", yaml.safe_dump(world, allow_unicode=True, sort_keys=False))
    if not boundary(context, store):
        return {"status": "replan", "summary": "The project changed before actor completion was accepted."}
    routes = {"complete": "candidate", "yield": "progress", "replan": "replan", "blocked": "waiting"}
    prior = [row for row in store.state["attempts"][:-1] if row["leaf_id"] == node.id and row.get("progress_digest")]
    if report["outcome"]["status"] == "yield" and prior and prior[-1]["progress_digest"] == attempt["progress_digest"]:
        return {"status": "replan", "summary": "The repeated leaf attempt made no observed progress. Change the method."}
    return {"status": routes[report["outcome"]["status"]], "summary": report["summary"]}


def verify_node(context: RoleContext, store: TreeStore, node: Node, actor_attempt: str) -> dict:
    """Require original actor provenance and fresh independent world reads."""
    verifier_id = f"verifier-{len(store.state['decisions']) + 1:04d}-{node.id}"
    ident = identity(context, store, node=node, actor_attempt=actor_attempt, verifier_attempt=verifier_id)
    reads = observe_conditions(context, node.criteria)
    checks = evaluate(node.criteria, reads, certificates=store.state["acceptances"])
    relative = f"verification/{verifier_id}.json"
    # Persist original reads before inference so an interruption retains evidence.
    store.write(relative, {"schema": "pm.minecraft-leaf-verification.v1", "identity": ident,
                           "verified": False, "reads": reads, "checks": checks, "review": {"status": "pending"}})
    review_result = reason(context, store, "verifier", VerificationReview, {
        "identity": ident, "specification": node.model_dump(mode="json"), "reads": scoped_reads(reads), "checks": checks,
        "original_receipt": relative})
    complete_ids = {row.id for row in node.criteria}
    verified = (all(checks.values()) and review_result.status == "accepted"
                and set(review_result.criterion_ids) == complete_ids and boundary(context, store))
    files = []
    if node.kind == "leaf":
        attempt = next(row for row in store.state["attempts"] if row["id"] == actor_attempt)
        actor = yaml.safe_load((store.root / attempt["report"]).read_text(encoding="utf-8"))
        journal = [json.loads(line) for line in (store.root / attempt["journal"]).read_text(encoding="utf-8").splitlines()]
        dispatches = {row["seq"]: row for row in journal if row["event"] == "dispatch"}
        results = [row for row in journal if row["event"] == "result"]
        verified = (verified and actor["outcome"]["status"] == "complete" and bool(results)
                    and actor["task_reference"] == {**store.state["selection"], "actor_attempt": actor_attempt}
                    and all(row["seq"] in dispatches and dispatches[row["seq"]]["tool"] == row["tool"]
                            and dispatches[row["seq"]]["arguments"] == row["arguments"] for row in results)
                    and len({row["seq"] for row in results}) == len(results)
                    and len(dispatches) == len(results))
        files = [attempt["report"], attempt["journal"]]
    store.write(relative, {"schema": "pm.minecraft-leaf-verification.v1", "identity": ident,
        "verified": verified, "reads": reads, "checks": checks, "review": review_result.model_dump(mode="json")})
    if verified:
        store.accept(node.id, relative, files)
        return {"status": "accepted", "summary": review_result.summary}
    return {"status": "replan" if review_result.status == "accepted" else review_result.status,
            "summary": "Independent verification failed: " + review_result.summary}


def verify(context: RoleContext, store: TreeStore) -> dict:
    """Verify the exact completion candidate before releasing dependencies."""
    selected = store.state["selection"]
    node = next(node for node in store.load().nodes if node.id == selected["leaf_id"])
    return verify_node(context, store, node, selected["actor_attempt"])


def finish(context: RoleContext, store: TreeStore) -> dict:
    """Verify aggregate usability and close instructions before reporting success."""
    req = request(context)
    if not boundary(context, store):
        return {"status": "replan", "summary": "Project instructions changed before final verification."}
    if not req.plan_only:
        if not all(store.valid(node.id) for node in store.load().nodes):
            raise ValueError("Final publication requires verified leaf, story, and goal certificates.")
    final = context.runtime.tools["managed_instructions"](close=True)
    if final["revision"] != store.state["instruction_revision"]:
        return {"status": "replan", "summary": "The final instruction boundary changed the project."}
    evidence = verified_project(store.root) if not req.plan_only else {"verified": False}
    if not req.plan_only and not evidence["verified"]:
        raise ValueError("The completed project failed retained-evidence recomputation.")
    if not req.plan_only:
        from workflows.minecraft.steps import build_world
        with BODY_LEASES.command(context.runtime.tools["minecraft_endpoint"](), context.runtime.stop_event):
            observation = context.runtime.tools["minecraft_call"]("minecraft_observe", {"include_image": False})
        if not observation["ok"]:
            raise ValueError("The project has no usable final body observation.")
        req.goal = store.state["goal"]
        world = build_world(req, observation, None, [])
        world["outcome"] = world["episodes"][0]["outcome"] = "success"
        world["verification"] = evidence
        world["episodes"][0]["source_attempts"] = [row["id"] for row in store.state["attempts"]]
        context.runtime.environment.write_text("outbox/world.yaml", yaml.safe_dump(world, allow_unicode=True, sort_keys=False))
    report = {"schema": "pm.minecraft-project-report.v1", "accepted": True, "verified": evidence["verified"],
              "summary": "Reviewed plan retained without execution." if req.plan_only else "The project passed independent world verification.",
              "goal": store.state["goal"], "tree_revision": store.state["revision"], "verification": evidence,
              "plan_only": req.plan_only, "attempts": store.state["attempts"], "body_id": req.body_id}
    context.runtime.environment.write_text("outbox/report.yaml", yaml.safe_dump(report, allow_unicode=True, sort_keys=False))
    return {"status": "done", "summary": report["summary"]}


def _run(context: RoleContext) -> dict:
    """Route one declared role and retain its explicit failure or feedback."""
    store = store_for(context)
    if context.phase == "replan":
        return plan(context, store)
    if context.phase == "review":
        result = review(context, store)
        if result["status"] == "accepted" and request(context).plan_only:
            result["status"] = "inspect"
        return result
    operation = {"plan": plan, "survey": survey, "select": select, "act": act, "verify": verify, "finish": finish}[context.phase]
    result = operation(context, store)
    if result["status"] in {"replan", "waiting", "blocked"}:
        store.state["feedback"] = result["summary"]
        store.save()
    return result


def run(context: RoleContext) -> dict:
    """Run a role without the outer runner replaying it for a file lock.

    Local evidence commits retry only their file operation. Other persistent
    permission errors fail the project. Its original world receipts remain.
    """
    try:
        return _run(context)
    except PermissionError as error:
        raise RuntimeError(f"A11 file access failed in {context.phase}. Do not replay world work: {error}") from error
