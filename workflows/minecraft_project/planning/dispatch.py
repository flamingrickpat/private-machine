"""Act role: run one specialist session for the selected leaf and record its world effects."""
from __future__ import annotations

from dataclasses import replace
import json
import time

import yaml
from pm_workflows.python_role import RoleContext
from pydantic import ValidationError

from workflows.minecraft_body.lease import BODY_LEASES
from pm.host.cancellation import LinkedStopEvent
from pm.runs.workflow_runner import WorkflowRequest, WorkflowRunner, WorkflowStatus
from pm.host.folder_target import FolderEnvironment
from workflows.minecraft.schemas import MinecraftRequest
from workflows.minecraft_project.schemas import Node
from workflows.minecraft_project.tree_store import TreeStore, digest
from workflows.minecraft_project.acceptance import evaluate, observe_conditions
from workflows.minecraft_project.evidence_selection import dependency_scope, scoped_history
from workflows.minecraft_project.planning.run_context import InstructionsChanged, ROOT, boundary, identity, progress_facts, project_intent, request, scoped_reads
from workflows.minecraft_project.history_encoding import gameplay_history
from workflows.minecraft_project.planning.constraints import leaf_project_context


def execution_observations(context: RoleContext, store: TreeStore, node: Node, attempt: dict) -> dict:
    """Give one native actor original current reads for its exact predicates.

    The reads use the normal serialized, cancellable world boundary. They
    change no world state and cannot prove ordered effects or accept a task.
    Store the complete source inside the actor's readable inbox. Independent
    verification still obtains new reads after the completion candidate.
    """
    reads = observe_conditions(context, node.criteria)
    proof = {"schema": "pm.minecraft-execution-observations.v1",
             "identity": identity(context, store, node=node, actor_attempt=attempt["id"]),
             "scope": "Current read-only predicates. Required actor effects, ordered travel, durable notes, and independent acceptance remain separate.",
             "reads": reads, "checks": evaluate(node.criteria, reads, certificates=store.state["acceptances"])}
    relative = attempt["path"] + "/inbox/criterion-observations.json"
    store.write(relative, proof)
    return {**proof, "reads": scoped_reads(reads), "original_receipt": "inbox/criterion-observations.json",
            "original_digest": digest(proof)}


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
    history = gameplay_history(store)
    store.write(relative + "/inbox/gameplay-history.json", history)
    acceptance = {"criteria": [row.model_dump(mode="json") for row in node.criteria],
                  "current_criterion_observations": execution_observations(context, store, node, attempt),
                  "prerequisites": prerequisite_context,
                  "project": leaf_project_context(store, ancestors),
                  "gameplay_history": scoped_history(store, dependency_scope(store.load(), [node.id]), history, {node.id}),
                  "full_gameplay_history": "inbox/gameplay-history.json",
                  "history_journals": [str(store.path_for(row["path"] + "/scratch/actor-journal.jsonl"))
                                       for row in store.state["attempts"][:-1]]}
    earlier = [row for row in store.state["attempts"][:-1] if row["leaf_id"] == node.id]
    if earlier:
        previous = earlier[-1]
        handoff = store.path_for(previous["path"] + "/outbox/handoff.md")
        acceptance["previous_attempt"] = {"identity": previous, "handoff": handoff.read_text(encoding="utf-8") if handoff.is_file() else "",
                                          "journals": [str(store.path_for(row["path"] + "/scratch/actor-journal.jsonl")) for row in earlier],
                                          "scope": "Past work; current specification and fresh observations take precedence."}
    payload = MinecraftRequest(goal=node.instruction, instruction=node.instruction, body_id=req.body_id,
        project_intent=project_intent(context, store),
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
        return {"status": "waiting" if errors else "replan", "summary":
                "The actor stopped without acceptance. Restore the service, then observe retained effects before a new method. " + detail}
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
    world["episodes"][0]["gameplay"] = report.get("gameplay", {})
    context.runtime.environment.write_text("outbox/world.yaml", yaml.safe_dump(world, allow_unicode=True, sort_keys=False))
    publish = context.runtime.tools.get("managed_milestone")
    if publish is not None:
        publish(attempt_id, report["summary"], attempt["journal"])
    if not boundary(context, store):
        return {"status": "replan", "summary": "The project changed before actor completion was accepted."}
    routes = {"complete": "candidate", "yield": "progress", "replan": "replan", "blocked": "waiting"}
    prior = [row for row in store.state["attempts"][:-1] if row["leaf_id"] == node.id and row.get("progress_digest")]
    if report["outcome"]["status"] == "yield" and prior and prior[-1]["progress_digest"] == attempt["progress_digest"]:
        return {"status": "replan", "summary": "The repeated leaf attempt made no observed progress. Change the method."}
    return {"status": routes[report["outcome"]["status"]], "summary": report["summary"]}

