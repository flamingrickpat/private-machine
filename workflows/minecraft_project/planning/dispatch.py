"""Act role: run one specialist session for the selected leaf and record its world effects."""
from __future__ import annotations
from workflows.minecraft_project.schemas import ProjectPhaseResult
from workflows.minecraft.runtime import require_minecraft_runtime
from pm.runs.managed_boundary import BoundaryReadSettings

from dataclasses import replace
from pm.runs.role_runtime import PMWorkflowRuntime
from pm.runs.managed_boundary import LeafBoundarySettings
from workflows.minecraft.runtime import MinecraftBodyLease
from workflows.minecraft.execution_contracts import ActorAcceptance, CriterionObservations, PrerequisiteEvidence, LeafProject, PreviousAttempt
from workflows.minecraft_project.planning.context_contracts import DecisionIdentity, AttemptReceipt, ProjectCheck, RecordedRead
from workflows.minecraft_project.planning.context_sources import decision_identity, decision_intent, read_gameplay_history, scoped_planning_history
from workflows.minecraft_project.planning.context_selection import select_recorded_reads
from workflows.minecraft.rendering import render_gameplay_archive
from pm.llm.role_model import role_model
from pm.context_render.prompt import RendererCapture, RendererInput
import json
import time

from pm.utils import yaml_util
from pm_workflows.python_role import RoleContext
from pydantic import ValidationError

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
from workflows.minecraft_project.planning.constraints import caller_plan_constraints, project_check_context


def execution_observations(context: RoleContext, store: TreeStore, node: Node, attempt: AttemptReceipt) -> CriterionObservations:
    reads = observe_conditions(context, node.criteria)
    checks = evaluate(node.criteria, reads, certificates=store.state["acceptances"])
    original = CriterionObservations(identity=decision_identity(context, store, node=node, actor_attempt=attempt.id),
        scope="Current read-only predicates. Required actor effects, ordered travel, durable notes, and independent acceptance remain separate.",
        reads=tuple(RecordedRead.model_validate(row) for row in reads), checks=checks,
        original_receipt="inbox/criterion-observations.json", original_digest="")
    relative = attempt.path + "/inbox/criterion-observations.json"
    store.write(relative, original)
    selected = original.model_copy(deep=True)
    selected.reads = select_recorded_reads(original.reads)
    selected.original_digest = digest(original)
    return selected


def act(context: RoleContext, store: TreeStore) -> ProjectPhaseResult:
    """Dispatch a fresh native actor child for the exact selected revision."""
    if not boundary(context, store):
        return ProjectPhaseResult(status="replan", summary="Project instructions changed before specialist dispatch.")
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
    runtime = require_minecraft_runtime(context.runtime)
    if runtime.managed_boundary is None:
        raise TypeError("Minecraft actor dispatch requires a managed PMWorkflowRuntime.")
    parent_revision = store.state["instruction_revision"]
    child_boundary = runtime.managed_boundary.for_leaf(LeafBoundarySettings(node.instruction, parent_revision))
    endpoint = require_minecraft_runtime(runtime).minecraft_transport.settings.endpoint
    lease = runtime.body_lease if runtime.body_lease.ownership is not None else MinecraftBodyLease(endpoint, child_id, leases=runtime.minecraft_transport.leases)
    child_stop = LinkedStopEvent(runtime.stop_event)
    child_runtime = replace(runtime, environment=child_environment, stop_event=child_stop,
        managed_boundary=child_boundary, body_lease=lease)
    nodes = {row.id: row for row in store.load().nodes}
    prerequisite_context: dict[str, PrerequisiteEvidence] = {}
    for key, certificate in attempt["prerequisites"].items():
        verified = json.loads(store.path_for(certificate["verification"]).read_text(encoding="utf-8"))
        source_actor = next(row for row in store.state["attempts"] if row["id"] == verified["identity"]["actor_attempt"])
        source_report = yaml_util.safe_load(store.path_for(source_actor["report"]).read_text(encoding="utf-8"))
        prerequisite_context[key] = PrerequisiteEvidence(verification=certificate["verification"],
            certificate_digest=certificate["certificate_digest"], specification=nodes[key],
            identity=DecisionIdentity.model_validate(verified["identity"]), checks=verified["checks"], actor_summary=source_report["summary"])
    ancestors: list[Node] = []
    cursor = node.parent
    while cursor is not None:
        ancestors.append(nodes[cursor])
        cursor = nodes[cursor].parent
    history = read_gameplay_history(store)
    session_model = role_model(runtime)
    render_settings = session_model.render_settings()
    with RendererCapture(session_model.capture_settings("minecraft.gameplay_archive"),
            RendererInput("minecraft.gameplay_archive", history, render_settings)) as capture:
        history_prompt = render_gameplay_archive(history, render_settings)
        capture.save_output(history_prompt)
    store.write(relative + "/scratch/original-gameplay-history.json", history)
    child_environment.write_text("inbox/gameplay-history.md", history_prompt.turns[1].content.text)
    check = project_check_context(store)
    project = LeafProject(goal=store.state["goal"], ancestor_specifications=tuple(ancestors),
        caller_constraints=caller_plan_constraints(store), latest_project_check=ProjectCheck.model_validate(check) if check is not None else None)
    acceptance = ActorAcceptance(criteria=tuple(node.criteria),
        current_criterion_observations=execution_observations(context, store, node, AttemptReceipt.model_validate(attempt)),
        prerequisites=prerequisite_context, project=project,
        gameplay_history=scoped_planning_history(store, dependency_scope(store.load(), [node.id]), history, {node.id}),
        full_gameplay_history="inbox/gameplay-history.md",
        history_journals=tuple(str(store.path_for(row["path"] + "/scratch/actor-journal.jsonl")) for row in store.state["attempts"][:-1]))
    earlier = [row for row in store.state["attempts"][:-1] if row["leaf_id"] == node.id]
    if earlier:
        previous = earlier[-1]
        handoff = store.path_for(previous["path"] + "/outbox/handoff.md")
        acceptance.previous_attempt = PreviousAttempt(identity=AttemptReceipt.model_validate(previous),
            handoff=handoff.read_text(encoding="utf-8") if handoff.is_file() else "",
            journals=tuple(str(store.path_for(row["path"] + "/scratch/actor-journal.jsonl")) for row in earlier),
            scope="Past work; current specification and fresh observations take precedence.")
    actor_identity = DecisionIdentity.model_validate(selected)
    actor_identity.actor_attempt = attempt_id
    payload = MinecraftRequest(goal=node.instruction, instruction=node.instruction, body_id=req.body_id,
        project_intent=decision_intent(context, store), specialist=node.specialist,
        persona_context=req.persona_context, include_image=req.include_image,
        acceptance=acceptance, task_reference=actor_identity).model_dump(mode="json")
    runner = WorkflowRunner(ROOT / "workflows/minecraft/minecraft.workflow.md", store.root / "kernel-children", namespace="minecraft_leaf")
    try:
        result = runner.invoke(WorkflowRequest(child_id, payload), child_runtime)
    except InstructionsChanged as error:
        attempt.update(state="stale", summary=str(error))
        store.save()
        return ProjectPhaseResult(status="replan", summary=str(error))
    except (ValidationError, json.JSONDecodeError) as error:
        # Native output is untrusted. Cleanup has finished before invoke raises.
        # Keep the rejected candidate and world calls; never invent missing data.
        attempt.update(state="invalid", summary=f"{type(error).__name__}: {error}",
                       journal=relative + "/scratch/actor-journal.jsonl")
        store.save()
        return ProjectPhaseResult(status="replan", summary="The native completion contract was rejected. "
                "Obtain fresh world state and preserve prior effects before a new attempt. " + attempt["summary"])
    context.runtime.check_cancelled()
    if result.status != WorkflowStatus.Completed:
        journal_path = relative + "/scratch/actor-journal.jsonl"
        rows = [json.loads(line) for line in store.path_for(journal_path).read_text(encoding="utf-8").splitlines()] if store.path_for(journal_path).is_file() else []
        errors = [row for row in rows if row["event"] == "uncertain"]
        attempt.update(state="uncertain" if errors else "failed", summary=str(result.kernel_summary), journal=journal_path, errors=errors)
        store.save()
        detail = errors[-1]["error"] if errors else str(result.kernel_summary)
        return ProjectPhaseResult(status="waiting" if errors else "replan", summary="The actor stopped without acceptance. Restore the service, then observe retained effects before a new method. " + detail)
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
    context.runtime.environment.write_text("outbox/world.yaml", yaml_util.safe_dump(world, allow_unicode=True, sort_keys=False))
    runtime.managed_boundary.milestone(attempt_id, report["summary"], attempt["journal"])
    if not boundary(context, store):
        return ProjectPhaseResult(status="replan", summary="The project changed before actor completion was accepted.")
    routes = {"complete": "candidate", "yield": "progress", "replan": "replan", "blocked": "waiting"}
    prior = [row for row in store.state["attempts"][:-1] if row["leaf_id"] == node.id and row.get("progress_digest")]
    if report["outcome"]["status"] == "yield" and prior and prior[-1]["progress_digest"] == attempt["progress_digest"]:
        return ProjectPhaseResult(status="replan", summary="The repeated leaf attempt made no observed progress. Change the method.")
    return ProjectPhaseResult(status=routes[report["outcome"]["status"]], summary=report["summary"])

