"""Retention role: keep valid accepted work across a revised plan."""
from __future__ import annotations
from workflows.minecraft_project.schemas import ProjectPhaseResult
from workflows.minecraft.runtime import require_minecraft_runtime
from pm.runs.managed_boundary import BoundaryReadSettings

from pm_workflows.python_role import RoleContext

from workflows.minecraft_body.lease import BODY_LEASES
from workflows.minecraft_project.schemas import RetentionChoice
from workflows.minecraft_project.tree_store import TreeStore
from workflows.minecraft_project.evidence_selection import retention_operation, retention_tree
from workflows.minecraft_project.planning.run_context import boundary, identity, request
from workflows.minecraft_project.planning.decisions import reason
from workflows.minecraft_project.planning.context_contracts import RetentionContext, RetentionCandidate, DecisionRole
from workflows.minecraft_project.planning.context_sources import decision_identity


def retain(context: RoleContext, store: TreeStore, observation: dict | None = None) -> ProjectPhaseResult:
    """Run a separate native proof decision before a material actor is eligible."""
    if not store.state["revision"] or not store.acceptance_candidates():
        return ProjectPhaseResult(status="unchanged", summary="No historical specification restoration is pending.")
    if observation is None:
        with BODY_LEASES.command(require_minecraft_runtime(context.runtime).minecraft_transport.settings.endpoint, context.runtime.stop_event):
            reply = require_minecraft_runtime(context.runtime).minecraft_transport.call("minecraft_observe", {"include_image": False})
        if not reply["ok"]:
            return ProjectPhaseResult(status="waiting", summary="Historical restoration needs an identified current body observation.")
        observation = reply["result"]
    tree = store.load()
    candidates = store.acceptance_candidates()
    requested_ids = {row["specification"]["id"] for row in candidates}
    typed_candidates: list[RetentionCandidate] = []
    for row in candidates:
        candidate = RetentionCandidate.model_validate(row)
        candidate.currently_valid = store.valid(candidate.specification.id)
        typed_candidates.append(candidate)
    payload = RetentionContext(identity=decision_identity(context, store),
        goal=request(context).goal or request(context).instruction, observation=observation,
        feedback=store.state.get("retention_feedback", ""),
        current_nodes=tuple(node for node in tree.nodes if node.id in requested_ids), candidates=tuple(typed_candidates))
    choice = reason(context, store, DecisionRole.Retention, RetentionChoice, payload)
    if choice.status != "planned":
        return ProjectPhaseResult(status=choice.status, summary=choice.summary)
    try:
        proposal = retention_operation(store, choice)
        store.write(store.state["decisions"][-1]["path"] + "/operation.json", proposal.model_dump(mode="json"))
        proposed = retention_tree(store, proposal)
        if not boundary(context, store):
            return ProjectPhaseResult(status="unchanged", summary="Project instructions changed before historical restoration.")
        store.revise(proposed)
    except ValueError as error:
        store.state["retention_feedback"] = str(error)
        store.save()
        return ProjectPhaseResult(status="invalid", summary="Native historical restoration was rejected: " + str(error))
    current = require_minecraft_runtime(context.runtime).instructions()
    store.state.update(instruction_revision=current["revision"],
                       goal=current["instruction"] or request(context).goal or request(context).instruction,
                       renewal_nodes=list(dict.fromkeys([*store.state.get("renewal_nodes", []),
                                                         *[row.node.id for row in proposal.renewals]])), retention_feedback="", feedback="")
    store.state.pop("require_leaf_change", None)
    store.save()
    return ProjectPhaseResult(status="planned", summary=proposal.summary)

