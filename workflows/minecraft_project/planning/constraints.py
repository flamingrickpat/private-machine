"""Caller plan constraints and the project-check context that plan, review, and act share."""
from __future__ import annotations
from workflows.minecraft.runtime import require_minecraft_runtime
from pm.runs.managed_boundary import BoundaryReadSettings

import json

from pm_workflows.python_role import RoleContext

from workflows.minecraft_project.tree_store import TreeStore, digest
from workflows.minecraft_project.planning.run_context import identity
from workflows.minecraft_project.planning.context_contracts import CallerRequirements, CallerAuditResult


def caller_plan_constraints(store: TreeStore) -> CallerRequirements | None:
    """Read complete caller requirements with their original receipt digest.

    These requirements describe coverage and separate publication audits.
    They cannot establish world truth or accept a task. This read changes no
    task state and returns ``None`` when no caller contract was recorded.
    """
    relative = store.state.get("planning_constraints")
    if not relative:
        return None
    proof = json.loads(store.path_for(relative).read_text(encoding="utf-8"))
    result = CallerAuditResult.model_validate(proof["result"])
    return CallerRequirements(errors=result.errors, requirements=result.requirements, original_receipt=relative, original_digest=digest(proof))


def record_plan_constraints(context: RoleContext, store: TreeStore) -> CallerAuditResult | None:
    """Retain current caller operands before planning or independent review.

    The callback describes the plan contract and changes no world state.
    Keep prior receipts intact when the caller changes its requirements.
    Errors remain planning evidence; the reviewer decides whether to dispatch.
    """
    constraint = require_minecraft_runtime(context.runtime).project_audit
    if constraint is None:
        return None
    result = constraint.plan_constraints(store.load())
    number = store.state.get("constraint_reviews", 0) + 1
    relative = f"planning/constraints/{number:04d}.json"
    store.write(relative, {"identity": identity(context, store), "tree_digest": digest(store.load().model_dump(mode="json")),
                           "result": result.model_dump(mode="json")})
    store.state.update(constraint_reviews=number, planning_constraints=relative)
    store.save()
    return result


def project_check_context(store: TreeStore) -> dict | None:
    """Keep the latest publication check explicit across planning retries.

    Schema errors can replace feedback text, but cannot erase the separate
    publication failure. Detailed effects remain in the original receipt.
    This read changes no task state and cannot accept a completion candidate.
    """
    relative = store.state.get("project_check")
    if not relative:
        return None
    proof = json.loads(store.path_for(relative).read_text(encoding="utf-8"))
    return {**{key: proof["result"][key] for key in ("errors", "checks", "requirements") if key in proof["result"]},
            "identity": proof["identity"], "original_receipt": relative, "original_digest": digest(proof)}


def leaf_project_context(store: TreeStore, ancestors: list[dict]) -> dict:
    """Give private actors the caller requirements that planners received.

    These operands explain publication audits and their evidence needs.
    They do not accept work or extend the selected leaf's instruction.
    Keep their original receipt and digest beside the supplied facts.
    """
    project = {"goal": store.state["goal"], "ancestor_specifications": ancestors}
    constraints = caller_plan_constraints(store)
    if constraints is not None:
        project["caller_constraints"] = constraints
    check = project_check_context(store)
    if check is not None:
        project["latest_project_check"] = check
    return project

