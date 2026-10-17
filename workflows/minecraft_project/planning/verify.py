"""Verifier role: mechanical condition evaluation plus a model review of one finished leaf."""
from __future__ import annotations

import json

from pm.utils import yaml_util
from pm_workflows.python_role import RoleContext
from pm.runs.review import PASSING_REVIEWS, review_passes, review_policy

from workflows.minecraft_project.schemas import Node, VerificationReview
from workflows.minecraft_project.tree_store import TreeStore
from workflows.minecraft_project.acceptance import evaluate, observe_conditions
from workflows.minecraft_project.planning.run_context import boundary, identity, project_intent, request, scoped_reads
from workflows.minecraft_project.planning.decisions import reason


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
        "project_intent": project_intent(context, store), "review_policy": review_policy(request(context).review_mode),
        "identity": ident, "specification": node.model_dump(mode="json"), "reads": scoped_reads(reads), "checks": checks,
        "original_receipt": relative})
    complete_ids = {row.id for row in node.criteria}
    reported_ids = list(review_result.criterion_ids)
    verified = (all(checks.values()) and review_passes(review_result.status, request(context).review_mode)
                and boundary(context, store))
    files = []
    if node.kind == "leaf":
        attempt = next(row for row in store.state["attempts"] if row["id"] == actor_attempt)
        actor = yaml_util.safe_load((store.root / attempt["report"]).read_text(encoding="utf-8"))
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
    # Coverage comes from original reads and backend checks. The model's ID
    # list is diagnostic; it must not make a valid result fail bookkeeping.
    review_result.criterion_ids = sorted(complete_ids) if verified else reported_ids
    recorded_review = review_result.model_dump(mode="json")
    if verified:
        recorded_review["status"] = "accepted"
    store.write(relative, {"schema": "pm.minecraft-leaf-verification.v1", "identity": ident,
        "verified": verified, "reads": reads, "checks": checks, "review": recorded_review,
        "review_judgment": review_result.status, "review_mode": request(context).review_mode,
        "reported_criterion_ids": reported_ids, "criterion_coverage_source": "backend_checks",
        "outcome": "completion_evidence" if verified else "unmet_completion_conditions"})
    if verified:
        store.accept(node.id, relative, files)
        return {"status": "accepted", "summary": review_result.summary}
    return {"status": "replan" if review_result.status in PASSING_REVIEWS else review_result.status,
            "summary": "Independent verification failed: " + review_result.summary}


def verify(context: RoleContext, store: TreeStore) -> dict:
    """Verify the exact completion candidate before releasing dependencies."""
    selected = store.state["selection"]
    node = next(node for node in store.load().nodes if node.id == selected["leaf_id"])
    return verify_node(context, store, node, selected["actor_attempt"])

