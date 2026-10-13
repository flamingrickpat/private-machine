"""Pure attention competition over scored codelet nominations.

The attention stage asks the model for scores only. This module chooses the
winners with fixed, inspectable rules, so a reader can see why one nomination
won and another waited. It never calls a model and never writes state.

Rules, in order:

1. A nomination that has no valid score is ``rejected`` with reason
   ``unscored``. It cannot win by default.
2. ``score`` is the weighted sum of the four 0..1 scores. Weights come from
   the per-loop ``AttentionPolicy``. In danger, urgency gets an extra bonus,
   so danger outranks contemplation.
3. Reserved codelets win first. The main policy reserves one place for
   ``direct_stimulus``: the companion always notices the message or event
   that started the cycle, even when it does not attend to every detail.
4. The remaining places go to the highest scores at or above ``min_score``,
   up to ``capacity``. Ties keep proposal order.
5. Every loser is ``deferred`` with reason ``capacity`` or ``below_min_score``
   and the score that it would have needed.
"""
from __future__ import annotations

from workflows.main_cognition.codelet_schedule import AttentionPolicy
from workflows.main_cognition.schemas import AttentionNomination, AttentionResult

SCORE_AXES = ("urgency", "novelty", "goal_relevance", "reportability")


def weighted_score(nomination: AttentionNomination, policy: AttentionPolicy, *, danger: bool) -> float:
    """Return the competition score of one scored nomination."""
    score = sum(float(policy.weights.get(axis, 0.0)) * float(getattr(nomination, axis)) for axis in SCORE_AXES)
    if danger:
        score += policy.danger_urgency_bonus * float(nomination.urgency)
    return round(score, 4)


def compete(nominations: list[AttentionNomination], policy: AttentionPolicy, *,
            danger: bool = False) -> AttentionResult:
    """Choose winners and give every nomination a status and a reason.

    The input nominations already carry their 0..1 axis scores. The result's
    ``selected_ids`` lists the target knoxel ids of the winners, in winning
    order and without duplicates. Workspace focus is limited to those ids.
    """
    scored: list[tuple[float, int, AttentionNomination]] = []
    for index, nomination in enumerate(nominations):
        if nomination.status == "rejected":
            continue
        raw = weighted_score(nomination, policy, danger=danger)
        # AttentionRecord stores score in 0..1. The raw score stays in the trace.
        nomination.score = min(1.0, raw)
        scored.append((raw, index, nomination))
    order = sorted(scored, key=lambda item: (-item[0], item[1]))
    winners: list[AttentionNomination] = []
    reserved_used: dict[str, int] = {}
    for raw, _, nomination in order:
        allowance = policy.reserved_codelets.get(nomination.codelet, 0)
        if allowance and reserved_used.get(nomination.codelet, 0) < allowance and len(winners) < policy.capacity:
            reserved_used[nomination.codelet] = reserved_used.get(nomination.codelet, 0) + 1
            nomination.status, nomination.reason = "selected", f"reserved place for {nomination.codelet}; score {raw:g}"
            winners.append(nomination)
    for raw, _, nomination in order:
        if nomination.status == "selected":
            continue
        if raw < policy.min_score:
            nomination.status = "deferred"
            nomination.reason = f"below_min_score: {raw:g} < {policy.min_score:g}"
        elif len(winners) >= policy.capacity:
            weakest = min((float(item.score) for item in winners), default=0.0)
            nomination.status = "deferred"
            nomination.reason = f"capacity: {policy.capacity} places filled; weakest winner {weakest:g}, this {raw:g}"
        else:
            nomination.status, nomination.reason = "selected", f"score {raw:g}"
            winners.append(nomination)
    selected_ids = list(dict.fromkeys(int(key) for item in winners for key in item.target_ids))
    trace = {
        "policy": policy.model_dump(mode="json"),
        "danger": danger,
        "ranking": [{"nomination_id": item.nomination_id, "codelet": item.codelet, "score": raw,
                     "status": item.status, "reason": item.reason} for raw, _, item in order],
        "rejected": [{"nomination_id": item.nomination_id, "codelet": item.codelet, "reason": item.reason}
                     for item in nominations if item.status == "rejected"],
    }
    summary = ", ".join(f"{item.codelet}:{item.nomination_id}" for item in winners) or "nothing"
    return AttentionResult(
        nominations=nominations,
        selected_ids=selected_ids,
        reasoning=f"Selected {len(winners)} of {len(nominations)} nominations: {summary}.",
        competition=trace,
    )


__all__ = ["SCORE_AXES", "compete", "weighted_score"]
