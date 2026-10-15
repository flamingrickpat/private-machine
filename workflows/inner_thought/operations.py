"""The delivered private thought operations and the old type mapping.

The historical ``thought_graph.py`` catalogue named 37 thought types with
random transition weights. A17 does not reproduce them all. This module
delivers the operations with a live use, and maps every old type to its
successor or to an explicit non-port decision, so nothing silently disappears.

An operation belongs to one required group. Generative work creates
possibilities; discriminatory work evaluates or distinguishes them. Neither
mode establishes an external event by generation alone: the reality label on
the committed thought keeps imagined content apart from observed content.
"""
from __future__ import annotations

from dataclasses import dataclass

GROUPS = ("remember", "imagine", "evaluate", "plan", "interpret", "notice")


@dataclass(frozen=True)
class Operation:
    """One delivered thought operation and its required declarations."""

    name: str
    group: str
    mode: str  # "generative" creates possibilities, "discriminatory" evaluates them.
    temporal_focus: str  # "past", "present", or "future"; the model may narrow it.
    description: str
    cause_required: bool = True


OPERATIONS: dict[str, Operation] = {
    "memory_search": Operation(
        "memory_search", "remember", "generative", "past",
        "Retrieve episodes or preferences relevant to a current concern, and attribute each result to its source."),
    "episodic_recall": Operation(
        "episodic_recall", "remember", "discriminatory", "past",
        "Revisit one attributed episode and separate what happened from how it reads now."),
    "memory_gap_probe": Operation(
        "memory_gap_probe", "remember", "discriminatory", "past",
        "Inspect a memory gap: what is missing, why it is missing, and whether it matters."),
    "future_projection": Operation(
        "future_projection", "imagine", "generative", "future",
        "Project one possible outcome of a current goal or relationship, keeping it a prediction."),
    "counterfactual": Operation(
        "counterfactual", "imagine", "generative", "past",
        "Compare what happened with one alternative path, without rewriting the observed record."),
    "perspective_shift": Operation(
        "perspective_shift", "imagine", "discriminatory", "present",
        "Reconsider an event from the partner's or a third position's perspective."),
    "feasibility_check": Operation(
        "feasibility_check", "evaluate", "discriminatory", "present",
        "Weigh competing motives or feasible actions and record each alternative with its disposition."),
    "commitment_review": Operation(
        "commitment_review", "evaluate", "discriminatory", "present",
        "Review an accepted commitment: retain, reprioritize, defer, or revise it, with the reason."),
    "plan_next_step": Operation(
        "plan_next_step", "plan", "generative", "future",
        "Develop the next concrete step for an active goal, including one contingency."),
    "curiosity_spark": Operation(
        "curiosity_spark", "plan", "generative", "future",
        "Turn an open question or boredom into a private goal worth pursuing later."),
    "emotional_debrief": Operation(
        "emotional_debrief", "interpret", "discriminatory", "past",
        "Debrief a recent emotional episode: what it meant, what remains unresolved, and what is deferred."),
    "narrative_stitching": Operation(
        "narrative_stitching", "interpret", "generative", "past",
        "Link attributed episodes into one revisable interpretation, distinguishing order from causality."),
    "rumination_notice": Operation(
        "rumination_notice", "notice", "discriminatory", "present",
        "Notice that the same concern keeps returning, and decide whether to continue, summarize, or suspend."),
    "attention_drift_notice": Operation(
        "attention_drift_notice", "notice", "discriminatory", "present",
        "Notice that attention drifted from an active goal, and propose a focus intention for the next main cycle."),
    "expectation_review": Operation(
        "expectation_review", "notice", "discriminatory", "present",
        "Compare a pending expectation with what was actually observed and propose its resolution."),
}

#: The mapping from every historical thought type to its successor. A value of
#: ``None`` is an explicit non-port: the named owner already covers that use.
OLD_TYPE_MAPPING: dict[str, str | None] = {
    "TH_PRESENT_APPRAISAL": "narrative_stitching",
    "TH_IMAGINATIVE_PROJECTION": "future_projection",
    "TH_FEASIBILITY_CHECK": "feasibility_check",
    "TH_MEMORY_SEARCH": "memory_search",
    "TH_INTROSPECTIVE_RUMINATION": "rumination_notice",
    "TH_PLAN_NEXT_STEP": "plan_next_step",
    "TH_COMPOSE_REPLY": None,  # The dialogue workflows own public replies.
    "TH_BACKTRACK_PIVOT": "commitment_review",
    "TH_EPISODIC_SNAPSHOT": "episodic_recall",
    "TH_PATTERN_MINING": "narrative_stitching",
    "TH_COUNTERFACTUAL_REWRITE": "counterfactual",
    "TH_REGRET_AUDIT": "emotional_debrief",
    "TH_GRATITUDE_RECALL": "emotional_debrief",
    "TH_SKILL_ABSTRACTION": "narrative_stitching",
    "TH_SOURCE_ATTRIBUTION": "memory_search",
    "TH_NARRATIVE_STITCHING": "narrative_stitching",
    "TH_EMOTIONAL_DEBRIEF": "emotional_debrief",
    "TH_MEMORY_GAP_PROBE": "memory_gap_probe",
    "TH_NEED_CHECK": "commitment_review",
    "TH_BOUNDARY_GUARD": "commitment_review",
    "TH_PERSPECTIVE_SHIFT": "perspective_shift",
    "TH_CURIOSITY_SPARK": "curiosity_spark",
    "TH_SOCIAL_TEMPERATURE_READ": "perspective_shift",
    "TH_SYSTEMS_SCAN": "memory_gap_probe",
    "TH_VALUE_ALIGNMENT_CHECK": "feasibility_check",
    "TH_SELF_COMPASSION_TAKE": "emotional_debrief",
    "TH_OPPORTUNITY_SPOTTING": "feasibility_check",
    "TH_SCENARIO_SKETCH": "future_projection",
    "TH_RISK_FORECAST": "future_projection",
    "TH_MICRO_PLAN": "plan_next_step",
    "TH_CONTINGENCY_BRANCHING": "plan_next_step",
    "TH_COMMITMENT_TEST": "commitment_review",
    "TH_SOCIAL_BID_CRAFTING": None,  # The main loop's action selection owns bids.
    "TH_LONG_HORIZON_WEAVE": "plan_next_step",
    "TH_WISHFUL_IDEATION": "future_projection",
    "TH_SIMULATED_FEEDBACK": "future_projection",
    "TH_EXIT_PIVOT_DECISION": "commitment_review",
}


def operation_lines() -> list[str]:
    """Render the delivered operations for a generation prompt."""
    lines = []
    for operation in OPERATIONS.values():
        lines.append(f"- {operation.name} ({operation.group}, {operation.temporal_focus}): {operation.description}")
    return lines


def validate_operation(name: str) -> Operation:
    """Return the declared operation or raise for an unknown name."""
    if name not in OPERATIONS:
        raise KeyError(f"unknown thought operation: {name}")
    return OPERATIONS[name]


__all__ = ["GROUPS", "OLD_TYPE_MAPPING", "OPERATIONS", "Operation", "operation_lines", "validate_operation"]
