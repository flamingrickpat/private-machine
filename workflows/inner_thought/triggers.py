"""The deterministic trigger policy for private cycles.

The host evaluates these triggers before any model call. A private cycle only
reaches the model when at least one trigger passes. A cycle with no trigger
commits nothing: the loop does not generate thoughts to satisfy a quota.

Every trigger names its cause and suggests operations from the delivered
catalogue. The trigger stage may pick one suggested operation; the host never
lets the model invent an operation name.
"""
from __future__ import annotations

from workflows.inner_thought.schemas import TriggerCandidate

#: An active goal older than this many ticks with low fulfilment is unresolved.
UNRESOLVED_GOAL_TICKS = 8
#: A pending expectation is worth a review after this many ticks.
EXPECTATION_REVIEW_TICKS = 6
#: A surprise magnitude above this value triggers a debrief.
SURPRISE_THRESHOLD = 0.35
#: This many seconds without user input make idle triggers available.
IDLE_SECONDS = 300.0
#: The same source recurring in this many recent thoughts is rumination.
RUMINATION_REPEATS = 3
#: An offline gap above this value triggers a reconnect thought after startup.
STARTUP_GAP_SECONDS = 3600.0


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def evaluate_triggers(request) -> list[TriggerCandidate]:
    """Return the triggers that fire for one frozen private request.

    The input is the validated ``InnerRequest``. The output order is stable so
    tests and traces are reproducible. Urgency stays in 0..1.
    """
    candidates: list[TriggerCandidate] = []
    tick = int(request.tick_id or 0)
    # A parked concern stays quiet until its review time or new outside evidence.
    suppressed = {int(item) for item in (getattr(request, "episode", {}) or {}).get("suppressed_source_ids", [])}
    episode = getattr(request, "episode", {}) or {}
    if (episode.get("episode_id") and not episode.get("closed")
            and not episode.get("awaiting_acknowledgment") and episode.get("next_operation")):
        candidates.append(TriggerCandidate(
            name="episode_continuation", cause=f"main appraisal acknowledged episode {episode['episode_id']}",
            urgency=0.45, operations=[episode["next_operation"]],
            reason="The previous private step requested more work after main appraisal."))

    for goal in request.active_goal_records:
        if int(goal.get("knoxel_id")) in suppressed:
            continue
        age = tick - int(goal.get("tick_id", tick) or tick)
        fulfilment = _as_float(goal.get("fulfilment"), 0.0)
        if age >= UNRESOLVED_GOAL_TICKS and fulfilment < 0.5:
            candidates.append(TriggerCandidate(
                name="unresolved_commitment", cause=f"goal {goal.get('knoxel_id')} is {age} ticks old and {fulfilment:.2f} fulfilled",
                urgency=min(1.0, 0.4 + age / 100.0),
                operations=["commitment_review", "plan_next_step", "feasibility_check"],
                reason="An accepted goal with no progress needs private attention.",
            ))
            break

    for expectation in request.pending_expectations:
        if int(expectation.get("knoxel_id")) in suppressed:
            continue
        age = tick - int(expectation.get("tick_id", tick) or tick)
        if age >= EXPECTATION_REVIEW_TICKS:
            candidates.append(TriggerCandidate(
                name="pending_expectation", cause=f"expectation {expectation.get('knoxel_id')} is {age} ticks old and unresolved",
                urgency=min(1.0, 0.35 + age / 100.0),
                operations=["expectation_review", "emotional_debrief"],
                reason="A recorded anticipation has not met its outcome yet.",
            ))
            break

    surprise = _as_float((request.main_frame or {}).get("surprise"))
    if abs(surprise) >= SURPRISE_THRESHOLD:
        candidates.append(TriggerCandidate(
            name="surprising_outcome", cause=f"the last main frame reports surprise {surprise:.2f}",
            urgency=min(1.0, 0.4 + abs(surprise) / 2.0),
            operations=["emotional_debrief", "expectation_review", "perspective_shift"],
            reason="A surprising outcome deserves a private interpretation.",
        ))

    if (int(request.main_frame_age_ticks or 0) == 0 and (request.main_frame or {}).get("summary")
            and not request.main_frame.get("self_feedback")):
        candidates.append(TriggerCandidate(
            name="fresh_main_frame", cause="the main loop published a new workspace frame",
            urgency=0.3,
            operations=["perspective_shift", "narrative_stitching", "emotional_debrief"],
            reason="A fresh accepted workspace gives private thought new material.",
        ))

    if request.seconds_since_user_input is None or request.seconds_since_user_input >= IDLE_SECONDS:
        idle = request.seconds_since_user_input
        candidates.append(TriggerCandidate(
            name="idle_opportunity", cause=f"the last user input was {idle:.0f} seconds ago" if idle is not None else "no user input is recorded",
            urgency=0.25,
            operations=["curiosity_spark", "attention_drift_notice", "memory_gap_probe", "rumination_notice"],
            reason="Idle time is the opportunity for private work without a conversation partner.",
        ))

    repeated = _repeated_source(request.recent_thoughts)
    if repeated is not None and int(repeated) in suppressed:
        repeated = None
    if repeated is not None:
        candidates.append(TriggerCandidate(
            name="repeated_concern", cause=f"the source {repeated} recurs in recent thoughts",
            urgency=0.55,
            operations=["rumination_notice", "attention_drift_notice", "emotional_debrief"],
            reason="A recurring concern may preserve and reinforce negative affect.",
        ))

    for thought in request.recent_thoughts:
        if str(thought.get("trigger")) == "startup_gap":
            break
    else:
        gap = _as_float((request.main_frame or {}).get("gap_seconds"))
        if gap >= STARTUP_GAP_SECONDS:
            candidates.append(TriggerCandidate(
                name="startup_gap", cause=f"the application was stopped for {gap:.0f} seconds",
                urgency=0.45,
                operations=["emotional_debrief", "narrative_stitching", "perspective_shift"],
                reason="An offline gap is an attributed absence, not invented activity.",
            ))

    doubt = _capability_doubt(request)
    if doubt is not None:
        candidates.append(doubt)

    return [item for item in candidates if item.operations]


def _capability_doubt(request) -> TriggerCandidate | None:
    """Fire when the inspected capability evidence is stale or unobserved.

    The trigger reads the public overlay the host already built. It never
    reads a file itself and never treats a missing plan as a live capability.
    """
    overlay = (request.inspection or {}).get("capability_overlay") or {}
    capabilities = overlay.get("capabilities") or []
    stale = set(overlay.get("stale") or [])
    concerns = sorted(
        str(item.get("name")) for item in capabilities
        if str(item.get("name")) in stale
        or item.get("state") in {"intended", "implemented", "unavailable", "uncertain"}
    )
    if not concerns:
        return None
    return TriggerCandidate(
        name="capability_doubt",
        cause="inspected capability evidence is stale, planned, or unobserved: " + ", ".join(concerns[:4]),
        urgency=0.4,
        operations=["capability_inspection", "episode_inspection", "memory_gap_probe"],
        reason="A capability belief that outruns its evidence is worth a private check.",
    )


def _repeated_source(recent_thoughts) -> int | str | None:
    """Return the id that appears in enough recent thoughts to count as rumination."""
    counts: dict[str, int] = {}
    for thought in recent_thoughts:
        for source_id in thought.get("source_ids") or []:
            key = str(source_id)
            counts[key] = counts.get(key, 0) + 1
            if counts[key] >= RUMINATION_REPEATS:
                return source_id
    return None


__all__ = ["evaluate_triggers"]
