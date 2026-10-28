"""Deterministic felt-state channel projections.

Each channel projects accepted mental state into one position with explicit
units. The projection is pure math over committed state: it creates no new
state authority and no parallel store. Velocity and acceleration use the
elapsed seconds between accepted samples, so a stopped application supplies
no imaginary intermediate samples.

Sampling convention for nonuniform intervals: each accepted private cycle
samples the latest accepted mental state. ``elapsed_seconds`` is the wall
clock difference between the previous sample and this one. A nonpositive
interval means the sample carries no new time information: the derivatives are
reported as 0.0 with ``stalled=True``, never as measured stillness. The first
sample reports ``initialized=True`` with zero derivatives, which is not
presented as an absence of movement.
"""
from __future__ import annotations

from datetime import datetime

from pm.records.cognitive_records import Intention
from pm.records.mental_state import FullMentalState
from pm.records.cycle_state import CycleState
from pm.records.qualia_records import QualeChannelState

#: Channel identity, projection units, and the mental-state fields each
#: channel reads. ``negative`` fields subtract from the position.
CHANNELS: dict[str, dict] = {
    "affect.valence": {"units": "-1..1", "positive": ("joy", "tenderness", "pride", "amusement", "relief"),
                       "negative": ("sadness", "anger", "fear", "disgust", "resentment", "jealousy", "boredom", "shame", "guilt"),
                       "scale": 5.0},
    "affect.arousal": {"units": "0..1", "positive": ("fear", "anger", "surprise", "intimacy_arousal", "curiosity"),
                       "negative": (), "scale": 4.0},
    "relationship.warmth": {"units": "-1..1", "positive": ("affection", "trust", "communication_warmth"),
                           "negative": (), "scale": 3.0, "partner": True},
    "attention.engagement": {"units": "0..1", "positive": ("curiosity", "playfulness"), "negative": (), "scale": 2.0},
    "goal.conflict": {"units": "0..1", "positive": (), "negative": (), "scale": 1.0, "goals": True},
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def channel_position(channel_id: str, state: FullMentalState, *, active_goals: list | None = None) -> tuple[float, str]:
    """Project one channel from accepted state. Return ``(position, note)``.

    ``note`` is empty on a normal projection. An absent source (for example,
    no partner relationship in the state yet) reports ``0.0`` with the note
    ``absent source``, which is an uninitialized projection and not a measured
    neutral value.
    """
    spec = CHANNELS[channel_id]
    if spec.get("partner"):
        relationship = state.state_relationship
        if relationship is None or relationship.entity_id is None:
            return 0.0, "absent source"
        values = [getattr(relationship, name) for name in spec["positive"]]
        return _clamp(sum(values) / spec["scale"], -1.0, 1.0), ""
    if spec.get("goals"):
        goals = list(active_goals or [])
        if not goals:
            return 0.0, "absent source"
        conflicted = [goal for goal in goals if goal.affective_valence <= -0.3 and goal.fulfilment < 0.5]
        return _clamp(len(conflicted) / len(goals), 0.0, 1.0), ""
    emotions = state.state_emotions
    positive = sum(getattr(emotions, name) for name in spec["positive"])
    negative = sum(getattr(emotions, name) for name in spec["negative"])
    position = (positive - negative) / spec["scale"]
    low = 0.0 if spec["units"] == "0..1" else -1.0
    return _clamp(position, low, 1.0), ""


def _comparable(previous_time: datetime, now: datetime) -> tuple[datetime, datetime]:
    """Return both timestamps on the same naive local clock.

    ``CycleState`` timestamps are naive local times, and a checkpoint can mix
    them with zone-aware times from older seeds. Elapsed time is a duration,
    so both sides are compared on one clock: an aware timestamp keeps its
    wall time and loses its zone marker.
    """
    previous = previous_time.replace(tzinfo=None) if previous_time.tzinfo else previous_time
    current = now.replace(tzinfo=None) if now.tzinfo else now
    return previous, current


def project_channel(channel_id: str, state: FullMentalState, *,
                    active_goals: list[Intention] | None = None,
                    previous: QualeChannelState | None, previous_time: datetime | None,
                    now: datetime, driver_ids: list[int]) -> QualeChannelState:
    """Project one channel with its time derivatives.

    ``driver_ids`` name the accepted records that moved the channel this
    sample: the appraisal record that produced the state and the attention
    records that won the cycle. Reportability is decided by the caller, which
    knows whether those drivers were causal or won attention.
    """
    position, _ = channel_position(channel_id, state, active_goals=active_goals)
    velocity = 0.0
    acceleration = 0.0
    if previous is not None and previous_time is not None:
        started, current = _comparable(previous_time, now)
        elapsed = (current - started).total_seconds()
        if elapsed > 0.0:
            velocity = (position - previous.position) / elapsed
            if previous.velocity != 0.0 or velocity != 0.0:
                acceleration = (velocity - previous.velocity) / elapsed
    return QualeChannelState(
        channel_id=channel_id, position=position, velocity=velocity, acceleration=acceleration,
        last_driver_ids=list(driver_ids),
    )


def project_channels(state: FullMentalState, *, active_goals: list[Intention] | None = None,
                     previous: dict[str, QualeChannelState] | None,
                     previous_time: datetime | None, now: datetime | None,
                     driver_ids: list[int]) -> dict[str, QualeChannelState]:
    """Project every delivered channel into a per-cycle channel state map.

    ``previous`` is the last committed channel state map, taken from the
    preceding cycle's ``CycleState.qualia_channels``. Passing ``None`` or an
    absent channel initializes that channel with zero derivatives.
    """
    now = now or datetime.now()
    result: dict[str, QualeChannelState] = {}
    for channel_id in CHANNELS:
        prior = previous[channel_id] if previous is not None and channel_id in previous else None
        result[channel_id] = project_channel(
            channel_id, state, active_goals=active_goals, previous=prior,
            previous_time=previous_time, now=now, driver_ids=driver_ids,
        )
    return result


def notable_channels(channels: dict[str, QualeChannelState], *, threshold: float = 0.05) -> list[str]:
    """Return the channels whose movement exceeds the snapshot threshold."""
    return [channel_id for channel_id, state in channels.items()
            if abs(state.velocity) >= threshold or abs(state.acceleration) >= threshold]


def is_reportable(driver_ids: list[int], *, attention_ids: set[int], causal_ids: set[int]) -> bool:
    """Apply the driver gate.

    A quale is reportable only when at least one driver won attention or was
    causal in accepted state. Dormant candidates stay visible and unreported.
    """
    drivers = {int(item) for item in driver_ids}
    return bool(drivers & (attention_ids | causal_ids))


def previous_cycle_sample(cycles: list[CycleState]) -> tuple[dict[str, QualeChannelState] | None, datetime | None]:
    """Return the channel state map and timestamp of the newest cycle that has one.

    The input is the accepted ``CycleState`` history, oldest first. A restart
    keeps committed history, so continuity survives checkpoints. An offline
    gap contributes no intermediate samples.
    """
    for cycle in reversed(cycles):
        if cycle.qualia_channels:
            return dict(cycle.qualia_channels), cycle.timestamp
    return None, None


__all__ = [
    "CHANNELS",
    "channel_position",
    "is_reportable",
    "notable_channels",
    "previous_cycle_sample",
    "project_channel",
    "project_channels",
]
