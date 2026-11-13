"""Shared helpers for the thread-free direct module tests."""
from __future__ import annotations

from pm.records.copying import pickle_copy

from functools import cache
import tempfile
from pathlib import Path

from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord, Feature
from pm.records.enums import ActionType, FeatureType, InterlocusType
from pm.memory.yaml_codec import ghost_state_from_dict, ghost_state_to_dict, load_ghost_state_from_yaml

from pm.memory.store import MemoryStore
from workflows.companion import build_app
from pm.bus.envelopes import Envelope, new_id

SEED_PATH = Path(__file__).resolve().parent.parent / "test_data" / "personas" / "spamuel" / "canon.yaml"

_ACTION_TYPE = {
    "dialogue": ActionType.Reply,
    "tool_dialogue": ActionType.ToolCallAndReply,
    "schedule": ActionType.ToolCall,
    "status": ActionType.ToolCall,
}


class ManualClock:
    def __init__(self, start: float = 0.0):
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class Recorder:
    """Subscribes to the real bus and records matching envelopes."""

    def __init__(self, bus, topic: str | None = None):
        self.events: list[Envelope] = []
        self.topic = topic
        bus.subscribe(f"recorder-{id(self)}", self._accepts, self._deliver)

    def _accepts(self, envelope: Envelope) -> bool:
        if envelope.kind == "control":
            return False
        return self.topic is None or envelope.topic == self.topic

    def _deliver(self, envelope: Envelope) -> None:
        self.events.append(envelope)


@cache
def load_seed_state():
    state = load_ghost_state_from_yaml(SEED_PATH)
    return ghost_state_to_dict(state)


def fresh_ghost(seed_dict: dict) -> MemoryStore:
    state = ghost_state_from_dict(pickle_copy(seed_dict))
    return MemoryStore.from_state(state)


def make_app(seed_dict: dict, runtime_root: Path | None = None):
    """Build a fresh app with a temporary runtime root and manual clock."""
    root = runtime_root or Path(tempfile.mkdtemp(prefix="pmnext-"))
    clock = ManualClock()
    replies: list[str] = []
    app = build_app(
        seed_path=SEED_PATH,
        ghost=fresh_ghost(seed_dict),
        runtime_root=root,
        thought_period=30.0,
        task_period=1.0,
        clock=clock,
        emit_reply=replies.append,
    )
    app.clock = clock
    app.replies = replies
    return app


def store_feature(ghost: MemoryStore, content: str = "test feature") -> int:
    feature = ghost.store_knoxel(Feature(
        content=content,
        feature_type=FeatureType.Dialogue,
        interlocus=InterlocusType.Public,
        causal=True,
        override_source=ghost.get_user_name(),
    ))
    return feature.id


def store_action(
    ghost: MemoryStore,
    *,
    route: str = "dialogue",
    content: str = "test",
    task_id: str = "",
    task_kind: str = "",
) -> Envelope:
    changes = ghost.begin_batch(new_id(), new_tick=True)
    tick = changes.tick_id
    action = changes.create(ActionSchemaRecord(
        content=content,
        action_type=_ACTION_TYPE[route],
        metadata={"route": route, "task_id": task_id, "task_kind": task_kind},
    ))
    changes.set_gwt(GwtTurnRecord(
        tick_id=tick, knoxel_ids=[action.id], summary="x",
    ))
    ghost.commit(changes)
    correlation_id = new_id()
    return Envelope(
        id=new_id(), source="test", kind="data", topic=f"action.{route}",
        correlation_id=correlation_id, payload={"knoxel_id": action.id, "gwt_tick": tick},
    )


class PrivateInquiryRouteReached(AssertionError):
    """A private cycle entered the inquiry graph in a test that pins another path."""


def refuse_private_inquiry(request: object, run_stage: object) -> object:
    """Satisfy the ``inner_thought`` tool grant while supplying no inquiry owner.

    The private loop's manifest grants ``run_private_inquiry`` to its role, so a
    test that pins the retained linear episode path must still grant the tool.
    This boundary names the routing change instead of faking an inquiry result.
    """
    raise PrivateInquiryRouteReached(
        "This private-loop test grants no inquiry owner; the graph path must not run.")


def seed_retained_episode(ghost: MemoryStore, *, episode_id: str = "retained-linear",
                          sequence: int = 0, closed: bool = False,
                          next_operation: str = "plan_next_step", stalled: int = 0,
                          parked_sources: tuple[int, ...] = (), parked_until_seconds: float | None = None,
                          control_generation: int | None = None, acknowledged: bool = True,
                          content: str = "the story draft still waits") -> int:
    """Commit one accepted private thought that carries a retained linear episode.

    B05 sends every fresh private opportunity to the inquiry graph. A retained
    pre-graph episode is the only cycle the linear pipeline still runs, so a
    linear-pipeline test starts from this record instead of a fresh cycle.
    ``parked_until_seconds`` opens a parking interval that has not elapsed.
    """
    from datetime import datetime, timedelta, timezone

    from pm.records.cycle_records import StimulusGroup
    from pm.records.cycle_state import CycleState
    from pm.records.enums import SourceKind
    from pm.records.memory_records import new_feature
    from workflows.inner_thought.schemas import LegacyEpisodeState

    # A new tick also claims the record in that tick's cycle state, which would
    # acknowledge the thought. An unacknowledged seed therefore joins no new tick.
    batch = ghost.begin_batch(new_id(), new_tick=acknowledged)
    thought = new_feature(
        ghost, content, FeatureType.InnerMonologue,
        source=ghost.get_companion_name(), changes=batch, source_kind=SourceKind.Self,
        metadata={
        "loop": "inner", "reality": "internal", "operation": "curiosity_spark",
        **({"control_generation": control_generation} if control_generation is not None else {}),
        "episode": LegacyEpisodeState(
            episode_id=episode_id,
            sequence=sequence,
            started_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=5),
            elapsed_seconds=5.0,
            closed=closed,
            closed_reason="test" if closed else None,
            operations=("curiosity_spark",),
            evidence_ids=(),
            alternatives=(),
            thoughts=(content,),
            progress=("none",),
            stalled=stalled,
            continuation="continue" if not closed else "summarize",
            next_operation=next_operation,
            parked_until=(datetime.now(timezone.utc).replace(tzinfo=None)
                          + timedelta(seconds=parked_until_seconds)) if parked_until_seconds is not None else None,
            parked_sources=parked_sources,
            parent_thought_id=None,
        ).model_dump(mode="json"),
        },
    )
    if acknowledged:
        batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, stimulus_ids=[thought.id],
                                    knoxel_ids=[thought.id]),
                      CycleState(tick_id=batch.tick_id,
                                 control_generation=control_generation or 0,
                                 stimulus_groups=[StimulusGroup(stimulus_ids=[thought.id])]))
    ghost.commit(batch)
    return thought.id
