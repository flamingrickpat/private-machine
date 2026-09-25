"""Shared helpers for the thread-free direct module tests."""
from __future__ import annotations

import copy
from functools import cache
import tempfile
from pathlib import Path

from pm_core.model.gwt import GwtTurnRecord
from pm_core.model.knoxel_common import ActionSchemaKnoxel, Feature
from pm_core.model.knoxel_enums import ActionType, FeatureType, InterlocusType
from pm_core.persist.dataprovider_yaml import (
    ghost_state_from_dict,
    ghost_state_to_dict,
    load_ghost_state_from_yaml,
)

from ghost import Ghost
from main import build_app
from shell import Envelope, _new_id

SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "spamuel_agentic_canon.yaml"

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


def fresh_ghost(seed_dict: dict) -> Ghost:
    state = ghost_state_from_dict(copy.deepcopy(seed_dict))
    return Ghost.from_state(state)


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


def store_feature(ghost: Ghost, content: str = "test feature") -> int:
    feature = ghost.store_knoxel(Feature(
        content=content,
        feature_type=FeatureType.Dialogue,
        interlocus=InterlocusType.Public,
        causal=True,
        override_source=ghost.get_user_name(),
    ))
    return feature.id


def store_action(
    ghost: Ghost,
    *,
    route: str = "dialogue",
    content: str = "test",
    task_id: str = "",
    task_kind: str = "",
) -> Envelope:
    changes = ghost.begin_batch(_new_id(), new_tick=True)
    tick = changes.tick_id
    action = changes.create(ActionSchemaKnoxel(
        content=content,
        action_type=_ACTION_TYPE[route],
        metadata={"route": route, "task_id": task_id, "task_kind": task_kind},
    ))
    changes.set_gwt(GwtTurnRecord(
        tick_id=tick, attention_frame_id=None, knoxel_ids=[action.id], summary="x",
    ))
    ghost.commit(changes)
    correlation_id = _new_id()
    return Envelope(
        id=_new_id(), source="test", kind="data", topic=f"action.{route}",
        correlation_id=correlation_id, payload={"knoxel_id": action.id, "gwt_tick": tick},
    )
