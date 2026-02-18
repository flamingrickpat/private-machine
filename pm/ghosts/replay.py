from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class TickEvent(BaseModel):
    tick: int
    phase: str
    ok: bool
    duration_ms: float
    error: Optional[str] = None
    llm_last_phase: str = ""
    llm_last_ok: bool = False
    key_decisions: Dict[str, Any] = Field(default_factory=dict)


class TickReplaySnapshot(BaseModel):
    version: int = 1
    tick: int
    created_at: str
    phase_events: List[TickEvent] = Field(default_factory=list)
    phase_trace: List[Dict[str, Any]] = Field(default_factory=list)
    debug_panel: Dict[str, Any] = Field(default_factory=dict)
    key_decisions: Dict[str, Any] = Field(default_factory=dict)
    csm_state: Dict[str, Any] = Field(default_factory=dict)
    ccq_story: str = ""
    ccq_assistant: str = ""
    ccq_knoxel_ids: List[int] = Field(default_factory=list)
    tick_knoxels: List[Dict[str, Any]] = Field(default_factory=list)


def capture_tick_snapshot(ghost, ccq=None, include_tick_knoxels: bool = True) -> TickReplaySnapshot:
    tick = int(getattr(ghost, "current_tick_id", 0) or 0)
    phase_events_raw = list(getattr(ghost, "last_cycle_events", []) or [])
    phase_events = [TickEvent(**x) for x in phase_events_raw if isinstance(x, dict)]

    phase_trace = deepcopy(list(getattr(ghost, "last_cycle_trace", []) or []))
    debug_panel = deepcopy(getattr(ghost, "runtime_debug_panel", {}) or {})

    if ccq is not None:
        ccq_story = str(getattr(ccq, "as_story", "") or "")
        ccq_assistant = str(getattr(ccq, "as_assistant", "") or "")
        ccq_knoxel_ids = [int(k.id) for k in (getattr(ccq, "knoxels", None).to_list() if getattr(ccq, "knoxels", None) else [])]
    else:
        ccq_story = str(getattr(ghost, "simulated_reply", "") or "")
        ccq_assistant = ccq_story
        ccq_knoxel_ids = [int(x) for x in ((debug_panel.get("ccq", {}) or {}).get("knoxel_ids", []) or [])]

    key_decisions = {
        "winner_lens": str((getattr(ghost, "ego_decision_last", {}) or {}).get("winner_lens", "") or ""),
        "runner_up_lens": str((getattr(ghost, "ego_decision_last", {}) or {}).get("runner_up_lens", "") or ""),
        "dissonance": float((getattr(ghost, "ego_decision_last", {}) or {}).get("dissonance", 0.0) or 0.0),
        "workspace_gain": float(getattr(ghost, "workspace_gain", 0.0) or 0.0),
        "generation_token_cap": int(getattr(ghost, "generation_token_cap", 0) or 0),
        "thought_directive": str(getattr(ghost, "thought_blueprint_directive", "") or ""),
    }

    csm_state = {}
    if hasattr(ghost, "csm_manager") and getattr(ghost, "csm_manager", None) is not None:
        state = getattr(ghost.csm_manager, "state", None)
        if state is not None:
            csm_state = _jsonable(state)

    tick_knoxels: List[Dict[str, Any]] = []
    if include_tick_knoxels:
        for k in list(getattr(ghost, "all_knoxels", {}).values()):
            if int(getattr(k, "tick_id", -1)) != tick:
                continue
            tick_knoxels.append(_jsonable(k))

    return TickReplaySnapshot(
        tick=tick,
        created_at=datetime.utcnow().isoformat() + "Z",
        phase_events=phase_events,
        phase_trace=_jsonable(phase_trace),
        debug_panel=_jsonable(debug_panel),
        key_decisions=_jsonable(key_decisions),
        csm_state=_jsonable(csm_state),
        ccq_story=ccq_story,
        ccq_assistant=ccq_assistant,
        ccq_knoxel_ids=ccq_knoxel_ids,
        tick_knoxels=tick_knoxels,
    )


def save_tick_snapshot(snapshot: TickReplaySnapshot, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")


def load_tick_snapshot(path: str | Path) -> TickReplaySnapshot:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    return TickReplaySnapshot.model_validate(data)


class TickReplayHarness:
    def __init__(self, snapshot: TickReplaySnapshot):
        self.snapshot = snapshot

    @classmethod
    def from_file(cls, path: str | Path) -> "TickReplayHarness":
        return cls(load_tick_snapshot(path))

    def iter_events(self) -> Iterable[TickEvent]:
        for event in self.snapshot.phase_events:
            yield event

    def summary(self) -> Dict[str, Any]:
        events = list(self.snapshot.phase_events)
        err_count = sum(1 for e in events if not e.ok)
        total_ms = round(sum(float(e.duration_ms) for e in events), 3)
        return {
            "tick": self.snapshot.tick,
            "phase_count": len(events),
            "err_count": err_count,
            "total_ms": total_ms,
            "winner_lens": str(self.snapshot.key_decisions.get("winner_lens", "") or ""),
            "workspace_gain": float(self.snapshot.key_decisions.get("workspace_gain", 0.0) or 0.0),
            "ccq_story_len": len(self.snapshot.ccq_story or ""),
            "ccq_knoxel_count": len(self.snapshot.ccq_knoxel_ids),
            "tick_knoxel_count": len(self.snapshot.tick_knoxels),
        }

