"""Send observed world changes to the main cognition loop during silence."""
from __future__ import annotations

from pm import vclock

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel
from pm.bus.envelopes import Envelope, new_envelope, new_id
from pm.host.application import BusModule
from pm.memory.snapshot import GhostCancelled
from pm.records.enums import FeatureType, SourceKind
from pm.records.memory_records import new_feature
from pm.runs.errors import TaskServiceError


class GwtUpdate(BaseModel):
    """The payload of a ``gwt.updated`` message: the tick of the new turn record."""

    gwt_tick: int = 0
    slot: str = "main"


def world_conditions(observation: dict) -> dict:
    """Select structured body facts that can invalidate a survival decision.

    Images and camera rotation do not enter this comparison. World time uses
    day phases, so each server tick does not cause a cognition cycle.
    """
    player = observation.get("player", {})
    world = observation.get("world", player)
    return {
        "position": observation.get("camera", {}).get("block_position"),
        "player": {key: player[key] for key in (
            "health", "food", "food_level", "on_ground", "in_water", "in_lava",
            "alive", "life_id", "nutrition", "sleep", "dimension", "oxygen", "biome",
        ) if key in player},
        "inventory": sorted(observation.get("inventory", []), key=lambda item: json.dumps(item, sort_keys=True)),
        "world": {key: world[key] for key in ("dimension", "weather", "is_raining") if key in world},
        "day_phase": int(world["time_of_day"]) // 6000 if "time_of_day" in world else None,
        "life": [observation.get("body_session"), observation.get("death_count")],
        "nearby_blocks": observation.get("nearby_blocks", []),
    }


class WorldEvents(BusModule):
    """Observe the body and commit one event before asking cognition to choose.

    The application owns this module and its poll timer. The memory store owns
    accepted observations. A pending event prevents a second queued observation.
    Transport failures stay in the journal and do not establish a world fact.
    Shutdown cancels queued reads and discards a late response before commit.
    """

    DATA_TOPICS = frozenset({"gwt.updated", "action.receipt", "user.input"})

    def __init__(self, *args, ghost, observe: Callable[[], dict], journal: Path,
                 quiet_seconds: float = 120.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.ghost = ghost
        self.observe = observe
        self.journal = Path(journal)
        self.quiet_seconds = quiet_seconds
        self.pending_id: int | None = None
        self.pending_correlation = ""
        self.last_conditions: dict | None = None
        self.retry_observation = False
        self.last_user_at = self.clock()

    def _record(self, row: dict) -> None:
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def deliver(self, envelope: Envelope) -> None:
        """Record user arrival before a queued observation can finish."""
        if envelope.topic == "user.input" and envelope.kind == "data":
            self.last_user_at = self.clock()
        super().deliver(envelope)

    def _active_work(self) -> bool:
        with self.ghost.lock:
            return any(run["workflow_name"] in {"minecraft", "minecraft_project"}
                       and run["state"] in {"accepted", "running"}
                       for run in self.ghost.managed_runs.values())

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Poll after local silence, or acknowledge an accepted world event.

        A user message resets the silence clock. A restart starts a new clock;
        elapsed stopped time never creates an observation or a task receipt.
        """
        for envelope in batch:
            if envelope.topic == "gwt.updated" and self.pending_id is not None:
                with self.ghost.lock:
                    tick = GwtUpdate.model_validate(envelope.payload).gwt_tick
                    turn = self.ghost.gwt_turns[tick] if tick in self.ghost.gwt_turns else None
                    # The main cycle that claimed the observation acknowledges it.
                    if turn is not None and self.pending_id in turn.stimulus_ids:
                        self.pending_id = None
            if (envelope.topic == "action.receipt" and self.pending_id is not None
                    and envelope.correlation_id == self.pending_correlation
                    and envelope.payload.get("recovery_outcome") == "cycle_rejected"):
                self.pending_id = None
                self.retry_observation = True
        silent = self.clock() - self.last_user_at >= self.quiet_seconds
        if batch or self.pending_id is not None or not (silent or self.last_conditions is not None and self._active_work()):
            return
        requested_at = vclock.now_aware().isoformat()
        try:
            response = self.observe()
        except TaskServiceError as error:
            self._record({"status": "unavailable", "requested_at": requested_at, "error": str(error)})
            return
        if self.stop_event.is_set():
            return
        silent = self.clock() - self.last_user_at >= self.quiet_seconds
        if not silent and not (self.last_conditions is not None and self._active_work()):
            return
        observed_at = vclock.now_aware().isoformat()
        if not response.get("ok"):
            self._record({"status": "unavailable", "observed_at": observed_at, "response": response})
            return
        conditions = world_conditions(response["result"])
        if conditions == self.last_conditions and not self.retry_observation:
            return
        previous = self.last_conditions
        correlation = "world-event:" + new_id()
        circumstance = "during user silence" if silent else "during active game work"
        text = "Current Minecraft observation " + circumstance + ": " + json.dumps(conditions, ensure_ascii=False)
        changes = self.ghost.begin_batch("world-observation:" + correlation, cancel=self.stop_event)
        try:
            feature = new_feature(
                self.ghost, text, FeatureType.WorldEvent, source=self.module_id,
                source_kind=SourceKind.Workflow,
                changes=changes,
                metadata={"origin": "minecraft.world", "executed_at": observed_at,
                          "requested_at": requested_at, "user_silent": silent,
                          "evidence": response["result"], "previous_conditions": previous,
                          "correlation_id": correlation, "self_origin": True},
            )
            self.ghost.commit(changes)
        except GhostCancelled:
            return
        self.last_conditions = conditions
        self.retry_observation = False
        self.pending_id, self.pending_correlation = feature.id, correlation
        self._record({"status": "committed", "observed_at": observed_at, "knoxel_id": feature.id,
                      "correlation_id": correlation, "conditions": conditions, "previous_conditions": previous})
        self.bus.put(new_envelope(self.module_id, "data", "minecraft.world", correlation,
                                  {"knoxel_id": feature.id}))


__all__ = ["WorldEvents", "world_conditions"]
