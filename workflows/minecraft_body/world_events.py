"""Send observed world changes to the main cognition loop during silence."""
from __future__ import annotations

from pm import vclock

from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ConfigDict, JsonValue
from dataclasses import dataclass, field
from workflows.minecraft_body.world_text import WorldObservation, WorldItem, NearbyBlock, render_world_observation
from workflows.minecraft_project.schemas import Point
from pm.bus.envelopes import Envelope, new_envelope, new_id
from pm.host.application import BusModule
from pm.memory.snapshot import GhostCancelled
from pm.records.enums import FeatureType, SourceKind
from pm.records.memory_records import new_feature
from pm.runs.errors import TaskServiceError

if TYPE_CHECKING:
    from pm.memory.store import MemoryStore


class GwtUpdate(BaseModel):
    """The payload of a ``gwt.updated`` message: the tick of the new turn record."""

    gwt_tick: int = 0
    slot: str = "main"


class WorldConditionPlayer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    health: float | None = None
    food: int | None = None
    food_level: int | None = None
    on_ground: bool | None = None
    in_water: bool | None = None
    in_lava: bool | None = None
    alive: bool | None = None
    life_id: str | int | None = None
    nutrition: JsonValue = None
    sleep: JsonValue = None
    dimension: str | None = None
    oxygen: float | None = None
    biome: str | None = None


class WorldConditionEnvironment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    dimension: str | None = None
    weather: str | None = None
    is_raining: bool | None = None


class WorldConditionLife(BaseModel):
    model_config = ConfigDict(frozen=True)
    body_session: str | None
    death_count: int | None


class WorldConditions(BaseModel):
    model_config = ConfigDict(frozen=True)
    comparison_fingerprint: str
    position: Point | None
    player: WorldConditionPlayer
    inventory: tuple[WorldItem, ...]
    world: WorldConditionEnvironment
    day_phase: int | None
    life: WorldConditionLife
    nearby_blocks: tuple[NearbyBlock, ...]


@dataclass(frozen=True, order=True, slots=True)
class InventoryConditionOrder:
    canonical: str
    item: WorldItem = field(compare=False)


@dataclass(frozen=True, slots=True)
class WorldEventRenderInput:
    observation: WorldObservation
    conditions: WorldConditions
    user_silent: bool


def render_world_event(input: WorldEventRenderInput) -> str:
    circumstance = "during user silence" if input.user_silent else "during active game work"
    environment = input.conditions.world
    world = f"World phase {input.conditions.day_phase}; dimension {environment.dimension}; weather {environment.weather}; raining={environment.is_raining}"
    return "Current Minecraft observation " + circumstance + ":\n" + world + "\n" + render_world_observation(input.observation)



def normalized_comparison_numbers(value: JsonValue) -> JsonValue:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: normalized_comparison_numbers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalized_comparison_numbers(item) for item in value]
    return value


@dataclass(frozen=True, order=True, slots=True)
class RawInventoryOrder:
    canonical: str
    item: JsonValue = field(compare=False)



def world_day_phase(world: dict[str, JsonValue]) -> int | None:
    from pydantic import TypeAdapter
    if "time_of_day" not in world:
        return None
    clock = TypeAdapter[str | int | float](str | int | float).validate_python(world["time_of_day"])
    return int(clock) // 6000

def condition_fingerprint(observation: dict[str, JsonValue]) -> str:
    from pydantic import TypeAdapter
    player = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(observation.get("player", {}))
    world = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(observation.get("world", player))
    camera = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(observation.get("camera", {}))
    inventory = TypeAdapter[list[JsonValue]](list[JsonValue]).validate_python(observation.get("inventory", []))
    ranked = sorted(RawInventoryOrder(json.dumps(item, sort_keys=True), item) for item in inventory)
    selected: dict[str, JsonValue] = {
        "position": camera.get("block_position"),
        "player": {key: player[key] for key in ("health", "food", "food_level", "on_ground", "in_water", "in_lava",
            "alive", "life_id", "nutrition", "sleep", "dimension", "oxygen", "biome") if key in player},
        "inventory": [item.item for item in ranked],
        "world": {key: world[key] for key in ("dimension", "weather", "is_raining") if key in world},
        "day_phase": world_day_phase(world),
        "life": [observation.get("body_session"), observation.get("death_count")],
        "nearby_blocks": observation.get("nearby_blocks", []),
    }
    # Python's original deep equality treats integral floats and booleans as their integer values.
    canonical = json.dumps(normalized_comparison_numbers(selected), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def world_conditions(observation: dict[str, JsonValue]) -> WorldConditions:
    parsed = WorldObservation.model_validate(observation)
    player = observation.get("player", {})
    world = observation.get("world", player)
    environment = WorldConditionEnvironment.model_validate(world)
    from pydantic import TypeAdapter
    world_values = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(world)
    ranked = sorted(InventoryConditionOrder(json.dumps(item.model_dump(mode="json"), sort_keys=True), item)
        for item in parsed.inventory)
    return WorldConditions(comparison_fingerprint=condition_fingerprint(observation), position=parsed.camera.block_position,
        player=WorldConditionPlayer.model_validate(player), inventory=tuple(rank.item for rank in ranked),
        world=environment, day_phase=world_day_phase(world_values),
        life=WorldConditionLife(body_session=parsed.body_session, death_count=parsed.death_count),
        nearby_blocks=parsed.nearby_blocks)


class WorldEvents(BusModule):
    """Observe the body and commit one event before asking cognition to choose.

    The application owns this module and its poll timer. The memory store owns
    accepted observations. A pending event prevents a second queued observation.
    Transport failures stay in the journal and do not establish a world fact.
    Shutdown cancels queued reads and discards a late response before commit.
    """

    DATA_TOPICS = frozenset({"gwt.updated", "action.receipt", "user.input"})

    def __init__(self, *args: Any, ghost: MemoryStore, observe: Callable[[], dict[str, Any]],
                 journal: Path, quiet_seconds: float = 120.0, **kwargs: Any) -> None:
        """Forward the module configuration and retain the observation source."""
        super().__init__(*args, **kwargs)
        self.ghost = ghost
        self.observe = observe
        self.journal = Path(journal)
        self.quiet_seconds = quiet_seconds
        self.pending_id: int | None = None
        self.pending_correlation = ""
        self.last_conditions: WorldConditions | None = None
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
        observation = WorldObservation.model_validate(response["result"])
        conditions = world_conditions(response["result"])
        if conditions == self.last_conditions and not self.retry_observation:
            return
        previous = self.last_conditions
        correlation = "world-event:" + new_id()
        text = render_world_event(WorldEventRenderInput(observation, conditions, silent))
        previous_wire = previous.model_dump(mode="json") if previous is not None else None
        changes = self.ghost.begin_batch("world-observation:" + correlation, cancel=self.stop_event)
        try:
            feature = new_feature(
                self.ghost, text, FeatureType.WorldEvent, source=self.module_id,
                source_kind=SourceKind.Workflow,
                changes=changes,
                metadata={"origin": "minecraft.world", "executed_at": observed_at,
                          "requested_at": requested_at, "user_silent": silent,
                          "evidence": response["result"], "previous_conditions": previous_wire,
                          "correlation_id": correlation, "self_origin": True},
            )
            self.ghost.commit(changes)
        except GhostCancelled:
            return
        self.last_conditions = conditions
        self.retry_observation = False
        self.pending_id, self.pending_correlation = feature.id, correlation
        self._record({"status": "committed", "observed_at": observed_at, "knoxel_id": feature.id,
                      "correlation_id": correlation, "conditions": conditions.model_dump(mode="json"), "previous_conditions": previous_wire})
        self.bus.put(new_envelope(self.module_id, "data", "minecraft.world", correlation,
                                  {"knoxel_id": feature.id}))


__all__ = ["WorldEvents", "world_conditions"]
