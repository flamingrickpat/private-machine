from __future__ import annotations

from dataclasses import dataclass
import math
from datetime import datetime
from pydantic import BaseModel, ConfigDict, JsonValue
from workflows.minecraft_project.schemas import Point
from workflows.minecraft_body.spatial import WorldPosition


class WorldItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    count: int
    slot: int | None = None
    durability: int | None = None
    max_durability: int | None = None


class WorldCamera(BaseModel):
    model_config = ConfigDict(extra="ignore")
    feet_position: WorldPosition
    block_position: Point
    cardinal: str
    yaw: float
    pitch: float


class WorldPlayer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    health: float
    max_health: float
    food: int
    oxygen: int
    dimension: str
    on_ground: bool
    in_water: bool = False
    in_lava: bool | None = None
    alive: bool | None = None
    life_id: str | int | None = None
    nutrition: JsonValue = None
    sleep: JsonValue = None
    food_level: int | None = None
    biome: str = ""
    time_of_day: int = 0
    is_day: bool = False
    is_raining: bool = False


class NearbyBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    block_name: str
    count: int
    nearest: Point
    distance: float


class NearbyEntity(BaseModel):
    model_config = ConfigDict(extra="ignore")
    entity_id: int
    entity_type: str
    position: WorldPosition
    distance: float
    health: float | None = None


class WorldObservation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    camera: WorldCamera
    player: WorldPlayer
    body_session: str = ""
    death_count: int = 0
    held_item: WorldItem | None
    inventory: tuple[WorldItem, ...] = ()
    nearby_blocks: tuple[NearbyBlock, ...] = ()
    nearby_entities: tuple[NearbyEntity, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, order=True, slots=True)
class NearbyOrder:
    distance: float
    index: int


def coordinates(point: Point | WorldPosition) -> str:
    return f"({point.x}, {point.y}, {point.z})"


def block_bearing(origin: Point, target: Point) -> str:
    """Compass bearing between block columns. Positive Minecraft Z is south."""
    east = target.x - origin.x
    north = origin.z - target.z
    if east == 0 and north == 0:
        return "in the body's column"
    directions = ("north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest")
    sector = math.floor(math.atan2(east, north) / (math.pi / 4) + 0.5)
    return directions[sector % 8] + " of the body"


def item_text(item: WorldItem) -> str:
    text = f"{item.count} {item.name}"
    if item.slot is not None:
        text += f"; slot {item.slot}"
    if item.durability is not None:
        text += f"; durability {item.durability}/{item.max_durability}"
    return text


class InventoryChange(BaseModel):
    item: str
    before: int
    after: int


class InventoryDelta(BaseModel):
    changes: tuple[InventoryChange, ...] = ()


class MinedBlock(BaseModel):
    position: Point
    block_name: str


class MinecraftToolResult(BaseModel):
    ok: bool
    reason: str | None = None
    block: MinedBlock | None = None
    block_name: str | None = None
    position: WorldPosition | None = None
    final_position: WorldPosition | None = None
    tool_used: str | None = None
    inventory_delta: InventoryDelta | None = None
    placed_count: int | None = None
    dropped: int | None = None

    def facts(self) -> str:
        fields = ["ok=" + str(self.ok)]
        if self.reason:
            fields.append(self.reason)
        if self.block:
            fields.append(self.block.block_name + " at " + coordinates(self.block.position))
        if self.tool_used:
            fields.append("used " + self.tool_used)
        elif self.block and "tool_used" in self.model_fields_set:
            fields.append("no tool equipped")
        if self.block_name and self.position:
            fields.append(self.block_name + " at " + coordinates(self.position))
        if self.final_position:
            fields.append("ended at " + coordinates(self.final_position))
        if self.placed_count is not None:
            fields.append(f"placed {self.placed_count}")
        if self.dropped is not None:
            fields.append(f"dropped {self.dropped}")
        if self.inventory_delta:
            fields.extend(f"inventory {item.item} {item.before} -> {item.after}" for item in self.inventory_delta.changes)
            if not self.inventory_delta.changes and "changes" in self.inventory_delta.model_fields_set:
                fields.append("no inventory change")
        return ", ".join(fields)


def inventory_text(items: tuple[JsonValue, ...]) -> str:
    return "; ".join(item_text(WorldItem.model_validate(item)) for item in items)


def observation_text(value: JsonValue) -> str:
    return render_world_observation(WorldObservation.model_validate(value))


def render_world_observation(observation: WorldObservation) -> str:
    camera, player = observation.camera, observation.player
    lines = [f"Body {observation.body_session}; deaths {observation.death_count}",
        f"Feet {coordinates(camera.feet_position)}; body cell {coordinates(camera.block_position)}; facing {camera.cardinal}; yaw {camera.yaw}; pitch {camera.pitch}",
        f"Health {player.health}/{player.max_health}; food {player.food}/20; oxygen {player.oxygen}/20; grounded={player.on_ground}; in water={player.in_water}",
        f"Dimension {player.dimension}; biome {player.biome}; world time {player.time_of_day}; day={player.is_day}; raining={player.is_raining}",
        "Held: " + (item_text(observation.held_item) if observation.held_item is not None else "empty"),
        "Owned inventory: " + "; ".join(item_text(item) for item in observation.inventory)]
    if player.in_lava is not None or player.alive is not None:
        lines.append(f"Alive={player.alive}; in lava={player.in_lava}; life ID={player.life_id}")
    if player.nutrition is not None:
        lines.append("Nutrition: " + foreign_operands_text(player.nutrition))
    if player.sleep is not None:
        lines.append("Sleep: " + foreign_operands_text(player.sleep))
    ranked_blocks = sorted(NearbyOrder(block.distance, index) for index, block in enumerate(observation.nearby_blocks))
    for order in ranked_blocks[:12]:
        block = observation.nearby_blocks[order.index]
        lines.append(f"{block.block_name}: {block.count} blocks across the sampled area. "
                     f"Nearest single block at {coordinates(block.nearest)}, {block.distance:.2f} blocks away, "
                     f"{block_bearing(camera.block_position, block.nearest)}.")
    ranked_entities = sorted(NearbyOrder(entity.distance, index) for index, entity in enumerate(observation.nearby_entities))
    for order in ranked_entities[:8]:
        entity = observation.nearby_entities[order.index]
        lines.append(f"Entity {entity.entity_id} {entity.entity_type}: {coordinates(entity.position)}; distance {entity.distance:.2f}; health {entity.health}")
    if len(ranked_blocks) > 12 or len(ranked_entities) > 8:
        lines.append(f"Other distant observations omitted: {max(0, len(ranked_blocks)-12)} block groups, {max(0, len(ranked_entities)-8)} entities. Original observation remains in its receipt.")
    lines.extend(observation.warnings)
    return "\n".join(lines)


def foreign_operands_text(value: JsonValue, indent: int = 0) -> str:
    if isinstance(value, dict):
        lines: list[str] = []
        for name, operand in value.items():
            rendered = foreign_operands_text(operand, indent + 1)
            lines.append("  " * indent + name + ": " + rendered)
        return "\n".join(lines)
    if isinstance(value, list):
        return "\n".join("  " * indent + f"{index + 1}. " + foreign_operands_text(item, indent + 1)
            for index, item in enumerate(value))
    if value is None:
        return "unspecified"
    return str(value)


def world_result_text(tool: str, result: JsonValue) -> str:
    if tool == "minecraft_observe":
        return observation_text(result)
    return foreign_operands_text(result)


class BodyReading(BaseModel):
    model_config = ConfigDict(frozen=True)
    observed_at: datetime
    observation: WorldObservation

    def summary(self) -> str:
        world = self.observation
        player = world.player
        if player.alive is False or player.health <= 0:
            return f"Body observed dead at {self.observed_at.isoformat()}."
        inventory = ", ".join(f"{item.count} {item.name}" for item in world.inventory) or "empty"
        return (f"At {self.observed_at.isoformat()} the body was at {coordinates(world.camera.feet_position)}, "
                f"health {player.health}/{player.max_health}, hunger {player.food}/20, inventory {inventory}.")


class StopObservation(BaseModel):
    run_id: str
    epoch: int
    acknowledged_at: datetime
    uncertain_effects: tuple[str, ...] = ()
    reading: BodyReading
