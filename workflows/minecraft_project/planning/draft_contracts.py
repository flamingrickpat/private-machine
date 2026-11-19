from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from workflows.minecraft_project.schemas import Condition, Node, Point


class CellBox(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min: Point
    max: Point


class ConditionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = ""
    description: str = ""
    kind: Literal["inventory", "block", "region", "position", "alive", "food", "health", "accepted"]
    node_id: str = ""
    names: list[str] = Field(default_factory=list[str])
    minimum: int = 1
    position: Point | None = None
    positions: list[Point] = Field(default_factory=list[Point])
    end: Point | None = None
    radius: float = 2.0
    dimension: str = "overworld"
    box: CellBox | None = None


class DraftNode(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    revision: int = 1
    kind: Literal["goal", "story", "leaf"]
    parent: str | None = None
    children: list[str] = Field(default_factory=list[str])
    depends_on: list[str] = Field(default_factory=list[str])
    instruction: str
    specialist: Literal["gatherer", "navigator", "recovery", "builder", "miner"] = "gatherer"
    deferred: bool = False
    defer_reason: str = ""
    criteria: list[Condition] = Field(default_factory=list[Condition])
    assumptions: list[Condition] = Field(default_factory=list[Condition])

    @classmethod
    def from_node(cls, node: Node) -> DraftNode:
        return cls(id=node.id, revision=node.revision, kind=node.kind, parent=node.parent,
            children=list(node.children), depends_on=list(node.depends_on), instruction=node.instruction,
            specialist=node.specialist, deferred=node.deferred, defer_reason=node.defer_reason,
            criteria=[row.model_copy(deep=True) for row in node.criteria],
            assumptions=[row.model_copy(deep=True) for row in node.assumptions])

    def specification(self) -> Node:
        return Node(id=self.id, revision=self.revision, kind=self.kind, parent=self.parent,
            children=list(self.children), depends_on=list(self.depends_on), instruction=self.instruction,
            specialist=self.specialist, deferred=self.deferred, defer_reason=self.defer_reason,
            criteria=list(self.criteria), assumptions=list(self.assumptions))


class NodeEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["goal", "story", "leaf"] | None = None
    depends_on: list[str] | None = None
    instruction: str | None = None
    specialist: Literal["gatherer", "navigator", "recovery", "builder", "miner"] | None = None
    deferred: bool | None = None
    defer_reason: str | None = None
    criteria: list[Condition] | None = None
    assumptions: list[Condition] | None = None


class TaskAddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["story", "leaf"]
    instruction: str
    parent: str = ""
    specialist: Literal["gatherer", "navigator", "recovery", "builder", "miner"] = "gatherer"
    depends_on: list[str] = Field(default_factory=list[str])


class ConditionSetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str
    condition: ConditionInput
    assumption: bool = False


class ConditionCopyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_id: str
    source_id: str
    condition_ids: list[str] = Field(default_factory=list[str])


class PlanOutcomeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["planned", "survey", "waiting", "blocked"]
    summary: str
    queries: list[Condition] = Field(default_factory=list[Condition])
    recipe_queries: list[str] = Field(default_factory=list[str])
    restore_nodes: list[str] = Field(default_factory=list[str])


class DraftToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool = True
    node_id: str = ""
    parent: str = ""
    changed_fields: tuple[str, ...] = ()
    condition_id: str = ""
    exact_cells: int = 0
    copied_conditions: tuple[str, ...] = ()
    task: DraftNode | None = None
    error: str = ""


class DraftSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    goal_id: str
    nodes: tuple[DraftNode, ...]
    status: Literal["planned", "survey", "waiting", "blocked"]
    summary: str
    queries: tuple[Condition, ...]
    recipe_queries: tuple[str, ...]
    restore_nodes: tuple[str, ...]
