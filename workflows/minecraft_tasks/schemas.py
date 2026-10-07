"""Contracts for open-ended goals, stable subtasks, and observed acceptance."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    """Reject unknown fields before a proposal can change the durable tree."""

    model_config = ConfigDict(extra="forbid")


class Point(Contract):
    """An integer block cell in the named game dimension."""

    x: int
    y: int
    z: int


class Condition(Contract):
    """A fact that an independent read can establish.

    Inventory uses an item-name glob. Regions require complete visible coverage.
    Block conditions permit explicit air cells. Position distances use blocks.
    Acceptance records past attainment; assumptions require current truth.
    Accepted references require an original independent node certificate.
    """

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    kind: Literal["inventory", "block", "region", "position", "alive", "accepted"]
    node_id: str = ""
    names: list[str] = Field(default_factory=list, description="Exact MCP block names or inventory globs. The vanilla minecraft: prefix is optional.")
    minimum: int = Field(default=1, ge=0, description="Count threshold for inventory or region only. A block condition requires every specified cell.")
    position: Point | None = None
    positions: list[Point] = Field(default_factory=list)
    end: Point | None = Field(default=None, description="Inclusive opposite corner for region only. For exact blocks, use positions instead.")
    radius: float = Field(default=2.0, ge=0)
    dimension: str = "overworld"

    @model_validator(mode="after")
    def usable(self) -> Condition:
        """Require the operands needed by each supported observation."""
        self.names = [name.removeprefix("minecraft:") for name in self.names]
        self.dimension = self.dimension.removeprefix("minecraft:")
        if (self.kind == "accepted") != bool(self.node_id):
            raise ValueError("Only an accepted condition needs an explicit source node_id.")
        if self.kind == "accepted" and (self.names or self.position or self.positions or self.end or self.minimum != 1):
            raise ValueError("An accepted condition cites a node certificate, not world operands or count thresholds.")
        if self.kind in {"inventory", "block", "region"} and not self.names:
            raise ValueError("This condition needs explicit item or block names.")
        if self.kind in {"region", "position"} and self.position is None:
            raise ValueError("This condition needs an explicit position.")
        if self.kind == "block" and (self.position is None) == (not self.positions):
            raise ValueError("A block condition needs one position or a nonempty position list.")
        if self.kind == "block" and (self.end is not None or self.minimum != 1):
            raise ValueError("A block condition requires exact cells. Use positions, not end or a count threshold.")
        if len({(point.x, point.y, point.z) for point in self.positions}) != len(self.positions):
            raise ValueError("A block condition cannot repeat a position.")
        if self.kind == "region" and self.end is None:
            raise ValueError("A region needs both inclusive corners.")
        return self


class Node(Contract):
    """A goal, story, or meaningful leaf outcome with stable identity."""

    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    revision: int = Field(default=1, ge=1)
    kind: Literal["goal", "story", "leaf"]
    parent: str | None = None
    children: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    instruction: str = Field(min_length=1)
    specialist: Literal["gatherer", "navigator", "recovery", "builder", "miner"] = "gatherer"
    criteria: list[Condition] = Field(min_length=1)
    assumptions: list[Condition] = Field(default_factory=list)


class TaskTree(Contract):
    """A complete specification snapshot. Execution evidence lives elsewhere."""

    schema_version: Literal["pm.minecraft-task-tree.v1"] = "pm.minecraft-task-tree.v1"
    goal_id: str
    revision: int = Field(default=1, ge=1)
    nodes: list[Node] = Field(min_length=1)


class ProjectRequest(Contract):
    """Start a generic project, or inspect its reviewed plan without execution."""

    run_id: str = ""
    instruction: str = ""
    goal: str = ""
    artifact_path: str = ""
    answer: str = ""
    question_id: str = ""
    resume_phase: str = ""
    body_id: str = "body"
    persona_context: dict = Field(default_factory=dict)
    plan_only: bool = False
    proposed_tree: TaskTree | None = None
    include_image: bool = False


class PlanProposal(Contract):
    """Supply a tree, request world reads, or report an external blocker."""

    status: Literal["planned", "survey", "waiting", "blocked"]
    summary: str
    tree: TaskTree | None = None
    queries: list[Condition] = Field(default_factory=list, description="Read-only world operands. Survey thresholds do not establish completion.")

    @model_validator(mode="after")
    def survey_scope(self) -> PlanProposal:
        """Require explicit world operands before a planning survey can run."""
        if self.status == "survey":
            if self.tree is not None or not self.queries or any(row.kind == "accepted" for row in self.queries):
                raise ValueError("A survey needs world queries and cannot propose a tree or certificate lookup.")
            if len({row.id for row in self.queries}) != len(self.queries):
                raise ValueError("Survey query IDs must be unique.")
        elif self.queries:
            raise ValueError("Only a survey proposal can request world queries.")
        return self


class PlanReview(Contract):
    """An independent scope and acceptance review of an exact tree revision."""

    status: Literal["accepted", "refine", "revise", "waiting", "blocked"]
    summary: str
    target_id: str = ""


class VerificationReview(Contract):
    """Interpret observed facts without replacing the mechanical evidence gate."""

    status: Literal["accepted", "replan", "waiting", "blocked"]
    summary: str
    criterion_ids: list[str]
