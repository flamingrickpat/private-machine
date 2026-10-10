"""Contracts for open-ended goals, stable subtasks, and observed acceptance."""
from __future__ import annotations

from collections import Counter
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
    kind: Literal["inventory", "block", "region", "position", "alive", "food", "health", "accepted"]
    node_id: str = ""
    names: list[str] = Field(default_factory=list, description="Exact MCP block names or inventory globs. The vanilla minecraft: prefix is optional.")
    minimum: int = Field(default=1, ge=0, description="Inclusive count, food, or health threshold. Food and health use the observed 0..20 scale. A block condition requires every specified cell.")
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
        if self.kind in {"block", "region"} and any(any(char in name for char in "*?[") for name in self.names):
            raise ValueError("Block and region conditions need exact block names. Only inventory names accept globs.")
        if self.kind in {"region", "position"} and self.position is None:
            raise ValueError("This condition needs an explicit position.")
        if self.kind == "block" and (self.position is None) == (not self.positions):
            raise ValueError("A block condition needs one position or a nonempty position list.")
        if self.kind == "block" and (self.end is not None or self.minimum != 1):
            raise ValueError("A block condition requires exact cells. Use positions, not end or a count threshold.")
        duplicates = sorted(cell for cell, count in Counter((point.x, point.y, point.z) for point in self.positions).items() if count > 1)
        if duplicates:
            raise ValueError(f"A block condition cannot repeat a position. Condition {self.id}, repeated cells: {duplicates}. List each cell once.")
        if self.kind == "region" and self.end is None:
            raise ValueError("A region needs both inclusive corners.")
        if self.kind in {"food", "health"} and (self.names or self.position or self.positions or self.end or self.minimum > 20):
            raise ValueError(f"A {self.kind} condition uses minimum on the observed 0..20 scale, without item or position operands.")
        return self


class Node(Contract):
    """A goal, story, or meaningful leaf outcome with stable identity."""

    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    revision: int = Field(default=1, ge=1)
    kind: Literal["goal", "story", "leaf"]
    parent: str | None = Field(default=None, description="Containing goal or story ID. Only the root goal uses null; copy the actual parent in a replacement node.")
    children: list[str] = Field(default_factory=list, description="Every contained child ID. Leaves use an empty list; replacement parents retain their actual child references.")
    depends_on: list[str] = Field(default_factory=list, description="Explicit prerequisite node IDs. A leaf citing an accepted node certificate must include that source ID directly, even when another prerequisite also depends on it.")
    instruction: str = Field(min_length=1)
    specialist: Literal["gatherer", "navigator", "recovery", "builder", "miner"] = "gatherer"
    deferred: bool = Field(default=False, description="Keep this unfinished leaf and its effects while other useful work proceeds.")
    defer_reason: str = ""
    criteria: list[Condition] = Field(min_length=1)
    assumptions: list[Condition] = Field(default_factory=list)

    @model_validator(mode="after")
    def deferral(self) -> Node:
        """Require a concrete reason for an unfinished leaf's deferral."""
        if self.deferred and (self.kind != "leaf" or not self.defer_reason.strip()):
            raise ValueError("Only a leaf with an explicit reason can be deferred.")
        return self


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


class EvidenceRead(Contract):
    """Read a window from an original source in the workflow evidence catalog."""

    source: str
    start_line: int = Field(default=1, ge=1)
    line_count: int = Field(default=100, ge=1, le=1000)
    query: str = ""


class EvidenceDecision(Contract):
    """Request original evidence before the workflow accepts a decision."""

    evidence_requests: list[EvidenceRead] = Field(default_factory=list)


class Renewal(Contract):
    """Give a superseded current prerequisite a separate physical outcome."""

    source_node_id: str
    node: Node


class RetentionChoice(EvidenceDecision):
    """Select original proof; the workflow derives exact current renewals."""

    status: Literal["planned", "unchanged", "waiting", "blocked"]
    summary: str
    restore_nodes: list[str] = Field(default_factory=list)


class RetentionProposal(RetentionChoice):
    """Restore original proof and preserve superseded current requirements."""

    renewals: list[Renewal] = Field(default_factory=list)


class PlanProposal(EvidenceDecision):
    """Supply a tree, request world reads, or report an external blocker."""

    status: Literal["planned", "survey", "waiting", "blocked"]
    summary: str
    tree: TaskTree | None = None
    updates: list[Node] = Field(default_factory=list, description="Complete replacements or additions by stable node ID. Use only with a previous tree; omitted nodes remain unchanged.")
    restore_nodes: list[str] = Field(default_factory=list, description="Stable IDs from retained_acceptance_candidates. Restore their exact historical specifications and proof instead of rewriting them or repeating consumed work.")
    queries: list[Condition] = Field(default_factory=list, description="Read-only world operands. Survey thresholds do not establish completion.")
    recipe_queries: list[str] = Field(default_factory=list, max_length=10, description="Read-only recipe-search patterns when current recipe evidence is missing or names need correction.")

    @model_validator(mode="after")
    def survey_scope(self) -> PlanProposal:
        """Keep tree changes, read-only surveys, and blockers distinct."""
        if self.status == "planned":
            if (self.tree is not None) == bool(self.updates or self.restore_nodes):
                raise ValueError("A planned result needs exactly one full tree or explicit node updates/restorations.")
            ids = [row.id for row in self.updates] + self.restore_nodes
            if len(set(ids)) != len(ids):
                raise ValueError("Each node update ID must be unique.")
        elif self.tree is not None or self.updates or self.restore_nodes:
            raise ValueError("Only a planned result can propose a tree or node updates.")
        if self.status == "survey":
            if self.tree is not None or not (self.queries or self.recipe_queries) or any(row.kind == "accepted" for row in self.queries):
                raise ValueError("A survey needs world or recipe queries and cannot propose a tree or certificate lookup.")
            if len({row.id for row in self.queries}) != len(self.queries):
                raise ValueError("Survey query IDs must be unique.")
        elif self.queries or self.recipe_queries:
            raise ValueError("Only a survey proposal can request world or recipe queries.")
        if any(not pattern.strip() for pattern in self.recipe_queries):
            raise ValueError("Recipe queries need nonempty search patterns.")
        return self


class RevisionProposal(EvidenceDecision):
    """Revise a retained tree through explicit node updates only."""

    status: Literal["planned", "survey", "waiting", "blocked"]
    summary: str
    updates: list[Node] = Field(default_factory=list, description="Complete replacements or additions by stable ID. Omitted nodes remain unchanged. Retain every existing root child when replacing its parent.")
    restore_nodes: list[str] = Field(default_factory=list, description="Explicitly restore these retained_acceptance_candidates, including their changed dependencies. Use IDs rather than copying or paraphrasing historical specifications.")
    queries: list[Condition] = Field(default_factory=list)
    recipe_queries: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def scope(self) -> RevisionProposal:
        """Apply the proposal checks without allowing a replacement full tree."""
        PlanProposal(status=self.status, summary=self.summary, updates=self.updates,
                     restore_nodes=self.restore_nodes,
                     queries=self.queries, recipe_queries=self.recipe_queries)
        return self


class RefinementProposal(EvidenceDecision):
    """Replace the named node explicitly and add only new descendants."""

    status: Literal["planned", "survey", "waiting", "blocked"]
    summary: str
    replacement: Node | None = Field(description="The complete replacement for refinement_target, preserving its stable ID, parent, and acceptance conditions. Use null for a survey or blocker.")
    additions: list[Node] = Field(default_factory=list, description="New descendant nodes only. Do not copy existing unrelated nodes.")
    queries: list[Condition] = Field(default_factory=list)
    recipe_queries: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def scope(self) -> RefinementProposal:
        """Separate an explicit replacement from surveys and external waits."""
        if self.status == "planned":
            if self.replacement is None or self.queries or self.recipe_queries:
                raise ValueError("A planned refinement needs replacement and cannot request a survey.")
            ids = [self.replacement.id, *(node.id for node in self.additions)]
            if len(set(ids)) != len(ids):
                raise ValueError("Replacement and addition IDs must be unique.")
        elif self.replacement is not None or self.additions:
            raise ValueError("Only a planned refinement can replace or add nodes.")
        if self.status == "survey":
            if not (self.queries or self.recipe_queries) or any(row.kind == "accepted" for row in self.queries):
                raise ValueError("A refinement survey needs explicit world or recipe reads.")
            if len({row.id for row in self.queries}) != len(self.queries):
                raise ValueError("Survey query IDs must be unique.")
        elif self.queries or self.recipe_queries:
            raise ValueError("Only a survey can request world or recipe reads.")
        if any(not pattern.strip() for pattern in self.recipe_queries):
            raise ValueError("Recipe queries need nonempty search patterns.")
        return self


class PlanReview(EvidenceDecision):
    """An independent scope and acceptance review of an exact tree revision."""

    status: Literal["accepted", "refine", "revise", "waiting", "blocked"]
    summary: str
    target_id: str = ""


class VerificationReview(EvidenceDecision):
    """Interpret observed facts without replacing the mechanical evidence gate."""

    status: Literal["accepted", "replan", "waiting", "blocked"]
    summary: str
    criterion_ids: list[str]
