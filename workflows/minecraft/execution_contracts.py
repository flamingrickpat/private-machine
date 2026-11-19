from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from typing import Literal
from workflows.minecraft_project.schemas import Node, Condition, Point
from workflows.minecraft_project.planning.context_contracts import (DecisionIdentity, ProjectIntent, RecordedRead,
    AttemptReceipt, GameplayHistory, ProjectCheck, CallerRequirements)


class ExecutionContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CriterionObservations(ExecutionContract):
    identity: DecisionIdentity
    scope: str
    reads: tuple[RecordedRead, ...]
    checks: dict[str, bool]
    original_receipt: str
    original_digest: str


class PrerequisiteEvidence(ExecutionContract):
    verification: str
    certificate_digest: str
    specification: Node
    identity: DecisionIdentity
    checks: dict[str, bool]
    actor_summary: str


class LeafProject(ExecutionContract):
    goal: str
    ancestor_specifications: tuple[Node, ...] = ()
    caller_constraints: CallerRequirements | None = None
    latest_project_check: ProjectCheck | None = None


class PreviousAttempt(ExecutionContract):
    identity: AttemptReceipt
    handoff: str
    journals: tuple[str, ...]
    scope: str


class ActorAcceptance(ExecutionContract):
    criteria: tuple[Condition, ...] = ()
    current_criterion_observations: CriterionObservations | None = None
    prerequisites: dict[str, PrerequisiteEvidence] = Field(default_factory=dict[str, PrerequisiteEvidence])
    project: LeafProject | None = None
    gameplay_history: GameplayHistory | None = None
    full_gameplay_history: str = ""
    history_journals: tuple[str, ...] = ()
    previous_attempt: PreviousAttempt | None = None
    kind: Literal["wood_return"] | None = None
    trees: int = 5
    minimum_travel: float = 16
    return_radius: float = 3


class LeafSpecification(ExecutionContract):
    identity: DecisionIdentity | None
    instruction: str
    criteria: tuple[Condition, ...]
    project_intent: ProjectIntent | None
    current_criterion_observations: CriterionObservations | None
    project: LeafProject | None


class CandidateFurnace(ExecutionContract):
    source: str
    seq: int
    position: Point


class UncertainWorldEffect(ExecutionContract):
    source: str
    seq: int
    tool: str
    arguments: dict[str, JsonValue]
    time: float | None
    candidate_furnaces: tuple[CandidateFurnace, ...] = ()
