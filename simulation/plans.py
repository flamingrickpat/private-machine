"""Validated scenario plans with mandatory facts and ordered virtual times.

A plan is the director's editable output and the run's contract: every mandatory
fact belongs to one ordered stage, and each stage names the participant that
receives the input. The models here reject an ambiguous plan before any process
starts: duplicate IDs, a fact without a stage, a stage behind virtual time, or a
fact whose stated time differs from its stage.

Completion is recorded delivery, never a model's claim of success. These models
are persisted in ``plan.json`` and exchanged with the browser, so field names and
defaults are part of the run format. The package overview is in
``simulation/__init__.py``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class Fact(BaseModel):
    """One required scenario fact. Completion needs recorded delivery evidence."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    at: datetime | None = Field(default=None, description="Exact scenario time when the brief states one, including its UTC offset.")


class Step(BaseModel):
    """One scenario stage. It can require several conversational turns."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    at: datetime
    instruction: str = Field(min_length=1)
    fact_ids: list[str] = Field(default_factory=list)
    completion: str = Field(min_length=1)
    target: str = "primary"
    speaker: str = "Jesse"


class ScenarioPlan(BaseModel):
    """Editable director output. Every fact must belong to an ordered stage."""
    model_config = ConfigDict(extra="forbid")
    title: str
    brief: str
    start_time: datetime
    facts: list[Fact] = Field(min_length=1)
    steps: list[Step] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_chronology(self) -> Self:
        """Reject missing facts, duplicate identities, and ambiguous time zones."""
        ids = {item.id for item in self.facts}
        if len(ids) != len(self.facts) or len({item.id for item in self.steps}) != len(self.steps):
            raise ValueError("Fact and stage IDs must be unique.")
        assigned = {key for step in self.steps for key in step.fact_ids}
        if ids != assigned:
            raise ValueError(f"Every required fact needs a stage. Missing: {sorted(ids - assigned)}. Unknown: {sorted(assigned - ids)}.")
        if self.start_time.tzinfo is None or any(item.at.tzinfo is None for item in self.steps):
            raise ValueError("Scenario times need explicit UTC offsets.")
        times = [self.start_time, *(item.at for item in self.steps)]
        if times != sorted(times):
            raise ValueError("Scenario stages must follow virtual time order.")
        facts = {item.id: item for item in self.facts}
        for step in self.steps:
            for key in step.fact_ids:
                at = facts[key].at
                if at is not None and (at.tzinfo is None or at != step.at):
                    raise ValueError(f"Stage {step.id} must retain the stated time of fact {key}.")
        return self


class ExtractedFacts(BaseModel):
    """The first director pass, before chronological planning.

    ``chronology`` keeps the director's raw ``{at, event}`` rows for the planning
    pass. They are JSON values, not a run contract, so they stay unvalidated
    beyond being JSON objects.
    """

    facts: list[Fact] = Field(min_length=1)
    chronology: list[dict[str, JsonValue]] = Field(default_factory=list[dict[str, JsonValue]])


class ActorTurn(BaseModel):
    """One simulated participant message. It cannot control PM behavior."""
    content: str = Field(min_length=1)


class ActorDraftReview(BaseModel):
    """Check the scripted speaker and facts before delivering a draft to PM."""
    accepted: bool
    reason: str


class StageJudgment(BaseModel):
    """Automated evidence check. It does not substitute for human ratings."""
    step_complete: bool
    supported_fact_ids: list[str]
    reason: str
    failure: str | None = None
    fact_evidence: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def cited_deliveries(self) -> Self:
        """Require recorded delivery IDs for every supported scenario fact."""
        if any(not self.fact_evidence.get(key) for key in self.supported_fact_ids):
            raise ValueError("Each supported fact needs its actor delivery IDs in fact_evidence.")
        return self
