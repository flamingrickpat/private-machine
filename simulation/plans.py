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
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FactKind(StrEnum):
    """An event is observed; a commitment is maintained through choices."""

    Event = "event"
    Commitment = "commitment"


class DirectorSettings(BaseModel):
    """Owner guidance for daily planning and natural actor behavior."""

    model_config = ConfigDict(extra="forbid")
    duration_days: int = Field(default=1, ge=1, le=366)
    scenes_per_day: int = Field(default=3, ge=1, le=24)
    minimum_turns_per_scene: int = Field(default=2, ge=1)
    first_hour: int = Field(default=8, ge=0, le=23)
    last_hour: int = Field(default=22, ge=0, le=23)
    daily_rhythm: str = "Ordinary work, meals, sleep, and time away from the phone."
    story_threads: str = ""
    scheduled_events: str = ""
    guidance: str = "Let conversations breathe. Mix ordinary chat with consequential events."
    actor_guidance: str = "Respond naturally and preserve the user's established voice."

    @model_validator(mode="after")
    def ordered_hours(self) -> Self:
        """Reject an ambiguous daily interval."""
        if self.last_hour < self.first_hour:
            raise ValueError("The last conversation hour must follow the first hour.")
        return self


class Fact(BaseModel):
    """One required scenario fact. Completion needs recorded delivery evidence."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    kind: FactKind = Field(description="Whether evidence establishes an event or a continuing actor commitment.")
    at: datetime | None = Field(default=None, description="Exact scenario time when the brief states one, including its UTC offset.")


class Step(BaseModel):
    """One scenario stage. It can require several conversational turns."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    at: datetime
    instruction: str = Field(min_length=1)
    actor_intent: str = Field(min_length=1, description="Private actor goal; never a mandatory line to recite.")
    fact_ids: list[str] = Field(default_factory=list)
    completion: str = Field(min_length=1)
    target: str = "primary"
    speaker: str = "Jesse"
    minimum_turns: int = Field(default=1, ge=1, description="Minimum actor turns before the director can finish this scene.")


class ScenarioPlan(BaseModel):
    """Editable director output. Every fact must belong to an ordered stage."""
    model_config = ConfigDict(extra="forbid")
    title: str
    brief: str
    start_time: datetime
    facts: list[Fact] = Field(min_length=1)
    steps: list[Step] = Field(min_length=1)
    director: DirectorSettings = Field(default_factory=DirectorSettings)

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


class ChronologyEvent(BaseModel):
    """One scheduled event extracted before planning."""

    at: datetime
    event: str


def facts_for_scene(plan: ScenarioPlan, step: Step) -> list[Fact]:
    """Expose current requirements and established commitments, never later plot."""
    visible: list[Fact] = []
    for fact in plan.facts:
        current = fact.id in step.fact_ids
        active = fact.kind == FactKind.Commitment and fact.at is not None and fact.at <= step.at
        if current or active:
            visible.append(fact)
    return visible


class ExtractedFacts(BaseModel):
    """The first director pass, before chronological planning.

    ``chronology`` preserves scheduled events for the chronological planning pass.
    """

    facts: list[Fact] = Field(min_length=1)
    chronology: list[ChronologyEvent] = Field(default_factory=list[ChronologyEvent])


class ActorTurn(BaseModel):
    """One simulated participant message. It cannot control PM behavior."""
    content: str = Field(min_length=1)


class ActorDraftReview(BaseModel):
    """Check immersion and scenario consistency before public delivery."""
    accepted: bool
    voice_faithful: bool
    identity_correct: bool
    natural_dialogue: bool
    scenario_consistent: bool
    reason: str

    @model_validator(mode="after")
    def consistent_verdict(self) -> Self:
        """A failed immersion check cannot be accepted by a separate verdict."""
        if self.accepted != all((self.voice_faithful, self.identity_correct,
                                 self.natural_dialogue, self.scenario_consistent)):
            raise ValueError("Acceptance must agree with all four review checks.")
        return self


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
