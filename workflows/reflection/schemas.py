"""Predictions, interventions, and outcomes for private experiments."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from workflows.cognitive_control.schemas import FocusProposal
from workflows.main_cognition.schemas import ReflectionGuidance


class Contract(BaseModel):
    """Reject unknown fields instead of accepting an unconsumed control."""

    model_config = ConfigDict(extra="forbid")


class EffectiveInputs(Contract):
    """The experiment versions that are in effect for one main cycle.

    ``versions`` maps an experiment id (as text) to its effective record id.
    ``source_ids`` are the effective records and their original sources.
    ``guidance`` holds one entry for each experiment in effect.
    """

    versions: dict[str, int] = Field(default_factory=dict)
    source_ids: list[int] = Field(default_factory=list)
    guidance: list[ReflectionGuidance] = Field(default_factory=list)


class Hypothesis(Contract):
    """One explanation. Confidence does not count as independent support."""

    explanation: str = Field(min_length=1)
    supporting_ids: list[int] = Field(default_factory=list)
    opposing_ids: list[int] = Field(default_factory=list)
    uncertainty: str = Field(min_length=1)
    confidence: float = Field(default=0.5, ge=0, le=1)


class Intervention(Contract):
    """A change through the belief owner or the temporary attention owner.

    A belief target is an existing subjective belief. Attention targets the
    latest advice record, or null when no advice exists. Sources remain intact.
    """

    owner: Literal["attention", "belief"]
    expected_record_id: int | None = None
    focus: FocusProposal | None = None
    belief: str = ""

    @model_validator(mode="after")
    def owner_contract(self):
        if self.owner == "attention" and (self.focus is None or self.belief):
            raise ValueError("An attention intervention needs focus and no belief text.")
        if self.owner == "belief" and (not self.belief.strip() or self.focus is not None
                                      or self.expected_record_id is None):
            raise ValueError("A belief intervention needs an existing belief and revised text.")
        return self


class ExperimentPlan(Contract):
    """A prediction that must commit before its intervention can occur."""

    triggering_episode_id: int
    finding: str = Field(min_length=1)
    diagnosis: Literal["missing_evidence", "implementation_failure", "interpretation", "personal_choice", "successful_pattern"]
    hypotheses: list[Hypothesis] = Field(min_length=2)
    intervention: Intervention
    predicted_effect: str = Field(min_length=1)
    discriminating_observation: str = Field(min_length=1)
    observation_event: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    learning: Literal["new_pattern", "strength_change"] = "new_pattern"


class MonitorResult(Contract):
    """A mismatch or useful pattern with its actual source records."""

    finding: str = ""
    episode_id: int | None = None
    evidence_ids: list[int] = Field(default_factory=list)
    relevant: bool = False
    reason: str


class Assessment(Contract):
    """Compare explanations before the guide chooses an action."""

    hypotheses: list[Hypothesis] = Field(default_factory=list)
    result: Literal["supported", "contradicted", "unresolved"] = "unresolved"
    observation_ids: list[int] = Field(default_factory=list)
    reason: str


class ReflectionDecision(Contract):
    """One lifecycle step. The host checks evidence and commits it atomically."""

    action: Literal["none", "propose", "apply", "await", "evaluate", "retain", "reverse", "revise", "interrupt"] = "none"
    experiment_id: int | None = None
    plan: ExperimentPlan | None = None
    result: Literal["supported", "contradicted", "unresolved"] = "unresolved"
    observation_ids: list[int] = Field(default_factory=list)
    reason: str = ""
