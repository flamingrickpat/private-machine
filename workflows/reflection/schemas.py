"""Predictions, interventions, and outcomes for private experiments."""
from typing import Literal, cast, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from workflows.cognitive_control.schemas import FocusProposal
from workflows.main_cognition.schemas import ReflectionGuidance
from workflows.cognitive_control.schemas import FocusState
from pm.context_render.records import RecordEvidence, RecordArchitecture as RecordArchitecture
from pm.records.global_workspace import CodeletRunRef


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
    source_ids: list[int] = Field(default_factory=list[int])
    guidance: list[ReflectionGuidance] = Field(default_factory=list[ReflectionGuidance])


class Hypothesis(Contract):
    """One explanation. Confidence does not count as independent support."""

    explanation: str = Field(min_length=1)
    supporting_ids: list[int] = Field(default_factory=list[int])
    opposing_ids: list[int] = Field(default_factory=list[int])
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
    def owner_contract(self) -> Self:
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
    evidence_ids: list[int] = Field(default_factory=list[int])
    relevant: bool = False
    reason: str


class Assessment(Contract):
    """Compare explanations before the guide chooses an action."""

    hypotheses: list[Hypothesis] = Field(default_factory=list[Hypothesis])
    result: Literal["supported", "contradicted", "unresolved"] = "unresolved"
    observation_ids: list[int] = Field(default_factory=list[int])
    reason: str


class ReflectionDecision(Contract):
    """One lifecycle step. The host checks evidence and commits it atomically."""

    action: Literal["none", "propose", "apply", "await", "evaluate", "retain", "reverse", "revise", "interrupt"] = "none"
    experiment_id: int | None = Field(default=None, description="Stable experiment_id from the supplied summary, not its current ledger record_id.")
    plan: ExperimentPlan | None = None
    result: Literal["supported", "contradicted", "unresolved"] = "unresolved"
    observation_ids: list[int] = Field(default_factory=list[int])
    reason: str = ""


ExperimentState = Literal["proposed", "applied", "awaiting_evidence", "evaluated", "retained",
                          "reversed", "revised", "interrupted"]
ExperimentResult = Literal["supported", "contradicted", "unresolved"]


class PreviousIntervention(Contract):
    """Exact owner state needed to reverse an intervention as a new knoxel."""

    record_id: int | None = None
    focus: FocusState = Field(default_factory=FocusState)
    expires_at: str | None = None
    text: str = ""
    supporting_ids: list[int] = Field(default_factory=list[int])
    opposing_ids: list[int] = Field(default_factory=list[int])


class ExperimentEvent(Contract):
    """One persisted lifecycle transition. An unobserved result stays unresolved."""

    action: ExperimentState
    reason: str
    tick_id: int
    at: str
    effective_record_id: int | None = None
    reversal_record_id: int | None = None
    replacement_experiment_id: int | None = None
    result: ExperimentResult = "unresolved"
    observation_ids: list[int] = Field(default_factory=list[int])


class ExperimentMetadata(Contract):
    """The complete experiment ledger stored at the memory boundary."""

    schema_version: Literal["pm.reflection-experiment.v1"] = Field(default="pm.reflection-experiment.v1", alias="schema")
    state: ExperimentState
    experiment_id: int | None = Field(default=None, description="The original proposal knoxel ID, stable across ledger successors.")
    plan: ExperimentPlan
    source_revisions: dict[int, int] = Field(description="Exact source IDs and their captured revisions.")
    previous: PreviousIntervention
    revised_from_experiment_id: int | None = None
    effective_record_id: int | None = None
    reversal_record_id: int | None = None
    replacement_experiment_id: int | None = None
    reviewed_source_id: int = 0
    applied_tick: int = 0
    observation_ids: list[int] = Field(default_factory=list[int])
    result: ExperimentResult = "unresolved"
    history: list[ExperimentEvent] = Field(default_factory=list[ExperimentEvent])
    memory_kind: str = "executed_event"
    executed_at: str | None = None
    head_source_ids: list[int] = Field(default_factory=list[int], description="Sources that were current heads when the prediction was proposed.")


class ReflectionSource(RecordEvidence):
    """Original citable evidence with a deliberate architectural projection."""

    metadata: RecordArchitecture = Field(default_factory=RecordArchitecture)
    belief_kind: str = ""
    active: bool | None = None
    supporting_ids: list[int] = Field(default_factory=list[int])
    opposing_ids: list[int] = Field(default_factory=list[int])


class ObservationGrant(Contract):
    """Later consuming decisions and eligible observations. This grants no truth label."""

    consumer_ids: list[int] = Field(default_factory=list[int])
    observation_ids: list[int] = Field(default_factory=list[int])


class ExperimentEvidence(Contract):
    """One frozen ledger entry and its exact evidence grant."""

    record_id: int
    experiment_id: int
    revision: int
    metadata: ExperimentMetadata
    stale_source_ids: list[int] = Field(default_factory=list[int])
    observation_grant: ObservationGrant = Field(default_factory=ObservationGrant)

    @model_validator(mode="before")
    @classmethod
    def migrate_flat_receipt(cls, value: object) -> object:
        """Read the prior flat request format at the persisted artifact boundary."""
        if isinstance(value, dict) and "metadata" not in value:
            legacy = LegacyExperimentEvidence.model_validate(value)
            return cls(record_id=legacy.record_id, experiment_id=legacy.experiment_id or legacy.record_id, revision=legacy.revision,
                metadata=ExperimentMetadata.model_validate(legacy.model_dump(by_alias=True,
                    exclude={"record_id", "revision", "stale_source_ids", "observation_grant"})),
                stale_source_ids=legacy.stale_source_ids, observation_grant=legacy.observation_grant)
        return cast(object, value)


class LegacyExperimentEvidence(ExperimentMetadata):
    """The flat experiment envelope written before typed reflection contexts."""

    record_id: int
    revision: int
    stale_source_ids: list[int] = Field(default_factory=list[int])
    observation_grant: ObservationGrant = Field(default_factory=ObservationGrant)


class AttentionMetadata(BaseModel):
    """The attention owner's persisted fields needed for reversible reflection."""

    model_config = ConfigDict(extra="ignore")
    focus: FocusState = Field(default_factory=FocusState)
    expires_at: str | None = None


class EpisodeAlternative(Contract):
    """A recorded alternative, with no invented reason or confidence score."""

    text: str
    disposition: str


class EpisodeArchitecture(BaseModel):
    """Useful episode details projected from the existing introspection envelope."""

    model_config = ConfigDict(extra="ignore")
    trigger: str = ""
    reality: str = ""
    alternatives: list[EpisodeAlternative] = Field(default_factory=list[EpisodeAlternative])
    uncertainty: str | float | None = None
    expectation: str = ""
    execution_mode: str = ""
    codelet_runs: list[CodeletRunRef] = Field(default_factory=list[CodeletRunRef])
    structure_ids: list[int] = Field(default_factory=list[int])
    causal_percept_ids: list[int] = Field(default_factory=list[int])


class ReflectionEpisode(BaseModel):
    """An episode's original sources and recorded outcome, without raw bookkeeping."""

    model_config = ConfigDict(extra="ignore")
    episode_id: int
    kind: str
    loop: str
    tick_id: int
    operation: str
    owner: str
    summary: str
    outcome: str
    revision: int = 1
    inputs: list[int] = Field(default_factory=list[int])
    outputs: list[int] = Field(default_factory=list[int])
    links: dict[str, list[int]] = Field(default_factory=dict)
    metadata: EpisodeArchitecture = Field(default_factory=EpisodeArchitecture)


class ReflectionContext(Contract):
    """Frozen typed evidence shared by reflection validation and prompt selection."""

    experiments: list[ExperimentEvidence] = Field(default_factory=list[ExperimentEvidence])
    omitted_experiments: int = 0
    attention_head_id: int | None = None
    review_due: bool = False
    newest_observation_id: int = 0
    episodes: list[ReflectionEpisode] = Field(default_factory=list[ReflectionEpisode])
    episode_limits: list[str] = Field(default_factory=list[str])
    records: list[ReflectionSource] = Field(default_factory=list[ReflectionSource])
