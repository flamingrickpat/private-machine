"""Typed contracts for the private thought loop.

Every stage of the loop validates its model output into one of the models in
this file. The models are pure data holders: they never read the MemoryStore
and never call a model. The host derives the request, the stages produce the
reports, and the parent report tells the host what to commit.

See :mod:`workflows.inner_thought.host` for the subsystem overview. The
InquiryContract models describe the graph storage boundary in inquiry and
nominations. Existing linear stage requests remain separate until their
execution path migrates.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pm.memory.dependencies import SourceDependency
from pm.records.base import RecordScope
from pm.records.mental_state import FullMentalState
from workflows.main_cognition.schemas import GoalChange
from workflows.reflection.schemas import ReflectionDecision, ReflectionContext

#: Reality labels a private thought may carry. They mirror the main loop's
#: labels so one vocabulary spans both loops.
REALITY_LABELS = ("observed", "remembered", "internal", "inferred", "imagined")


class ThoughtCandidate(BaseModel):
    """One private attention candidate the trigger policy selected.

    The host derives these deterministically from committed state. The model
    can only narrow the selection, never widen it: a thought may cite a
    candidate by id, and the host rejects ids outside this list.
    """

    knoxel_id: int
    kind: str = "record"
    text: str = ""
    revision: int = 1
    tick_id: int = 0
    urgency: float = 0.0
    reason: str = ""


class TriggerCandidate(BaseModel):
    """One deterministic trigger with its suggested operations."""

    name: str
    cause: str
    urgency: float = 0.0
    operations: list[str] = Field(default_factory=list)
    reason: str = ""


class TriggerChoice(BaseModel):
    """The model's chosen trigger and operation for this private cycle."""

    trigger: str = ""
    operation: str = "none"
    temporal_focus: str = "present"
    candidate_ids: list[int] = Field(default_factory=list)
    urgency: float = 0.0
    reason: str = ""
    retrieval_query: str = ""


class ThoughtAlternative(BaseModel):
    """One alternative a thought weighed, with its explicit disposition."""

    text: str
    disposition: str = "parked"
    reason: str = ""


class ThoughtResult(BaseModel):
    """The performed thought operation and its separation of fact and reading."""

    thought: str = ""
    evidence_ids: list[int] = Field(default_factory=list)
    evidence_summary: str = ""
    interpretation: str = ""
    uncertainty: float = 0.5
    reality: str = "internal"
    alternatives: list[ThoughtAlternative] = Field(default_factory=list)
    continuation: str = "summarize"
    continuation_reason: str = ""
    next_operation: str = ""
    topic_hints: list[str] = Field(default_factory=list)


class ExpectationUpdate(BaseModel):
    """One expectation lifecycle step proposed by the integrate stage.

    ``action`` is ``open`` (record a new anticipation), ``resolve`` (compare a
    pending expectation with an observed outcome), or ``none``. An unobserved
    outcome never resolves to rejection; the model must cite observation ids.
    """

    action: str = "none"
    expectation_id: int | None = None
    anticipation: str = ""
    expected_outcome: str = ""
    expected_valence: float = 0.0
    uncertainty: float = 0.5
    goal_ids: list[int] = Field(default_factory=list)
    status: str = "pending"
    observation_ids: list[int] = Field(default_factory=list)
    reason: str = ""


class DisclosureChoice(BaseModel):
    """What the companion chooses to share from this private work.

    The disclosure intention is recorded before any dialogue generation.
    ``mode`` is ``candid`` or ``withhold``. A deliberate misrepresentation is
    the main loop's presentation choice under A09R and stays out of this
    contract: the private loop only selects what may enter public context.
    """

    mode: str = "withhold"
    shareable_summary: str = ""
    reason: str = ""


class IntegrationResult(BaseModel):
    """The consequences the integrate stage proposes for accepted state."""

    goal_changes: list[GoalChange] = Field(default_factory=list)
    expectation: ExpectationUpdate = Field(default_factory=ExpectationUpdate)
    noticing: str = ""
    disclosure: DisclosureChoice = Field(default_factory=DisclosureChoice)
    summary: str = ""
    progress: str = "none"
    progress_reason: str = ""


class InnerRequest(BaseModel):
    """The frozen input of one private cycle.

    The host derives every field from committed state and records the
    revisions it read. The loop's own attention is the candidate list: the
    model sees only these records as citable evidence.
    """

    run_id: str = ""
    companion_name: str = ""
    user_name: str = ""
    state_summary: str = ""
    #: The accepted mental state of the last cycle, or None before the first cycle.
    mental_state: FullMentalState | None = None
    mental_state_age_seconds: float = 0.0
    seconds_since_user_input: float | None = None
    tick_id: int = 0
    #: The newest main turn: tick, summary, surprise, and ``frame_id``. Since
    #: A19F step 10, ``frame_id`` is the subjective experience of that turn.
    #: A19G gives the inner loop typed fields for this.
    main_frame: dict = Field(default_factory=dict)
    main_frame_age_ticks: int = 0
    active_goal_records: list[dict] = Field(default_factory=list)
    pending_expectations: list[dict] = Field(default_factory=list)
    recent_thoughts: list[dict] = Field(default_factory=list)
    candidates: list[ThoughtCandidate] = Field(default_factory=list)
    trigger_candidates: list[TriggerCandidate] = Field(default_factory=list)
    channels: dict = Field(default_factory=dict)
    operations: list[dict] = Field(default_factory=list)
    #: Selected implementation evidence for a capability or episode inspection.
    #: The host builds it from the real source grant and the real records, so
    #: the model reads a bounded view instead of scanning the repository.
    inspection: dict = Field(default_factory=dict)
    episode: dict = Field(default_factory=dict)
    reflection: ReflectionContext = Field(default_factory=ReflectionContext)


class InnerReport(BaseModel):
    """The parent report the private loop's host commits."""

    run_id: str = ""
    operation: str = "none"
    trigger: str = ""
    trigger_reason: str = ""
    temporal_focus: str = "present"
    candidate_ids: list[int] = Field(default_factory=list)
    thought: str = ""
    evidence_ids: list[int] = Field(default_factory=list)
    evidence_summary: str = ""
    interpretation: str = ""
    uncertainty: float = 0.5
    reality: str = "internal"
    alternatives: list[ThoughtAlternative] = Field(default_factory=list)
    continuation: str = "summarize"
    continuation_reason: str = ""
    goal_changes: list[GoalChange] = Field(default_factory=list)
    expectation: ExpectationUpdate = Field(default_factory=ExpectationUpdate)
    noticing: str = ""
    disclosure: DisclosureChoice = Field(default_factory=DisclosureChoice)
    channels: dict = Field(default_factory=dict)
    summary: str = ""
    next_operation: str = ""
    topic_hints: list[str] = Field(default_factory=list)
    progress: str = "none"
    progress_reason: str = ""
    retrieved_memories: list[dict] = Field(default_factory=list)
    reflection: ReflectionDecision = Field(default_factory=ReflectionDecision)


class InquiryContract(BaseModel):
    """Validate the persisted inquiry boundary without changing legacy requests."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ThoughtJudgments(InquiryContract):
    """Separate exploration value, evidence support, and proposed action value."""

    interestingness: float = Field(ge=0, le=1, description="Value of exploring this possibility, independent of truth.")
    character_relevance: float = Field(ge=0, le=1, description="Importance to this character's present concerns.")
    plausibility: float = Field(ge=0, le=1, description="Revisable support from supplied evidence, never an evidence count.")
    consequence: float = Field(ge=0, le=1, description="Potential effect on understanding or choices if true.")
    desirability: float = Field(ge=0, le=1, description="How welcome the possible outcome is to this character.")
    action_suitability: float = Field(ge=0, le=1, description="Suitability of proposed action under current constraints, without action authority.")
    reason: str = Field(min_length=1, description="Evidence and character reasons for these separate judgments.")


class ThoughtNomination(InquiryContract):
    """A proposed thought, separate from selection and executed elaboration."""

    text: str = Field(min_length=1, description="The incomplete idea or association proposed for private attention.")
    question: str = Field(min_length=1, description="The unresolved question that an inquiry can examine.")
    evidence_ids: tuple[int, ...] = Field(description="Exact supplied source identities; self-generated content remains internal evidence.")
    reality: Literal["remembered", "internal", "inferred", "imagined"]
    judgments: ThoughtJudgments
    suggested_operation: str = Field(min_length=1, description="Delivered operation name suggested to the separate attention owner.")


class InquiryGrant(InquiryContract):
    """Captured observer, reality, source meaning, and Critical epoch."""

    observer_entity_id: int = Field(gt=0, description="Existing companion identity holding this private inquiry.")
    scope: RecordScope
    sources: tuple[SourceDependency, ...] = Field(description="Fingerprints captured before generation, including original source roots.")
    critical_epoch: int = Field(ge=0, description="Accepted Critical epoch at capture; ordinary input generation is separate.")

    @model_validator(mode="after")
    def unique_sources(self) -> Self:
        """Reject contradictory grants rather than silently overwrite dependencies."""
        ids = [source.record_id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("An inquiry grant requires unique source identities.")
        return self


class NominationReceipt(InquiryContract):
    """A retained proposal with its original read grant and source roots."""

    nomination: ThoughtNomination
    grant: InquiryGrant
    root_source_ids: tuple[int, ...] = Field(description="Deduplicated original roots; repeated thought never adds corroboration.")


class ThoughtAttentionDecision(InquiryContract):
    """The attention owner's disposition, without a conscious thought event."""

    nomination_id: int | None = Field(description="Exact supplied nomination ID, or null when the whole batch is deferred or dismissed.")
    disposition: Literal["select", "defer", "dismiss"]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def selected_identity(self) -> Self:
        """Require an exact identity before allocating the private episode slot."""
        if self.disposition == "select" and self.nomination_id is None:
            raise ValueError("Selecting a thought requires its committed nomination identity.")
        return self


class ThoughtAttentionReceipt(InquiryContract):
    """Retain the exact offered batch beside the separate attention decision."""

    nomination_ids: tuple[int, ...]
    decision: ThoughtAttentionDecision
    grant: InquiryGrant


class ThoughtRelation(InquiryContract):
    """One causal relation to an already accepted node in this episode."""

    target_record_id: int = Field(gt=0)
    kind: Literal["extends", "supports", "challenges", "reformulates", "merges", "dismisses", "notices"]
    reason: str = Field(min_length=1, description="Why this executed step relates to the earlier node.")


class InquiryAlternative(InquiryContract):
    """A branch with its own support, assumptions, and epistemic status."""

    text: str = Field(min_length=1)
    evidence_ids: tuple[int, ...]
    assumptions: tuple[str, ...]
    reality: Literal["remembered", "internal", "inferred", "imagined"]
    judgments: ThoughtJudgments
    disposition: Literal["open", "dismissed", "unresolved"]
    reason: str = Field(min_length=1)


class InquiryProgress(InquiryContract):
    """Explicit evidence acquisition or discrimination, never text inequality."""

    kind: Literal["new_evidence", "discrimination", "none"]
    inspected_source_ids: tuple[int, ...] = Field(description="Original newly inspected sources for an evidence acquisition claim.")
    compared_node_ids: tuple[int, ...] = Field(description="Accepted graph nodes compared by a discriminatory operation.")
    meaningful_difference: bool = Field(description="Whether comparison found a changed assumption, prediction, consequence, or acquisition choice.")
    reason: str = Field(min_length=1, description="The concrete distinction, or why this step repeats earlier work.")


class InquiryStep(InquiryContract):
    """One generated operation awaiting guarded staging by the private host."""

    operation: str = Field(min_length=1)
    thought: str = Field(min_length=1)
    evidence_ids: tuple[int, ...]
    assumptions: tuple[str, ...]
    reality: Literal["remembered", "internal", "inferred", "imagined"]
    judgments: ThoughtJudgments
    relations: tuple[ThoughtRelation, ...]
    alternatives: tuple[InquiryAlternative, ...] = Field(max_length=3)
    progress: InquiryProgress
    outcome: Literal["continue", "finding", "information_gap", "wait", "pause", "completed", "abandoned"]
    reason: str = Field(min_length=1, description="Why inquiry continues or stops, including sufficiency and remaining uncertainty.")
    information_gap: str = Field(description="Specific missing evidence; an empty string means no stated information gap.")
    next_operation: str = Field(description="Delivered next operation for continuation; empty at a terminal boundary.")

    @model_validator(mode="after")
    def continuation_contract(self) -> Self:
        """Keep unfinished questions explicit and reject contradictory outcomes."""
        if (self.outcome == "continue") != bool(self.next_operation.strip()):
            raise ValueError("Only a continuing inquiry names its next operation.")
        if self.outcome == "information_gap" and not self.information_gap.strip():
            raise ValueError("An information-gap outcome requires its missing evidence.")
        return self


class InquiryState(InquiryContract):
    """Durable graph state after selection or an accepted operation."""

    episode_id: str = Field(min_length=1)
    origin_record_id: int = Field(gt=0, description="Selected nomination or accepted deliberate action identity.")
    grant: InquiryGrant
    node_ids: tuple[int, ...] = Field(description="Executed operation and branch IDs in causal creation order.")
    frontier_ids: tuple[int, ...] = Field(description="Retained unresolved branches and the latest executed node.")
    pending_generation_ids: tuple[int, ...] = Field(description="Generated nodes that still need discrimination before a finding.")
    root_source_ids: tuple[int, ...]
    inspected_root_ids: tuple[int, ...] = Field(description="Original sources already inspected during this inquiry.")
    sequence: int = Field(ge=0, description="Count of accepted executed operations; selection starts at zero.")
    status: Literal["active", "paused", "completed", "abandoned"]
    stop_reason: str
    started_at_utc: datetime
    active_elapsed_seconds: float = Field(ge=0, allow_inf_nan=False, description="Accumulated monotonic computation duration at accepted boundaries; offline time is excluded.")
    stalled_steps: int = Field(ge=0, description="Consecutive steps without acquired evidence or meaningful discrimination.")
    next_operation: str
    resume_after_utc: datetime | None
    resume_source_ids: tuple[int, ...]
    reflection_source_id: int | None = Field(default=None, gt=0)

    @field_validator("started_at_utc", "resume_after_utc")
    @classmethod
    def naive_utc(cls, value: datetime | None) -> datetime | None:
        """Keep the internal record convention; external adapters normalize offsets."""
        if value is not None and value.tzinfo is not None:
            raise ValueError("Internal inquiry timestamps require naive UTC.")
        return value


class InquiryReceipt(InquiryContract):
    """An append-only state transition with its exact current-parent identity."""

    parent_receipt_id: int | None
    state: InquiryState
    attention: ThoughtAttentionDecision | None
    step: InquiryStep | None
    operation_record_id: int | None
    alternative_record_ids: tuple[int, ...]


class LegacyEpisodeState(InquiryContract):
    """Read the retained linear episode without inventing graph judgments.

    ``elapsed_seconds`` is historical world-clock duration. It is not active
    computation time and cannot initialize the new monotonic duration.
    """

    episode_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    started_at: datetime
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    closed: bool
    closed_reason: str | None
    operations: tuple[str, ...]
    evidence_ids: tuple[int, ...]
    alternatives: tuple[str, ...]
    thoughts: tuple[str, ...]
    progress: tuple[str, ...]
    stalled: int = Field(ge=0)
    continuation: str
    next_operation: str
    parked_until: datetime | None
    parked_sources: tuple[int, ...]
    parent_thought_id: int | None


__all__ = [
    "REALITY_LABELS",
    "DisclosureChoice",
    "ExpectationUpdate",
    "InnerReport",
    "InnerRequest",
    "IntegrationResult",
    "ThoughtAlternative",
    "ThoughtCandidate",
    "ThoughtResult",
    "TriggerCandidate",
    "TriggerChoice",
    "InquiryAlternative", "InquiryGrant", "InquiryProgress", "InquiryReceipt",
    "InquiryState", "InquiryStep", "NominationReceipt", "ThoughtAttentionDecision",
    "ThoughtJudgments", "ThoughtNomination", "ThoughtRelation", "ThoughtAttentionReceipt",
    "LegacyEpisodeState",
]
