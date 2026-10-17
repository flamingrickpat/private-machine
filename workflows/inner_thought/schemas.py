"""Typed contracts for the private thought loop.

Every stage of the loop validates its model output into one of the models in
this file. The models are pure data holders: they never read the MemoryStore
and never call a model. The host derives the request, the stages produce the
reports, and the parent report tells the host what to commit.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from workflows.main_cognition.schemas import GoalChange
from workflows.reflection.schemas import ReflectionDecision

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
    companion_name: str = "Spamuel"
    user_name: str = "Jesse"
    state_summary: str = ""
    mental_state_latent_vec: list[float] = Field(default_factory=list)
    mental_state_age_seconds: float = 0.0
    seconds_since_user_input: float | None = None
    tick_id: int = 0
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
    reflection: dict = Field(default_factory=dict)


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
]
