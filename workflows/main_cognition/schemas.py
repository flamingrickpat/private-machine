"""Typed contracts for the main cognitive cycle.

Every stage boundary crosses here. Model output is validated into these
schemas once; the parent finalizer, not the model, builds knoxels. No stage
stores free-form unvalidated model text as canonical memory.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Stimulus(BaseModel):
    """One incoming item for one cycle, resolved from a bus envelope."""

    knoxel_id: int
    source: str = ""
    text: str = ""
    correlation_id: str = ""
    kind: str = "user.input"


class StageReceipt(BaseModel):
    """Evidence that one child stage ran for this cycle."""

    name: str
    status: str
    artifact: str = ""
    summary: str = ""


class ActionDispatch(BaseModel):
    """Typed application dispatch record (plan: workflows.md, action dispatch).

    ``content`` states the intent of the action, not a finished companion
    reply. ``depends_on`` names action IDs that must be released earlier in the
    same selection; it only orders dispatch. ``requires_completed`` names
    action IDs whose successful completion receipt is required before this
    action can start. A queued prerequisite never releases dependent work.
    """

    action_id: str
    workflow_name: str
    action_type: str = "Reply"
    content: str = ""
    arguments: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    requires_completed: list[str] = Field(default_factory=list)
    gwt_tick_id: int | None = None
    status: str = "proposed"  # proposed | dispatched | queued | rejected
    rejection_reason: str = ""


class TriageItem(BaseModel):
    knoxel_id: int
    category: str
    input_kind: str = "message"
    urgency: float = 0.0
    relevance: float = 0.0
    reasoning: str = ""
    evidence_ids: list[int] = Field(default_factory=list)


class TriageResult(BaseModel):
    items: list[TriageItem] = Field(default_factory=list)


class AppraisalResult(BaseModel):
    feeling: str = ""
    emotion_delta: dict[str, float] = Field(default_factory=dict)
    need_delta: dict[str, float] = Field(default_factory=dict)
    appraisal: dict[str, float] = Field(
        default_factory=dict,
        description="Optional appraisal axes. Keys match AppraisalGeneral or AppraisalSocial fields.",
    )
    expectation_error: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="Observed minus anticipated outcome. Positive is better than expected.",
    )
    reasoning: str = ""
    evidence_ids: list[int] = Field(default_factory=list)


class ContextResult(BaseModel):
    memories: list[str] = Field(default_factory=list)
    active_goals: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    candidate_ids: list[int] = Field(default_factory=list)
    belief_updates: list["BeliefUpdate"] = Field(default_factory=list)


class BeliefUpdate(BaseModel):
    """One reported user claim, copied from original evidence.

    A correction replaces beliefs with the same scoped key. The runtime keeps
    the prior beliefs and rejects quotes from generated or imagined sources.
    """

    source_id: int
    quote: str = Field(min_length=1)
    belief_key: str = Field(min_length=1)
    correction: bool = False


class AttentionNomination(BaseModel):
    target_ids: list[int] = Field(default_factory=list)
    attention_kind: str = "goal_relevance"
    score: float = 0.5
    novelty: float = 0.0
    urgency: float = 0.0
    goal_relevance: float = 0.0
    reportability: float = 0.0
    reasoning: str = ""


class AttentionResult(BaseModel):
    nominations: list[AttentionNomination] = Field(default_factory=list)
    selected_ids: list[int] = Field(default_factory=list)
    reasoning: str = ""


class GoalChange(BaseModel):
    change: str  # retain | add | complete | reprioritize
    goal: str = ""
    goal_key: str = ""  # stable identity; falls back to normalized goal text
    reason: str = ""
    priority: float = 0.5


class GoalsResult(BaseModel):
    changes: list[GoalChange] = Field(default_factory=list)


class WorkspaceResult(BaseModel):
    frame_reason: str = ""
    focus_ids: list[int] = Field(default_factory=list)
    control_mode: str = "reactive"
    subjective_experience: str = ""


class ActionIntent(BaseModel):
    action_id: str = ""  # optional durable identity; the parent fills a stable one
    action_type: str = "Reply"
    workflow_name: str = "dialogue_story"
    content: str = ""
    arguments: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    requires_completed: list[str] = Field(default_factory=list)


class ActionSelectionResult(BaseModel):
    actions: list[ActionIntent] = Field(default_factory=list)
    reasoning: str = ""


class MemoryCandidate(BaseModel):
    knoxel_id: int
    text: str = ""
    revision: int = 1
    memory_kind: str = "attributed_record"
    scope: dict = Field(default_factory=dict)
    provenance: dict | None = None
    source_ids: list[int] = Field(default_factory=list)
    timestamp: str | None = None
    historical: bool = False
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    deception_intent_id: int | None = None


class WorkflowOption(BaseModel):
    name: str
    description: str = ""


class CognitiveRequest(BaseModel):
    run_id: str = ""
    correlation_id: str = ""
    companion_name: str = "Spamuel"
    user_name: str = "Jesse"
    state_summary: str = ""
    mental_state_latent_vec: list[float] = Field(default_factory=list)
    mental_state_age_seconds: float = 0.0
    personality_vec: list[float] = Field(default_factory=list)
    active_goals: list[str] = Field(default_factory=list)
    stimuli: list[Stimulus] = Field(default_factory=list)
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    memory_trace: dict = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    available_workflows: list[WorkflowOption] = Field(default_factory=list)


class CognitiveReport(BaseModel):
    """Parent output. MainCogLoop commits the GWT and dispatches actions."""

    status: str = "done"
    summary: str = ""
    receipts: list[StageReceipt] = Field(default_factory=list)
    dispatches: list[ActionDispatch] = Field(default_factory=list)
    gwt_tick_id: int | None = None
    focus_ids: list[int] = Field(default_factory=list)
    winning_attention_ids: list[int] = Field(default_factory=list)
    active_goal_ids: list[int] = Field(default_factory=list)
    control_mode: str = "reactive"
    frame_reason: str = ""
    subjective_experience: str = ""
    appraisal: AppraisalResult | None = None
    goal_changes: list[GoalChange] = Field(default_factory=list)
    event_origins: dict[str, str] = Field(default_factory=dict)
    mental_state_latent_vec: list[float] = Field(default_factory=list)
    affect_trace: dict = Field(default_factory=dict)
    affect_summary: str = ""
    regulation: str = ""
    selected_memories: list[MemoryCandidate] = Field(default_factory=list)
    memory_trace: dict = Field(default_factory=dict)
    belief_updates: list[BeliefUpdate] = Field(default_factory=list)


__all__ = [
    "ActionDispatch",
    "ActionIntent",
    "ActionSelectionResult",
    "AppraisalResult",
    "BeliefUpdate",
    "AttentionNomination",
    "AttentionResult",
    "CognitiveReport",
    "CognitiveRequest",
    "ContextResult",
    "GoalChange",
    "GoalsResult",
    "MemoryCandidate",
    "StageReceipt",
    "Stimulus",
    "TriageItem",
    "TriageResult",
    "WorkflowOption",
    "WorkspaceResult",
]
