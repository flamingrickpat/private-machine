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
    reply. ``depends_on`` names action IDs whose successful receipts are
    required before dispatch.
    """

    action_id: str
    workflow_name: str
    action_type: str = "Reply"
    content: str = ""
    arguments: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    gwt_tick_id: int | None = None
    status: str = "proposed"  # proposed | dispatched | rejected
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
    reasoning: str = ""
    evidence_ids: list[int] = Field(default_factory=list)


class ContextResult(BaseModel):
    memories: list[str] = Field(default_factory=list)
    active_goals: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    candidate_ids: list[int] = Field(default_factory=list)


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
    action_type: str = "Reply"
    workflow_name: str = "dialogue_story"
    content: str = ""
    arguments: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)


class ActionSelectionResult(BaseModel):
    actions: list[ActionIntent] = Field(default_factory=list)
    reasoning: str = ""


class MemoryCandidate(BaseModel):
    knoxel_id: int
    text: str = ""


class WorkflowOption(BaseModel):
    name: str
    description: str = ""


class CognitiveRequest(BaseModel):
    run_id: str = ""
    correlation_id: str = ""
    companion_name: str = "Spamuel"
    user_name: str = "Jesse"
    state_summary: str = ""
    active_goals: list[str] = Field(default_factory=list)
    stimuli: list[Stimulus] = Field(default_factory=list)
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
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


__all__ = [
    "ActionDispatch",
    "ActionIntent",
    "ActionSelectionResult",
    "AppraisalResult",
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
