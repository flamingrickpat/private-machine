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
    presentation: dict = Field(default_factory=dict)


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


# Where a proposal's content comes from. A thought-origin proposal must stay
# distinguishable from an outside event, so a private worry never becomes a
# remembered fact.
PROPOSAL_ORIGINS = ("outside", "thought", "memory", "state")
# What the proposal content claims about reality. Workspace and dialogue read
# this label; "imagined" content is a possibility, never an executed event.
REALITY_LABELS = ("observed", "remembered", "internal", "inferred", "imagined")
PROPOSAL_KINDS = ("attention", "memory_access", "appraisal", "simulation", "regulation", "action_tendency")


class CodeletProposal(BaseModel):
    """One typed nomination that a codelet offers to attention competition.

    A proposal cannot change accepted state or start an action. The main
    module commits it as an ``AttentionRecord`` whether it wins or not.
    ``source_revisions`` maps each knoxel id that the codelet read to the
    revision it saw. The module rejects the proposal as stale when one of
    those revisions changed before commit.
    """

    nomination_id: str
    codelet: str
    family: str = ""
    proposal_kind: str = "attention"
    origin: str = "outside"
    reality: str = "observed"
    text: str = ""
    target_ids: list[int] = Field(default_factory=list)
    cause: list[str] = Field(default_factory=list)
    activation: float = 0.0
    source_revisions: dict[str, int] = Field(default_factory=dict)
    dropped_target_ids: list[int] = Field(default_factory=list)


class CodeletRun(BaseModel):
    """Evidence for one scheduled codelet decision in one cycle.

    ``status`` is ``ran`` (it produced proposals), ``silent`` (it ran and found
    nothing), ``skipped`` (it was not scheduled), or ``failed``. ``reason``
    names why, for example ``cooldown`` or ``budget``.
    """

    codelet: str
    kind: str
    family: str = ""
    status: str
    reason: str = ""
    activation: float = 0.0
    cause: list[str] = Field(default_factory=list)
    proposal_ids: list[str] = Field(default_factory=list)
    calls: int = 0
    input_tokens_estimate: int = 0
    output_tokens_estimate: int = 0
    seconds: float = 0.0


class CodeletsResult(BaseModel):
    """Report of the codelets stage: proposals plus the full schedule trace.

    ``policy`` is the attention policy of ``loop`` from the same catalogue
    read. The attention stage uses it, so both stages of one cycle apply one
    catalogue version even if the file changes in between.
    """

    proposals: list[CodeletProposal] = Field(default_factory=list)
    runs: list[CodeletRun] = Field(default_factory=list)
    catalogue_version: str = ""
    catalogue_digest: str = ""
    loop: str = "main"
    policy: dict = Field(default_factory=dict)
    signals: dict[str, float] = Field(default_factory=dict)
    cost: dict = Field(default_factory=dict)


class AttentionNomination(BaseModel):
    """One scored nomination and its competition outcome.

    The first block keeps the older attention contract. The A15 fields record
    which codelet produced the nomination and why it won or lost. ``status``
    is ``selected``, ``deferred``, or ``rejected``; the module can later mark
    a nomination ``stale`` before commit.
    """

    target_ids: list[int] = Field(default_factory=list)
    attention_kind: str = "goal_relevance"
    score: float = 0.5
    novelty: float = 0.0
    urgency: float = 0.0
    goal_relevance: float = 0.0
    reportability: float = 0.0
    reasoning: str = ""
    nomination_id: str = ""
    codelet: str = ""
    family: str = ""
    origin: str = "outside"
    reality: str = "observed"
    text: str = ""
    cause: list[str] = Field(default_factory=list)
    activation: float = 0.0
    source_revisions: dict[str, int] = Field(default_factory=dict)
    status: str = ""
    reason: str = ""


class AttentionResult(BaseModel):
    nominations: list[AttentionNomination] = Field(default_factory=list)
    selected_ids: list[int] = Field(default_factory=list)
    reasoning: str = ""
    competition: dict = Field(default_factory=dict)


class NominationScore(BaseModel):
    """Model scores for one nomination, as integers from 0 to 10.

    Small quantized models handle a coarse integer scale more reliably than
    floats. The competition divides each value by 10.
    """

    id: str
    urgency: int = Field(ge=0, le=10)
    novelty: int = Field(ge=0, le=10)
    goal_relevance: int = Field(ge=0, le=10)
    reportability: int = Field(ge=0, le=10)
    why: str = ""


class NominationScores(BaseModel):
    scores: list[NominationScore] = Field(default_factory=list)


class CodeletAnswer(BaseModel):
    """The small answer that one LLM codelet returns."""

    fires: bool
    text: str = ""
    target_ids: list[int] = Field(default_factory=list)


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
    presentation: dict = Field(default_factory=dict)


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
    evidence_role: str = "background"


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
    public_context: dict = Field(default_factory=dict)
    managed_runs: list[dict] = Field(default_factory=list)
    # A15 scheduler inputs. The module derives them from committed state.
    stimulus_revisions: dict[str, int] = Field(default_factory=dict)
    active_goal_records: list[dict] = Field(default_factory=list)
    codelet_history: dict[str, dict] = Field(default_factory=dict)
    seconds_since_user_input: float | None = None
    codelet_catalogue: str = ""
    # Selects the catalogue's attention policy. A17 sets "inner".
    attention_loop: str = "main"


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
    nominations: list[AttentionNomination] = Field(default_factory=list)
    codelet_trace: dict = Field(default_factory=dict)


__all__ = [
    "ActionDispatch",
    "ActionIntent",
    "ActionSelectionResult",
    "AppraisalResult",
    "BeliefUpdate",
    "AttentionNomination",
    "AttentionResult",
    "CognitiveReport",
    "CodeletAnswer",
    "CodeletProposal",
    "CodeletRun",
    "CodeletsResult",
    "CognitiveRequest",
    "ContextResult",
    "GoalChange",
    "GoalsResult",
    "MemoryCandidate",
    "NominationScore",
    "NominationScores",
    "PROPOSAL_KINDS",
    "PROPOSAL_ORIGINS",
    "REALITY_LABELS",
    "StageReceipt",
    "Stimulus",
    "TriageItem",
    "TriageResult",
    "WorkflowOption",
    "WorkspaceResult",
]
