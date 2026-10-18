"""Typed contracts for the main cognitive cycle.

Model output and phase results cross here. Each model answer is validated
into one of these schemas once. The commit, not the model, builds records.
No phase stores free-form unvalidated model text as canonical memory.

The request and the shared phase state are in
:mod:`workflows.main_cognition.cycle_work`.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from pm.records.mental_state import AppraisalGeneral, AppraisalSocial, FullMentalState, StateEmotions, StateNeeds
from workflows.cognitive_control.schemas import FocusProposal, FocusState, WaitRequest
from workflows.public_speech.public_context import Presentation


class Stimulus(BaseModel):
    """One incoming item for one cycle, resolved from a bus envelope."""

    knoxel_id: int = 0
    source: str = ""
    text: str = ""
    correlation_id: str = ""
    kind: str = "user.input"


class PhaseReceipt(BaseModel):
    """Evidence that one phase of the cycle ran, with its short summary."""

    name: str = ""
    summary: str = ""


class ActionArguments(BaseModel):
    """Workflow arguments of one action.

    The target workflow owns the meaning of its arguments, so unknown keys
    stay (``extra="allow"``). The cycle reads only the named fields.
    Serialization writes only the supplied fields. So ``payload()`` returns
    the arguments as the model supplied them, also after a JSON round trip
    between phases. ``timeout_seconds`` is strict: a bool is not a duration.
    """

    model_config = ConfigDict(extra="allow")

    operation: str = ""
    run_id: str = ""
    workflow_name: str = ""
    timeout_seconds: float | None = Field(default=None, strict=True)

    @model_serializer(mode="wrap")
    def _supplied_only(self, handler):
        data = handler(self)
        return {key: value for key, value in data.items()
                if key in self.model_fields_set or key not in type(self).model_fields}

    def payload(self) -> dict:
        """Return the supplied arguments for the target workflow."""
        return self.model_dump(mode="json")


class ActionDispatch(BaseModel):
    """Typed application dispatch record (plan: workflows.md, action dispatch).

    ``content`` states the intent of the action, not a finished companion
    reply. ``depends_on`` names action IDs that must be released earlier in the
    same selection; it only orders dispatch. ``requires_completed`` names
    action IDs whose successful completion receipt is required before this
    action can start. A queued prerequisite never releases dependent work.
    """

    action_id: str = ""
    workflow_name: str = ""
    action_type: str = "Reply"
    content: str = ""
    arguments: ActionArguments = Field(default_factory=ActionArguments)
    depends_on: list[str] = Field(default_factory=list)
    requires_completed: list[str] = Field(default_factory=list)
    gwt_tick_id: int | None = None
    status: str = "proposed"  # proposed | dispatched | queued | rejected
    rejection_reason: str = ""
    presentation: Presentation = Field(default_factory=Presentation)


class TriageItem(BaseModel):
    knoxel_id: int = 0
    category: str = ""
    input_kind: str = "message"
    urgency: float = 0.0
    relevance: float = 0.0
    reasoning: str = ""
    evidence_ids: list[int] = Field(default_factory=list)


class TriageResult(BaseModel):
    items: list[TriageItem] = Field(default_factory=list)


class AppraisalResult(BaseModel):
    """The model answer of the appraise phase.

    ``appraisal`` holds the appraisal axes by their ``AppraisalGeneral`` and
    ``AppraisalSocial`` field names. ``emotion_delta`` and ``need_delta`` use
    the ``StateEmotions`` and ``StateNeeds`` field names. An unknown name or
    a value outside the axis bounds is invalid model output. The repair loop
    asks the model to fix it.
    """

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

    @model_validator(mode="after")
    def _known_names(self):
        # The repair turn shows this message to the model. Name the allowed
        # keys, so the model can replace an invented name with a real one.
        groups = (
            ("appraisal", self.appraisal, [*AppraisalGeneral.model_fields, *AppraisalSocial.model_fields]),
            ("emotion_delta", self.emotion_delta, list(StateEmotions.model_fields)),
            ("need_delta", self.need_delta, list(StateNeeds.model_fields)),
        )
        problems = [f"{field} has unknown keys {', '.join(sorted(set(values) - set(allowed)))}; "
                    f"allowed keys are {', '.join(allowed)}"
                    for field, values, allowed in groups if set(values) - set(allowed)]
        if problems:
            raise ValueError("Unknown names. " + ". ".join(problems) + ".")
        # Raises on a value outside the declared axis bounds.
        self.general()
        self.social()
        return self

    def general(self) -> AppraisalGeneral:
        """Return the general appraisal axes. An omitted axis has its default."""
        return AppraisalGeneral.model_validate(self.appraisal)

    def social(self) -> AppraisalSocial:
        """Return the social appraisal axes. An omitted axis has its default."""
        return AppraisalSocial.model_validate(self.appraisal)


class AppraisalAffect(BaseModel):
    """The affect that the appraise phase derived from the appraisal.

    ``appraisal_state`` is the state that ``compute_state_delta`` derived
    from the appraisal axes and the explicit deltas. ``delta_vec`` is that
    state minus the start state. It is a plain vector, because a delta can
    break the value bounds of ``FullMentalState``. Both go onto the appraisal
    ``Feature``. ``entity_id`` is the relationship partner of the appraisal,
    or None for a self or workflow group. ``regulation`` is ``reactive``,
    ``reflective``, or ``blended``. ``felt`` is the accepted cycle state in
    words, for prompts.
    """

    appraisal_state: FullMentalState = Field(default_factory=FullMentalState)
    delta_vec: list[float] = Field(default_factory=list)
    entity_id: int | None = None
    regulation: str = ""
    felt: str = ""


class AppraisalView(AppraisalResult):
    """The appraisal with the accepted feeling in words, as later prompts render it."""

    felt_state: str = ""


class BeliefUpdate(BaseModel):
    """One reported user claim, copied from original evidence.

    A correction replaces beliefs with the same scoped key. The runtime keeps
    the prior beliefs and rejects quotes from generated or imagined sources.
    """

    source_id: int = 0
    quote: str = Field(min_length=1)
    belief_key: str = Field(min_length=1)
    correction: bool = False


class ContextResult(BaseModel):
    memories: list[str] = Field(default_factory=list)
    active_goals: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    candidate_ids: list[int] = Field(default_factory=list)
    belief_updates: list[BeliefUpdate] = Field(default_factory=list)


# Where a proposal's content comes from. A thought-origin proposal must stay
# distinguishable from an outside event, so a private worry never becomes a
# remembered fact.
PROPOSAL_ORIGINS = ("outside", "thought", "memory", "state")
# What the proposal content claims about reality. Workspace and dialogue read
# this label; "imagined" content is a possibility, never an executed event.
REALITY_LABELS = ("observed", "remembered", "internal", "inferred", "imagined")
PROPOSAL_KINDS = ("attention", "memory_access", "appraisal", "simulation", "regulation", "action_tendency", "belief")

# A16 proposal-consumer map. A typed nomination without a consumer does not
# establish a cognitive effect, so every kind names its downstream owner and
# the durable effect that owner produces. The host commits one
# AttentionRecord per proposal whatever happens; the consumer column is what
# turns an *accepted* (selected, not stale) proposal into behavior.
PROPOSAL_CONSUMERS = {
    "attention": "Workspace frame; the dialogue workflow then plans its memory context from the winning texts and target ids.",
    "memory_access": "Context planning for public speech: winning target ids become required records for the reply.",
    "appraisal": "Workspace frame and the goals phase: the interpretation informs the frame and can justify a goal change through the goals commit boundary.",
    "simulation": "Workspace frame and the decide phase: a possibility can steer a choice while its imagined reality label keeps it distinguishable from events.",
    "regulation": "Workspace frame and the decide phase: a way to stay steady can change the chosen action without changing the appraisal itself.",
    "action_tendency": "Goals phase and the decide phase: a motive can become a durable Intention change or a concrete choice, at the owner's discretion.",
    "belief": "The belief owner in the main module stages a revisable subjective BeliefRecord; an interpretation never rewrites the observed records it cites.",
}


class CodeletProposal(BaseModel):
    """One typed nomination that a codelet offers to attention competition.

    A proposal cannot change accepted state or start an action. The main
    module commits it as an ``AttentionRecord`` whether it wins or not.
    ``source_revisions`` maps each knoxel id that the codelet read to the
    revision it saw. The module rejects the proposal as stale when one of
    those revisions changed before commit.
    """

    nomination_id: str = ""
    codelet: str = ""
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
    topic: str = Field(default="", description="Stable topic key of a belief proposal. Other kinds leave it empty.")


class CodeletRun(BaseModel):
    """Evidence for one scheduled codelet decision in one cycle.

    ``status`` is ``ran`` (it produced proposals), ``silent`` (it ran and found
    nothing), or ``skipped`` (it was not scheduled). ``reason`` names why, for
    example ``cooldown`` or ``budget``.
    """

    codelet: str = ""
    kind: str = ""
    family: str = ""
    status: str = ""
    reason: str = ""
    activation: float = 0.0
    cause: list[str] = Field(default_factory=list)
    proposal_ids: list[str] = Field(default_factory=list)
    calls: int = 0
    input_tokens_estimate: int = 0
    output_tokens_estimate: int = 0
    seconds: float = 0.0


class CodeletCost(BaseModel):
    """Estimated model cost of the codelets of one cycle."""

    llm_calls: int = 0
    input_tokens_estimate: int = 0
    output_tokens_estimate: int = 0
    seconds: float = 0.0
    measurement: str = "estimated.tiktoken"


class AttentionPolicy(BaseModel):
    """Per-loop attention settings from the codelet catalogue.

    The main loop uses ``main``. A17 adds an ``inner`` policy for the second
    conscious loop without a code change. The codelets phase records the
    policy in its result. The attention phase uses that recorded copy, so both
    phases of one cycle apply one catalogue read.
    """

    model_config = ConfigDict(extra="forbid")

    activation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_llm_codelets: int = Field(default=3, ge=0)
    danger_llm_codelets: int = Field(default=0, ge=0)
    capacity: int = Field(default=4, ge=1)
    min_score: float = Field(default=0.3, ge=0.0, le=1.0)
    score_batch_size: int = Field(default=8, ge=1)
    weights: dict[str, float] = Field(default_factory=lambda: {
        "urgency": 0.35, "novelty": 0.15, "goal_relevance": 0.3, "reportability": 0.2})
    danger_urgency_bonus: float = Field(default=0.3, ge=0.0)
    reserved_codelets: dict[str, int] = Field(default_factory=dict)


class CodeletsResult(BaseModel):
    """Result of the codelets phase: proposals plus the full schedule trace."""

    proposals: list[CodeletProposal] = Field(default_factory=list)
    runs: list[CodeletRun] = Field(default_factory=list)
    catalogue_version: str = ""
    catalogue_digest: str = ""
    loop: str = "main"
    policy: AttentionPolicy = Field(default_factory=AttentionPolicy)
    signals: dict[str, float] = Field(default_factory=dict)
    # Pathway activation values without the ``pathway.`` prefix, so a trace
    # shows which accepted precursors drove this cycle.
    pathway_signals: dict[str, float] = Field(default_factory=dict)
    cost: CodeletCost = Field(default_factory=CodeletCost)


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
    topic: str = ""
    status: str = ""
    reason: str = ""


class RankingEntry(BaseModel):
    """One nomination in the competition order, with its outcome."""

    nomination_id: str = ""
    codelet: str = ""
    score: float = 0.0
    status: str = ""
    reason: str = ""


class CompetitionTrace(BaseModel):
    """How attention competition chose its winners.

    ``ranking`` lists the scored nominations in competition order.
    ``rejected`` lists the unscored ones. ``focus`` is the effective
    deliberate focus of this cycle, or None.
    """

    policy: AttentionPolicy = Field(default_factory=AttentionPolicy)
    danger: bool = False
    deliberate_bias: dict[str, float] = Field(default_factory=dict)
    ranking: list[RankingEntry] = Field(default_factory=list)
    rejected: list[RankingEntry] = Field(default_factory=list)
    focus: FocusState | None = None
    scoring_calls: int = 0


class AttentionResult(BaseModel):
    nominations: list[AttentionNomination] = Field(default_factory=list)
    selected_ids: list[int] = Field(default_factory=list)
    reasoning: str = ""
    competition: CompetitionTrace = Field(default_factory=CompetitionTrace)


class NominationScore(BaseModel):
    """Model scores for one nomination, as integers from 0 to 10.

    Small quantized models handle a coarse integer scale more reliably than
    floats. The competition divides each value by 10.
    """

    id: str = ""
    urgency: int = Field(ge=0, le=10)
    novelty: int = Field(ge=0, le=10)
    goal_relevance: int = Field(ge=0, le=10)
    reportability: int = Field(ge=0, le=10)
    why: str = ""


class NominationScores(BaseModel):
    scores: list[NominationScore] = Field(default_factory=list)


class CodeletAnswer(BaseModel):
    """The small answer that one LLM codelet returns.

    ``topic`` is the stable belief label that a ``belief``-kind codelet
    supplies. Every other codelet leaves it empty.
    """

    fires: bool
    text: str = ""
    topic: str = ""
    target_ids: list[int] = Field(default_factory=list)


class GoalChange(BaseModel):
    change: str = ""  # retain | add | complete | reprioritize | defer | refuse
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
    action_id: str = ""  # optional durable identity; the route phase fills a stable one
    action_type: str = "Reply"
    workflow_name: str = "dialogue_story"
    content: str = ""
    arguments: ActionArguments = Field(default_factory=ActionArguments)
    depends_on: list[str] = Field(default_factory=list)
    requires_completed: list[str] = Field(default_factory=list)
    presentation: Presentation = Field(default_factory=Presentation)

    @model_validator(mode="after")
    def validate_wait(self) -> "ActionIntent":
        """Reject unusable Sleep durations inside the phase's repair boundary."""
        if self.action_type == "Sleep" and "timeout_seconds" in self.arguments.model_fields_set:
            WaitRequest(seconds=self.arguments.timeout_seconds)
        return self


class ActionSelectionResult(BaseModel):
    actions: list[ActionIntent] = Field(default_factory=list)
    reasoning: str = ""


class MemoryCandidate(BaseModel):
    """One retrieved memory that the cycle may select.

    ``scope`` and ``provenance`` are record fields as the retrieval service
    serialized them. The cycle passes them on unchanged.
    """

    knoxel_id: int = 0
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
    name: str = ""
    description: str = ""


class ManagedRunView(BaseModel):
    """One recent managed run, as the cycle prompts show it."""

    run_id: str | None = None
    workflow_name: str | None = None
    state: str | None = None
    instructions: str | None = None
    parent_correlation: str | None = None


class GoalRecordView(BaseModel):
    """One active goal record with the revision that the cycle read."""

    knoxel_id: int = 0
    text: str = ""
    revision: int = 1


class CodeletHistory(BaseModel):
    """Run history of one codelet, derived from committed attention frames.

    ``cycles_since`` counts committed cycles after the last run.
    ``seconds_since`` is the wall time since that frame was created.
    """

    runs: int = 0
    last_tick: int = 0
    cycles_since: int | None = None
    seconds_since: float | None = None


class PathwayPrecursor(BaseModel):
    """One accepted codelet proposal from a recent cycle that feeds a pathway.

    The host derives it fresh from a committed ``AttentionRecord`` in every
    cycle. ``revision`` is the record revision at read time.
    """

    codelet: str = ""
    record_id: int = 0
    tick_id: int = 0
    attention_kind: str = ""
    reality: str = ""
    family: str = ""
    pathways: list[str] = Field(default_factory=list)
    selected: bool = False
    revision: int = 1


class MentalProposal(BaseModel):
    """A private plan from deliberate mental work that a stimulus carries.

    ``record_id`` is the stimulus record that holds the plan. The goals phase
    may accept, change, or decline the proposed goal changes.
    """

    record_id: int = 0
    instruction: str = ""
    summary: str = ""
    disposition: str = "propose"
    evidence_ids: list[int] = Field(default_factory=list)
    goal_changes: list[GoalChange] = Field(default_factory=list)


class FocusProposalRef(BaseModel):
    """A deliberate focus proposal and the stimulus record that carries it."""

    record_id: int = 0
    focus: FocusProposal = Field(default_factory=lambda: FocusProposal(instruction=""))


class ReflectionGuidance(BaseModel):
    """One scoped private experiment that is in effect for this cycle."""

    experiment_id: int = 0
    owner: str = ""
    effective_record_id: int = 0
    scope: str = ""
    instruction: str = ""
    status: str = ""


__all__ = [
    "ActionArguments",
    "ActionDispatch",
    "ActionIntent",
    "ActionSelectionResult",
    "AppraisalAffect",
    "AppraisalResult",
    "AppraisalView",
    "AttentionNomination",
    "AttentionPolicy",
    "AttentionResult",
    "BeliefUpdate",
    "CodeletAnswer",
    "CodeletCost",
    "CodeletHistory",
    "CodeletProposal",
    "CodeletRun",
    "CodeletsResult",
    "CompetitionTrace",
    "ContextResult",
    "FocusProposalRef",
    "GoalChange",
    "GoalRecordView",
    "GoalsResult",
    "ManagedRunView",
    "MemoryCandidate",
    "MentalProposal",
    "NominationScore",
    "NominationScores",
    "PROPOSAL_CONSUMERS",
    "PROPOSAL_KINDS",
    "PROPOSAL_ORIGINS",
    "PathwayPrecursor",
    "PhaseReceipt",
    "REALITY_LABELS",
    "RankingEntry",
    "ReflectionGuidance",
    "Stimulus",
    "TriageItem",
    "TriageResult",
    "WorkflowOption",
    "WorkspaceResult",
]
