"""Typed contracts for the main cognitive cycle.

Model output and phase results cross here. Each model answer is validated
into one of these schemas once. The commit, not the model, builds records.
No phase stores free-form unvalidated model text as canonical memory.

The request and the shared phase state are in
:mod:`workflows.main_cognition.cycle_work`.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_serializer, model_validator

from pm.constants import CODELET_BUDGET_DEFAULT, CODELET_BUDGET_MAX, DECISION_VARIANTS, EXPECTATION_COUNT
from pm.records.mental_state import AppraisalGeneral, AppraisalSocial, FullMentalState, StateEmotions, StateNeeds
from pm.records.codelet_records import StructureRelation
from pm.records.global_workspace import ShardChange
from pm.context_render.story import RecordView
from pm.records.workspace import ShardKey, WorkspaceShard
from workflows.cognitive_control.schemas import FocusProposal, FocusState, WaitRequest
from workflows.public_speech.public_context import Presentation


class Stimulus(BaseModel):
    """One incoming item for one cycle, resolved from a bus envelope.

    ``entity_id`` is the entity that spoke a dialogue stimulus, or None. The
    host sets it from the dialogue record when the speaker is a known entity
    other than the companion. The intake phase puts such a stimulus into the
    group of that entity.
    """

    knoxel_id: int = 0
    source: str = ""
    text: str = ""
    correlation_id: str = ""
    kind: str = "user.input"
    entity_id: int | None = None


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
    instructions: str = ""
    instruction: str = ""
    minecraft_read: str = ""
    timeout_seconds: float | None = Field(default=None, strict=True)

    @model_serializer(mode="wrap")
    def _supplied_only(self, handler):
        data = handler(self)
        return {key: value for key, value in data.items()
                if key in self.model_fields_set or key not in type(self).model_fields}

    def payload(self) -> dict:
        """Return the supplied arguments for the target workflow."""
        return self.model_dump(mode="json")


class ExpectationDraft(BaseModel):
    """One expected reaction to a routed action, before the commit stores it.

    Port of the expectation schema of ``_generate_expectations_for_action``.
    ``expected_valence`` is the feeling, -1..1, that the companion gets if
    the expectation comes true. ``urgency`` is how soon it is due.
    ``about_entity_ids`` are the entities whose reaction is expected.
    ``supersedes_ids`` are open expectations that this one replaces, because
    their embeddings are nearly equal (``EXPECTATION_SAME_SIMILARITY``).
    """

    content: str = ""
    expected_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    urgency: float = Field(default=0.3, ge=0.0, le=1.0)
    about_entity_ids: list[int] = Field(default_factory=list)
    supersedes_ids: list[int] = Field(default_factory=list)


class ActionDispatch(BaseModel):
    """Typed application dispatch record (plan: workflows.md, action dispatch).

    ``content`` is the in-character decision text of the stimulus group,
    exactly as the decide phase wrote it. The route phase never rewrites it.
    The target workflow reads it as the intent of the action: the public
    workflows read it from the stored action record. A tool workflow reads
    its task from ``arguments.instructions`` when the route supplies it. A
    workflow in ``route.DECISION_GOAL_WORKFLOWS`` (``tool_call``) never gets
    routed instructions: its goal is this decision text, unchanged.

    ``group``, ``stimulus_ids``, and ``correlation_id`` come from the
    stimulus group of the decision. The host publishes the action with that
    correlation, so each group has its own exchange. ``expectations`` are
    the expected reactions that the commit stores with the action.

    ``depends_on`` names action IDs that must be released earlier in the
    same cycle; it only orders dispatch. ``requires_completed`` names
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
    group: ShardKey = Field(default_factory=ShardKey)
    stimulus_ids: list[int] = Field(default_factory=list)
    correlation_id: str = ""
    expectations: list[ExpectationDraft] = Field(default_factory=list)


class RequestShard(BaseModel):
    """One workspace shard that an action request carries, with its story name.

    ``name`` is the label that the story prefix of the cycle used: ``self``,
    the entity name, or the workflow name. The shard holds record ids only.
    The target workflow reads the records from the memory store.
    """

    name: str = ""
    shard: WorkspaceShard = Field(default_factory=WorkspaceShard)


class ActionRequest(BaseModel):
    """The typed payload of one ``action.request`` envelope.

    Main cognition publishes one request for each released dispatch. The
    host stores the same request in the action record
    (``ActionMetadata.dispatch_payload``), so recovery can publish it again
    without a change.

    - ``knoxel_id`` is the id of the stored ``ActionSchemaRecord``.
    - ``content`` is the in-character decision text, unchanged.
    - ``group`` is the stimulus group of the decision. ``stimuli`` are the
      stimuli of that group.
    - ``partner_entity_id`` and ``partner_name`` name the entity that a reply
      addresses: the entity of an entity group, else the user. A second
      companion is an entity like the user.
    - ``shards`` are the self shard, the partner's entity shard, and the
      group's own shard when it is a workflow shard, each with its tail.
      The public workflows build their context from these shards. They do
      not retrieve memory again.
    - ``workspace`` is the subjective experience of the cycle. ``affect`` and
      ``regulation`` are the feeling in words and the regulation mode.

    Recovery and task replies build requests without a cycle. They set
    ``task_receipt`` (the managed-run receipt that the reply reports) and
    ``recovery_outcome``. The receipt is an opaque record of the run
    supervisor, so it stays a mapping.
    """

    model_config = ConfigDict(extra="forbid")

    knoxel_id: int | None = None
    gwt_tick: int | None = None
    action_id: str = ""
    workflow_name: str = ""
    content: str = ""
    arguments: ActionArguments = Field(default_factory=ActionArguments)
    group: ShardKey = Field(default_factory=ShardKey)
    partner_entity_id: int | None = None
    partner_name: str = ""
    stimuli: list[Stimulus] = Field(default_factory=list)
    shards: list[RequestShard] = Field(default_factory=list)
    presentation: Presentation = Field(default_factory=Presentation)
    workspace: str = ""
    affect: str = ""
    regulation: str = ""
    task_receipt: dict = Field(default_factory=dict)
    recovery_outcome: str = ""

    def payload(self) -> dict:
        """Return the JSON payload of the envelope."""
        return self.model_dump(mode="json")


class ActionMetadata(BaseModel):
    """The metadata fields of an ``ActionSchemaRecord`` that the runtime reads.

    Record metadata is an open mapping. Several modules add their own keys,
    for example the public reply id. This view reads the keys that main
    cognition, dispatch recovery, and public speech use and ignores the
    others. ``dispatch_status`` is ``pending`` until a workflow admits the
    action. The stored request itself is read by :class:`PendingDispatch`,
    only for a pending action, because actions of older checkpoints stored
    another payload format.
    """

    model_config = ConfigDict(extra="ignore")

    action_id: str = ""
    workflow_name: str = ""
    dispatch_status: str = ""
    dispatch_target: str = ""
    dispatch_correlation: str = ""
    presentation: Presentation = Field(default_factory=Presentation)
    affect_summary: str = ""
    regulation: str = ""
    public_reply_id: int | None = None


class PendingDispatch(ActionMetadata):
    """The metadata of a pending action, with its stored :class:`ActionRequest`."""

    dispatch_payload: ActionRequest = Field(default_factory=ActionRequest)


class RecordMetadataView(BaseModel):
    """The general metadata keys of a memory record that main cognition reads.

    Record metadata is an open mapping, so the other keys are ignored. An
    empty ``memory_kind`` means that the record names no kind.
    ``exchange_status`` is ``pending`` or ``resolved`` on a user message that
    waits for or got a public reply, and empty on other records.
    """

    model_config = ConfigDict(extra="ignore")

    memory_kind: str = ""
    exchange_status: str = ""


class TriageItem(BaseModel):
    knoxel_id: int = 0
    category: str = ""
    input_kind: str = "message"
    urgency: float = 0.0
    relevance: float = 0.0
    reasoning: str = ""
    evidence_ids: list[int] = Field(default_factory=list)


#: Triage categories from the least to the most important.
TRIAGE_CATEGORIES = ("Insignificant", "Moderate", "Significant", "Critical")


class TriageResult(BaseModel):
    """The triage answer. ``codelet_budget`` is the number of codelet runs of this cycle.

    A category outside :data:`TRIAGE_CATEGORIES` is invalid model output and
    gets a repair turn. A silent fallback could mark a group insignificant
    and so drop its decision.
    """

    items: list[TriageItem] = Field(default_factory=list)
    codelet_budget: int = Field(default=CODELET_BUDGET_DEFAULT, ge=1, le=CODELET_BUDGET_MAX)

    @model_validator(mode="after")
    def _known_categories(self):
        unknown = sorted({item.category for item in self.items} - set(TRIAGE_CATEGORIES))
        if unknown:
            raise ValueError(f"Unknown categories {', '.join(map(repr, unknown))}. "
                             f"Use one of {', '.join(TRIAGE_CATEGORIES)}.")
        return self


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


class VariantPercept(BaseModel):
    """One typed percept of a codelet variant, with its mini-appraisal.

    The conversion call fills the type, the story fields, and the header
    (salience, valence, appraisal axes). The codelets phase then copies the
    current mental state, sets the appraisal axes, and calls
    ``compute_state_delta``. ``appraisal_vec`` is the resulting state and
    ``delta_vec`` is the change from the copied state. A percept is not
    causal: neither vector changes the mental state until attention puts the
    percept into a workspace shard. Then the commit stages it as causal.

    ``record_id`` is a pending id below -1 until the commit stages the
    ``CodeletPercept`` record. ``activation`` is its activation when it
    entered the CSM.
    """

    record_id: int = 0
    percept_type: str = ""
    text: str = ""
    details: dict[str, str | list[str]] = Field(default_factory=dict)
    salience: float = 0.0
    valence: float = 0.0
    appraisal_general: AppraisalGeneral = Field(default_factory=AppraisalGeneral)
    appraisal_social: AppraisalSocial = Field(default_factory=AppraisalSocial)
    appraisal_vec: list[float] = Field(default_factory=list)
    delta_vec: list[float] = Field(default_factory=list)
    activation: float = 0.0


class CodeletVariant(BaseModel):
    """One output variant of one codelet run.

    ``recommendation`` is what the companion should feel, think, or do in the
    story, from the narrow concern of the codelet. ``recommendation_id`` is
    the pending id of the record that holds it in the CSM. Its vectors are
    the mean appraisal state and the summed delta of the percepts, as in the
    old ``CodeletOutput`` feature. ``about_ids`` are the CSM records that the
    variant is about. Each of them gets an endorsement from the codelet.
    """

    recommendation: str = ""
    recommendation_id: int = 0
    recommendation_vec: list[float] = Field(default_factory=list)
    recommendation_delta_vec: list[float] = Field(default_factory=list)
    salience: float = 0.0
    valence: float = 0.0
    percepts: list[VariantPercept] = Field(default_factory=list)
    target_shard: ShardKey = Field(default_factory=ShardKey)
    #: The target word the model wrote, when it named a shard that this cycle
    #: does not have. Empty when the target needed no correction.
    target_note: str = ""
    about_ids: list[int] = Field(default_factory=list)


class CodeletOutput(BaseModel):
    """The output of one codelet run: ``CODELET_OUTPUT_VARIANTS`` variants."""

    codelet: str = ""
    mode: str = ""
    wave: int = 0
    variants: list[CodeletVariant] = Field(default_factory=list)


class ActivationScore(BaseModel):
    """The activation score of one codelet and its parts, for the trace.

    See ``workflows.main_cognition.codelet_registry.activation_score``.
    """

    codelet: str = ""
    score: float = 0.0
    stored: float = 0.0
    stimulus: float = 0.0
    state: float = 0.0
    vector: float = 0.0
    base: float = 0.0
    recency: float = 0.0
    noise: float = 0.0
    cooling: bool = False


class PathwayBoost(BaseModel):
    """One pathway boost: ``source`` ran, and ``target`` got ``strength``.

    ``skipped`` names the unregistered codelets between them. Each skipped
    codelet multiplied the strength by ``PATHWAY_SKIP_DECAY``.
    ``precursor_ids`` are the percepts of ``source`` that the CSM view of
    ``target`` favors.
    """

    pathway: str = ""
    source: str = ""
    target: str = ""
    strength: float = 0.0
    skipped: list[str] = Field(default_factory=list)
    precursor_ids: list[int] = Field(default_factory=list)


class CodeletRun(BaseModel):
    """Evidence for one codelet run in one cycle.

    ``score`` is the activation score that chose it in its wave. ``view_ids``
    are the CSM records of its view, in story order. ``calls`` counts model
    calls; an agent session counts as one call.
    """

    codelet: str = ""
    mode: str = ""
    wave: int = 0
    status: str = "ran"
    reason: str = ""
    score: float = 0.0
    view_ids: list[int] = Field(default_factory=list)
    precursor_ids: list[int] = Field(default_factory=list)
    calls: int = 0
    seconds: float = 0.0


class CodeletCost(BaseModel):
    """Estimated model cost of the codelets of one cycle."""

    llm_calls: int = 0
    input_tokens_estimate: int = 0
    output_tokens_estimate: int = 0
    seconds: float = 0.0
    measurement: str = "estimated.tiktoken"


class CodeletsResult(BaseModel):
    """Result of the codelets phase: outputs and the full trace.

    ``budget`` is the number of codelet runs that triage granted.
    ``scores`` are the activation scores before the selector. ``ratings``
    are the selector ratings, by codelet name, and ``patterns`` is the
    selector's note on the patterns of the recent story. ``echo_ids`` are
    the earlier lines that echo a current stimulus (see
    ``phases.codelets.echoes``). ``outputs`` hold the variants in run order.
    """

    definitions_version: str = ""
    definitions_digest: str = ""
    budget: int = 0
    scores: list[ActivationScore] = Field(default_factory=list)
    ratings: dict[str, float] = Field(default_factory=dict)
    patterns: str = ""
    echo_ids: list[int] = Field(default_factory=list)
    runs: list[CodeletRun] = Field(default_factory=list)
    outputs: list[CodeletOutput] = Field(default_factory=list)
    pathway_boosts: list[PathwayBoost] = Field(default_factory=list)
    cost: CodeletCost = Field(default_factory=CodeletCost)


class CodeletRatings(BaseModel):
    """The answer of the codelet selector: a rating from 0 to 1 for each candidate.

    ``patterns`` comes first in the answer. The selector names the patterns
    of the recent story there (repeated moves with their count, dodges,
    slights, sudden turns) before it rates. A trigger that spans several
    turns, for example a request that comes back a fourth time, is easy to
    miss when the model rates the newest line alone.
    """

    patterns: str = ""
    ratings: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _bounded(self):
        outside = sorted(name for name, value in self.ratings.items() if not 0.0 <= value <= 1.0)
        if outside:
            raise ValueError(f"Ratings must be between 0 and 1. Fix: {', '.join(outside)}.")
        return self


class AgentCodeletJob(BaseModel):
    """The task of one agent codelet session.

    ``story`` is the shared story prefix as text. ``instruction`` is the
    codelet instruction, and ``retrieval_plan`` is its ``data_aquisation``
    list. ``view`` is the codelet's CSM view as story lines with record ids,
    and ``view_ids`` are those ids. ``known_ids`` are the other records that
    the cycle already holds, for example the stimuli and the memory
    candidates. A session may cite them without reading them again.
    """

    codelet: str = ""
    story: str = ""
    instruction: str = ""
    retrieval_plan: list[str] = Field(default_factory=list)
    view: str = ""
    view_ids: list[int] = Field(default_factory=list)
    known_ids: list[int] = Field(default_factory=list)


class AgentCodeletAnswer(BaseModel):
    """The validated result of one agent codelet session.

    ``evidence_ids`` are records that the session cited. Each one is in the
    CSM view, is a known record of the cycle, or was read through the memory
    tools. ``records`` holds a view of
    each cited memory record, so the CSM can hold it.
    """

    recommendation: str = Field(default="", min_length=1)
    evidence_ids: list[int] = Field(default_factory=list)
    records: list[RecordView] = Field(default_factory=list)
    session_ref: str = ""


class ShardStream(BaseModel):
    """The newest stream records that the host offers for one shard tail.

    ``record_ids`` are oldest first. The intake phase keeps the newest of
    them within ``WORKSPACE_TAIL_TOKENS``. The host supplies a view of each
    record in ``CycleRequest.records``.
    """

    key: ShardKey = Field(default_factory=ShardKey)
    record_ids: list[int] = Field(default_factory=list)


class StructureDraft(BaseModel):
    """One structure of this cycle before the commit stages its record.

    ``record_id`` is a pending id below -1 until the commit stages the
    ``Structure`` record. ``builder`` is ``topical_coalition`` or
    ``structure_binder``. ``kind`` is ``balanced`` or ``hard`` for a
    coalition and ``binding`` for a binder structure. ``record_ids`` are CSM
    record ids in world-time order. ``reason`` says why the records belong
    together. ``relations`` are the stated links of a binding.
    ``target_shard`` is the shard that the builder chose. Attention can
    place the records in another shard.
    """

    record_id: int = 0
    builder: str = ""
    kind: str = ""
    theme: str = ""
    reason: str = ""
    record_ids: list[int] = Field(default_factory=list)
    relations: list[StructureRelation] = Field(default_factory=list)
    target_shard: ShardKey = Field(default_factory=ShardKey)
    valence: float = 0.0
    salience: float = 0.0


class BoundRelation(BaseModel):
    """One relation of a binder answer: the source record, the link in words, the target record."""

    model_config = ConfigDict(extra="forbid")

    source_id: int = 0
    relation: str = ""
    target_id: int = 0


class BoundStructure(BaseModel):
    """One structure in the answer of the structure binder.

    ``record_ids`` are ids of the MIND list. ``reason`` states why the
    records belong together, for example a matching topic and a similar
    feeling at that time. ``target`` is one shard label of the story.
    """

    model_config = ConfigDict(extra="forbid")

    theme: str = Field(default="", min_length=1)
    reason: str = Field(default="", min_length=1)
    record_ids: list[int] = Field(default_factory=list, min_length=2)
    relations: list[BoundRelation] = Field(default_factory=list)
    target: str = "self"


class BinderAnswer(BaseModel):
    """The answer of the structure binder call."""

    model_config = ConfigDict(extra="forbid")

    structures: list[BoundStructure] = Field(default_factory=list)


class StructuresResult(BaseModel):
    """Result of the structures phase.

    ``structures`` holds the coalitions first, then the binder structures.
    ``view_ids`` is the CSM view that the binder read. ``clusters`` is the
    number of topical clusters.
    """

    structures: list[StructureDraft] = Field(default_factory=list)
    view_ids: list[int] = Field(default_factory=list)
    clusters: int = 0
    calls: int = 0


class StructureRating(BaseModel):
    """The rating of one structure for one shard in the attention answer."""

    model_config = ConfigDict(extra="forbid")

    structure: str = ""
    reasoning: str = ""
    rating: float = Field(default=0.0, ge=0.0, le=1.0)


class StructureRatings(BaseModel):
    """The answer of one attention call: one rating for each offered structure."""

    model_config = ConfigDict(extra="forbid")

    ratings: list[StructureRating] = Field(default_factory=list)


class RatedStructure(BaseModel):
    """The attention score of one structure for one shard.

    ``score`` is ``rating + aux / max_aux * COALITION_AUX_RATING_FACTOR``.
    The auxiliary part only breaks ties, as in the old code. ``won`` is True
    for the structures with the best scores of the shard, up to the winner
    count of the mental aperture. Only the records of a winner can enter the
    shard as newcomers.
    """

    structure_id: int = 0
    label: str = ""
    rating: float = 0.0
    aux: float = 0.0
    score: float = 0.0
    reasoning: str = ""
    won: bool = False


class ShardRatings(BaseModel):
    """The attention scores of all structures for one shard."""

    key: ShardKey = Field(default_factory=ShardKey)
    ratings: list[RatedStructure] = Field(default_factory=list)


class AttentionResult(BaseModel):
    """Result of the attention phase.

    ``ratings`` holds one entry for each updated shard. ``changes`` holds the
    entered, left, and kept items and the tail of each updated shard. The
    commit copies them into the turn record. ``focus`` is the effective
    deliberate focus advice, or None. ``focus_ids`` are the records that got
    a ``deliberate_focus`` endorsement. ``calls`` counts the rating calls.
    """

    ratings: list[ShardRatings] = Field(default_factory=list)
    changes: list[ShardChange] = Field(default_factory=list)
    focus: FocusState | None = None
    focus_ids: list[int] = Field(default_factory=list)
    calls: int = 0


class GoalChange(BaseModel):
    change: str = ""  # retain | add | complete | reprioritize | defer | refuse
    goal: str = ""
    goal_key: str = ""  # stable identity; falls back to normalized goal text
    reason: str = ""
    priority: float = 0.5


class GoalsResult(BaseModel):
    changes: list[GoalChange] = Field(default_factory=list)


class GroupExperience(BaseModel):
    """The subjective experience of one stimulus group in a cycle with several groups.

    The call reads only the self shard and the shard of ``key``. So the
    decision and the reply for this group do not read the content of the
    other groups. ``record_id`` is a pending id until the commit.
    """

    key: ShardKey = Field(default_factory=ShardKey)
    text: str = ""
    seed: str = ""
    record_id: int = 0


class ExperienceResult(BaseModel):
    """Result of the experience phase: the subjective experience of this moment.

    Port of ``_generate_subjective_experience``. ``text`` is the experience
    in the companion's felt inner voice: the seed phrases, then the model
    continuation, then the focus phrase. ``seed`` holds the seed phrases
    alone. ``record_id`` is the pending id of the ``SubjectiveExperience``
    feature until the commit stores it. The record is in the tail of the
    self shard, so the goals, decide, and route calls read it.

    ``text`` covers the whole moment. When more than one group is
    significant, ``groups`` also holds one experience for each of them.
    The decide and route calls and the reply of a group read the experience
    of their own group (:meth:`text_for`) and do not see the others
    (:meth:`hidden_for`).
    """

    text: str = ""
    seed: str = ""
    record_id: int = 0
    groups: list[GroupExperience] = Field(default_factory=list)

    def text_for(self, key: ShardKey) -> str:
        """Return the experience that the calls of group ``key`` read."""
        own = [item.text for item in self.groups if item.key == key]
        return own[0] if own else self.text

    def hidden_for(self, key: ShardKey | None) -> list[int]:
        """Return the experience record ids of this cycle that a view of group ``key`` leaves out.

        A view of one group shows its own group experience only. Without
        group experiences, or for the whole-mind view (``key`` None),
        nothing is hidden.
        """
        if key is None or not self.groups:
            return []
        return [self.record_id, *(item.record_id for item in self.groups if item.key != key)]

    def rekey(self, mapping: dict[int, int]) -> None:
        """Replace the pending record ids by the stored record ids."""
        self.record_id = mapping.get(self.record_id, self.record_id)
        for item in self.groups:
            item.record_id = mapping.get(item.record_id, item.record_id)


#: The action types that routing can choose.
ACTION_TYPES = ("Reply", "ToolCall", "ToolCallAndReply", "Ignore", "Idle", "Sleep")

#: Action types that make no dispatch. They stay in the decision record only.
QUIET_ACTIONS = frozenset({"Ignore", "Idle", "Sleep"})

#: Openings of planner or ticket text. A decision is a choice in the story, not a task for an assistant.
TICKET_OPENINGS = ("generate", "validate", "acknowledge", "provide", "ensure", "respond to", "offer",
                   "continue the conversation", "engage", "address")

#: System words that a decision must not hold. A workflow name is checked separately.
SYSTEM_WORDS = ("ToolCall", "workflow", "action_type")


class DecisionCheck(BaseModel):
    """The validation context of one decide answer.

    ``workflow_names`` are the configured workflow names. A decision that
    names one is routing text, not a choice of the character.
    """

    workflow_names: list[str] = Field(default_factory=list)


class DecisionAnswer(BaseModel):
    """The answer of one decide call: zero or ``DECISION_VARIANTS`` decision texts.

    An empty list means that the character does nothing about this group.
    A choice to stay silent or to wait is a decision too, and the route
    phase maps it to a quiet action. Validation rejects a text that names a
    workflow or a system word, or that opens like a planner ticket. The
    repair turn then asks for the choice in the words of the story.
    """

    model_config = ConfigDict(extra="forbid")

    variants: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _story_words(self, info: ValidationInfo):
        if len(self.variants) not in (0, DECISION_VARIANTS):
            raise ValueError(f"Write exactly {DECISION_VARIANTS} decision variants, or an empty list.")
        check = DecisionCheck.model_validate(info.context or {})
        for text in self.variants:
            words = text.strip()
            if not words:
                raise ValueError("A decision variant is empty.")
            named = [name for name in (*check.workflow_names, *SYSTEM_WORDS) if name in words]
            if named:
                raise ValueError(f"The decision names system terms ({', '.join(named)}). Write what the "
                                 "character does in the story, in the words of the story.")
            if words.lower().startswith(TICKET_OPENINGS):
                raise ValueError("The decision reads like a task ticket. Write it as a recommendation to the "
                                 "character, in the character's own voice and attitude.")
        return self


class RouteCheck(BaseModel):
    """The validation context of one route answer: the available workflow names."""

    workflow_names: list[str] = Field(default_factory=list)


class RouteChoice(BaseModel):
    """The action that carries out one decision.

    ``action_type`` is one of :data:`ACTION_TYPES`. A quiet type
    (:data:`QUIET_ACTIONS`) makes no dispatch and may leave
    ``workflow_name`` empty. Every other type names an available workflow.
    ``arguments`` are the workflow arguments. ``presentation`` is the public
    presentation choice. The route never writes the decision text: the
    phase passes it to the workflow unchanged.
    """

    action_type: str = "Idle"
    workflow_name: str = ""
    arguments: ActionArguments = Field(default_factory=ActionArguments)
    presentation: Presentation = Field(default_factory=Presentation)
    reasoning: str = ""


class RouteAnswer(RouteChoice):
    """The answer of one route call, validated against the available workflows.

    The validation context is a :class:`RouteCheck`. An unknown action type,
    an unavailable workflow, or an invalid Sleep duration is invalid model
    output and gets a repair turn. The cycle stores the validated answer as
    a :class:`RouteChoice`, because a stored copy has no validation context.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _known(self, info: ValidationInfo):
        if self.action_type not in ACTION_TYPES:
            raise ValueError(f"Unsupported action_type {self.action_type!r}. Allowed: {', '.join(ACTION_TYPES)}.")
        allowed = RouteCheck.model_validate(info.context or {}).workflow_names
        if self.action_type not in QUIET_ACTIONS and self.workflow_name not in allowed:
            raise ValueError(f"workflow_name {self.workflow_name!r} is not available. "
                             f"Available: {', '.join(allowed)}.")
        if self.action_type == "Sleep" and "timeout_seconds" in self.arguments.model_fields_set:
            WaitRequest(seconds=self.arguments.timeout_seconds)
        return self


class RoutedDecision(BaseModel):
    """One decision of the cycle with its route.

    ``decision`` is the decision text, unchanged. ``group``,
    ``stimulus_ids``, and ``correlation_id`` come from the stimulus group.
    ``route`` is the routing answer. A quiet route makes no dispatch. A
    Sleep route sets the wait of the cycle.
    """

    group: ShardKey = Field(default_factory=ShardKey)
    stimulus_ids: list[int] = Field(default_factory=list)
    correlation_id: str = ""
    decision: str = ""
    route: RouteChoice = Field(default_factory=RouteChoice)


class ExpectationAnswerItem(BaseModel):
    """One expectation in the answer of the anticipation call."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(default="", min_length=1)
    affective_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    urgency: float = Field(default=0.3, ge=0.0, le=1.0)


class ExpectationAnswer(BaseModel):
    """The answer of the anticipation call: 1 to ``EXPECTATION_COUNT`` expectations."""

    model_config = ConfigDict(extra="forbid")

    expectations: list[ExpectationAnswerItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _count(self):
        if not 1 <= len(self.expectations) <= EXPECTATION_COUNT:
            raise ValueError(f"Write between 1 and {EXPECTATION_COUNT} expectations.")
        return self


class OpenExpectation(BaseModel):
    """One pending expectation that the host offers to the cycle.

    The fields copy the ``ExpectationRecord``. ``revision`` is the revision
    that the cycle read, so the commit can reject a concurrent change.
    ``tick_id`` is the tick that recorded it.
    """

    record_id: int = 0
    content: str = ""
    expected_valence: float = 0.0
    urgency: float = 0.3
    fulfillment: float = 0.0
    tick_id: int = 0
    revision: int = 1
    about_entity_ids: list[int] = Field(default_factory=list)


class ExpectationFulfillment(BaseModel):
    """The result for one expectation in the answer of the expectation check.

    ``fulfillment_score`` is 0 when the stimulus contradicts the
    expectation, 0.5 when it is unrelated, and 1 when it confirms it.
    ``directness_score`` is how directly the stimulus addresses it.
    """

    model_config = ConfigDict(extra="forbid")

    expectation_id: int = 0
    fulfillment_score: float = Field(default=0.5, ge=0.0, le=1.0)
    directness_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ExpectationCheckAnswer(BaseModel):
    """The answer of the expectation check call: one result for each offered expectation."""

    model_config = ConfigDict(extra="forbid")

    fulfillments: list[ExpectationFulfillment] = Field(default_factory=list)


class ExpectationOutcome(BaseModel):
    """The outcome of one checked expectation in this cycle.

    ``previous`` and ``fulfillment`` are the fulfillment before and after
    the check: ``fulfillment = previous + (score - previous) * directness``.
    ``valence_impact`` and ``fear_impact`` are the old valence and anxiety
    impacts. ``status`` is ``pending``, ``supported``, or ``contradicted``.
    ``observation_ids`` are the stimuli of the check.
    """

    record_id: int = 0
    revision: int = 1
    content: str = ""
    previous: float = 0.0
    fulfillment: float = 0.0
    change: float = 0.0
    fulfillment_score: float = 0.5
    directness_score: float = 0.0
    valence_impact: float = 0.0
    fear_impact: float = 0.0
    status: str = "pending"
    observation_ids: list[int] = Field(default_factory=list)


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
    "ActionMetadata",
    "ActionRequest",
    "PendingDispatch",
    "RecordMetadataView",
    "RequestShard",
    "ActivationScore",
    "AgentCodeletAnswer",
    "AgentCodeletJob",
    "AppraisalAffect",
    "AppraisalResult",
    "AppraisalView",
    "AttentionResult",
    "BeliefUpdate",
    "CodeletCost",
    "CodeletOutput",
    "CodeletRatings",
    "CodeletRun",
    "CodeletVariant",
    "CodeletsResult",
    "ContextResult",
    "FocusProposalRef",
    "GoalChange",
    "GoalRecordView",
    "GoalsResult",
    "ManagedRunView",
    "MemoryCandidate",
    "MentalProposal",
    "PathwayBoost",
    "PhaseReceipt",
    "RecordView",
    "ReflectionGuidance",
    "Stimulus",
    "TriageItem",
    "TriageResult",
    "VariantPercept",
    "WorkflowOption",
    "BinderAnswer",
    "BoundRelation",
    "BoundStructure",
    "RatedStructure",
    "ShardRatings",
    "ShardStream",
    "StructureDraft",
    "StructureRating",
    "StructureRatings",
    "StructuresResult",
    "ACTION_TYPES",
    "TRIAGE_CATEGORIES",
    "DecisionAnswer",
    "DecisionCheck",
    "ExpectationAnswer",
    "ExpectationAnswerItem",
    "ExpectationCheckAnswer",
    "ExpectationDraft",
    "ExpectationFulfillment",
    "ExpectationOutcome",
    "ExperienceResult",
    "GroupExperience",
    "OpenExpectation",
    "QUIET_ACTIONS",
    "RouteAnswer",
    "RouteCheck",
    "RouteChoice",
    "RoutedDecision",
    "SYSTEM_WORDS",
    "TICKET_OPENINGS",
]
