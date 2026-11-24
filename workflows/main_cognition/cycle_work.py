"""The shared typed state of one main cognition cycle.

The cycle is one pm-workflows manifest with one Python phase for each step:
intake, triage, appraise, retrieve, codelets, structures, attention,
experience, goals, decide, route, and commit. All phases share one
:class:`CycleWork` object.

State flow between phases:

- The host writes :class:`CycleRequest` to ``inbox/request.yaml``. The
  workflow runner does this for every invocation.
- The intake phase validates the request and creates the first
  ``CycleWork``.
- Every phase receives the same typed ``CycleWork`` object. No intermediate
  phase state is serialized, parsed, or persisted. Interrupted cycles restart
  from the accepted Ghost rather than resuming halfway through deliberation.
- The commit phase writes the final ``CycleWork`` to ``outbox/report.yaml``.
  The host validates it again and stages the one write batch.

No phase writes the memory store. The host owns the store and the bus.
"""
from __future__ import annotations

from pm.llm.session import ModelSession

from pm.memory.retrieval import RetrievalTrace

from pm import vclock
from pm.records.media import RecordImage

from datetime import datetime

from pydantic import BaseModel, Field


from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState
from pm.records.mental_state import MentalStateSample, PersonalityProfile
from workflows.cognitive_control.schemas import FocusState, WaitRequest
from workflows.main_cognition.schemas import (
    ActionDispatch,
    AppraisalAffect,
    AppraisalResult,
    AppraisalView,
    AttentionResult,
    CodeletsResult,
    ContextResult,
    ExpectationOutcome,
    ExperienceResult,
    FocusProposalRef,
    GoalRecordView,
    GoalsResult,
    ManagedRunView,
    MemoryCandidate,
    MentalProposal,
    OpenExpectation,
    PhaseReceipt,
    RecordView,
    ReflectionGuidance,
    RoutedDecision,
    ShardStream,
    Stimulus,
    StructuresResult,
    TriageResult,
    WorkflowOption,
)
from workflows.public_speech.public_context import PublicContext
from workflows.main_cognition.interval_review import IntervalReview
from workflows.public_speech.situational_guidance import SituationalGuidance


class CycleRequest(BaseModel):
    """The frozen input of one main cycle.

    The host derives every field from committed state and records the
    revisions it read. ``previous_state`` is the last accepted cycle state, or
    None before the first cycle. The cycle changes a copy of it.
    ``user_entity_id`` is the entity record of the user, or None when the
    persona has no such record. ``memory_trace`` is the retrieval trace of
    ``pm.memory``. The cycle stores it unchanged in record metadata.

    ``cycle_time`` is the naive UTC time at which the host built the request.
    The EMA mental state measures elapsed time up to it. ``state_history``
    holds one sample for each causal feature in the store, oldest first.
    ``personality`` is the stable profile of the persona.

    ``character_card`` is the story character card of the companion. The
    shared story prefix opens with it.

    ``self_entity_id`` is the entity record of the companion, or None.
    ``entity_names`` maps the entity of each stimulus group to its name.
    ``narrative_ids`` are the newest narrative records of the self and of the
    group entities. The retrieve phase puts them into the CSM. ``streams``
    hold the newest stream records of each shard, for the shard tails.
    ``open_expectations`` are the newest pending expectations of the real,
    in-world reality, recorded by either loop, with the revisions that the
    cycle read. ``observed_stimulus_ids`` are the stimuli whose evidence role is
    observed (``pm.memory.evidence.OBSERVED_EVIDENCE_ROLES``); only they can
    move an expectation.
    """

    run_id: str = ""
    correlation_id: str = ""
    companion_name: str = ""
    user_name: str = ""
    user_entity_id: int | None = None
    self_entity_id: int | None = None
    entity_names: dict[int, str] = Field(default_factory=dict[int, str])
    state_summary: str = ""
    previous_state: CycleState | None = None
    cycle_time: datetime = Field(default_factory=vclock.utcnow)
    state_history: list[MentalStateSample] = Field(default_factory=list[MentalStateSample])
    appraised_partner_input_ids: list[int] = Field(default_factory=list[int],
        description="This request's partner inputs already claimed by an earlier accepted main cycle.")
    personality: PersonalityProfile = Field(default=PersonalityProfile(
        openness=0.5, conscientiousness=0.5, extraversion=0.5, agreeableness=0.5, neuroticism=0.5,
        bas_drive=0.5, bis_sensitivity=0.5, attachment_anxiety=0.5, attachment_avoidance=0.5,
        affiliation_need=0.5, self_control=0.5, propriety_baseline=0.5, humor_playfulness=0.5,
        warmth_expressive=0.5, bluntness_assertive=0.5))
    active_goals: list[str] = Field(default_factory=list[str])
    stimuli: list[Stimulus] = Field(default_factory=list[Stimulus])
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list[MemoryCandidate])
    memory_trace: RetrievalTrace | None = None
    latest_body_observation: str = ""
    latest_body_observation_id: int | None = None
    runtime_orientation: str = ""
    capabilities: list[str] = Field(default_factory=list[str])
    available_workflows: list[WorkflowOption] = Field(default_factory=list[WorkflowOption])
    public_context: PublicContext | None = None
    situational_guidance: list[SituationalGuidance] = Field(default_factory=list[SituationalGuidance])
    early_triage: TriageResult | None = None
    managed_runs: list[ManagedRunView] = Field(default_factory=list[ManagedRunView])
    # Inputs of the codelets phase. The module derives them from committed state.
    stimulus_revisions: dict[int, int] = Field(default_factory=dict[int, int])
    active_goal_records: list[GoalRecordView] = Field(default_factory=list[GoalRecordView])
    seconds_since_user_input: float | None = None
    character_card: str = ""
    images: tuple[RecordImage, ...] = ()
    # Path of a codelet definitions file. Empty selects codelets.yaml.
    codelet_definitions: str = ""
    # A view of every record that the CSM can hold in this cycle: the items
    # of the previous CSM, the stimuli, the recent dialogue, and the memory
    # candidates. JSON stores the keys as text; validation restores ints.
    records: dict[int, RecordView] = Field(default_factory=dict[int, RecordView])
    # The last dialogue turns before the stimuli, oldest first.
    recent_dialogue_ids: list[int] = Field(default_factory=list[int])
    narrative_ids: list[int] = Field(default_factory=list[int])
    streams: list[ShardStream] = Field(default_factory=list[ShardStream])
    open_expectations: list[OpenExpectation] = Field(default_factory=list[OpenExpectation])
    #: Stimuli that report an observed event. A thought, a reconsideration
    #: marker, or another derived stimulus is left out and moves no expectation.
    observed_stimulus_ids: list[int] = Field(default_factory=list[int])
    #: What the companion did since the last accepted review. The main host
    #: always sets it; None only in a request that never reaches the appraise
    #: or goals phase (grouping helpers, triage, the interaction fast path),
    #: and those phases raise when it is missing.
    interval_review: IntervalReview | None = None
    active_focus: FocusState | None = None
    focus_proposals: list[FocusProposalRef] = Field(default_factory=list[FocusProposalRef])
    mental_proposals: list[MentalProposal] = Field(default_factory=list[MentalProposal])
    reflection_guidance: list[ReflectionGuidance] = Field(default_factory=list[ReflectionGuidance])
    wait_default_seconds: float = 1800
    #: Quiet window that a spoken reply arms. Zero means the operator disabled it.
    post_reply_seconds: float = 0.0
    #: Ordinary sources whose local action belongs to the interaction workflow.
    interaction_source_ids: list[int] = Field(default_factory=list[int])
    #: Explicit partner requests whose task dispatch was committed by early triage.
    workflow_request_source_ids: list[int] = Field(default_factory=list[int])

    def reserved_selection_sources(self) -> set[int]:
        """Return sources main cognition can appraise without selecting another action."""
        return set(self.interaction_source_ids).union(self.workflow_request_source_ids)


class CycleWork(BaseModel):
    """The state that the phases of one cycle share.

    ``state`` is the previous cycle state. The phases change it in place:
    intake decays the CSM and sets the stimulus groups, appraise sets the
    EMA mental state, decide sets the decisions, route sets the dispatch
    references. The host sets the tick, the action record ids of the
    dispatch references, and the broadcast queue at commit.

    ``state_summary`` is the state text that prompts read. Before appraisal it
    is the last broadcast summary. After appraisal it is the accepted mental
    state in words (``affect.felt``).

    ``affect`` holds the appraisal state, its delta, the relationship
    partner, and the regulation mode. The commit stores them on the
    appraisal ``Feature``.

    ``records`` holds the record views that the CSM can name. Intake copies
    them from the request, and the codelets phase adds the views of new
    percepts. ``last_pending_id`` is the last pending id that this cycle gave
    to a new percept or structure (see :meth:`allocate_pending_id`).

    ``structures`` holds the structures of this cycle. ``attention`` holds
    the shard ratings and the shard changes. The attention phase changes the
    shards of ``state.workspace`` in place.

    ``expectations`` are the outcomes of the expectation check of the
    appraise phase. ``experience`` is the subjective experience of the
    experience phase. ``routes`` holds one routed decision for each decision
    of ``state.decisions``, and ``dispatches`` the actions that they make.

    Each phase result field holds an empty result until its phase runs.
    """

    request: CycleRequest = Field(default_factory=CycleRequest)
    prompt_summaries: dict[int, str] = Field(default_factory=dict[int, str])  # pm-lint: allow PM004 -- derived prompt text indexed by original stimulus ID
    state: CycleState = Field(default_factory=lambda: CycleState(tick_id=0))
    state_summary: str = ""
    phases: list[PhaseReceipt] = Field(default_factory=list[PhaseReceipt])
    triage: TriageResult = Field(default_factory=TriageResult)
    appraisal: AppraisalResult = Field(default_factory=AppraisalResult)
    affect: AppraisalAffect = Field(default_factory=AppraisalAffect)
    context: ContextResult = Field(default_factory=ContextResult)
    selected_memories: list[MemoryCandidate] = Field(default_factory=list[MemoryCandidate])
    codelets: CodeletsResult = Field(default_factory=CodeletsResult)
    structures: StructuresResult = Field(default_factory=StructuresResult)
    attention: AttentionResult = Field(default_factory=AttentionResult)
    expectations: list[ExpectationOutcome] = Field(default_factory=list[ExpectationOutcome])
    goals: GoalsResult = Field(default_factory=GoalsResult)
    experience: ExperienceResult = Field(default_factory=ExperienceResult)
    routes: list[RoutedDecision] = Field(default_factory=list[RoutedDecision])
    dispatches: list[ActionDispatch] = Field(default_factory=list[ActionDispatch])
    wait_request: WaitRequest | None = None
    summary: str = ""
    records: dict[int, RecordView] = Field(default_factory=dict[int, RecordView])
    last_pending_id: int = -1

    def prepare_stimulus_summaries(self, session: ModelSession) -> None:
        """Keep oversized stimuli readable by all phases while retaining their exact request text."""
        limits = session.render_settings()
        per_source = max(512, limits.input_ceiling // (4 * max(1, len(self.request.stimuli))))
        for stimulus in self.request.stimuli:
            text = session.summarize_long_source(stimulus.text, f"stimulus {stimulus.knoxel_id}", per_source)
            if text != stimulus.text:
                self.prompt_summaries[stimulus.knoxel_id] = text
                record = self.records[stimulus.knoxel_id].model_copy()
                record.content = text
                self.records[stimulus.knoxel_id] = record

    def appraisal_view(self) -> AppraisalView:
        """Return the appraisal with the accepted feeling in words, for prompts."""
        return AppraisalView(**self.appraisal.model_dump(), felt_state=self.affect.felt)

    def receipt(self, name: str, summary: str) -> None:
        """Record that phase ``name`` ran."""
        self.phases.append(PhaseReceipt(name=name, summary=summary))

    def allocate_pending_id(self) -> int:
        """Return a new pending record id for a record that the commit creates.

        The memory store assigns record ids only at commit. A new percept or
        structure needs an id earlier, because the CSM and the workspace hold
        it by id. Pending ids
        count down from -2: -1 is the store's "unassigned" id and 0 is no
        record. The commit replaces each pending id by the record id.
        """
        self.last_pending_id -= 1
        return self.last_pending_id


def primary_group(work: CycleWork) -> StimulusGroup | None:
    """Return the group that owns the appraisal of this cycle.

    It is the group that carries the cycle correlation, or the first group.
    Its entity is the relationship partner of the appraisal. Decisions are
    made for each group (see the decide phase). The result is None for a
    cycle without stimuli, for example a wake event that resolved to no
    record.
    """
    groups = work.state.stimulus_groups
    if not groups:
        return None
    return next((group for group in groups if group.correlation_id == work.request.correlation_id), groups[0])


def provisional_tick(work: CycleWork) -> int:
    """Return the tick that this cycle expects: the previous tick plus one.

    The host assigns the real tick at commit. Other modules can take ticks in
    between, so the real tick can be higher. The commit replaces this value
    in the CSM and the codelet state.
    """
    return work.state.tick_id + 1


__all__ = ["CycleRequest", "CycleWork", "primary_group", "provisional_tick"]
