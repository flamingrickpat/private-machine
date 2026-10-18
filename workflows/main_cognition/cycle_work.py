"""The shared typed state of one main cognition cycle.

The cycle is one pm-workflows manifest with one Python phase for each step:
intake, retrieve, triage, appraise, codelets, structures, attention,
experience, goals, decide, route, and commit. All phases share one
:class:`CycleWork` object.

State flow between phases:

- The host writes :class:`CycleRequest` to ``inbox/request.yaml``. The
  workflow runner does this for every invocation.
- The intake phase validates the request and creates the first
  ``CycleWork``.
- Every phase reads ``CycleWork`` from :data:`WORK_FILE` in the working
  target and validates it at entry. It returns the changed object, and the
  phase entry point writes it back. The working target of main cognition is
  an in-memory Bash machine, so the file never reaches the disk.
- The commit phase writes the final ``CycleWork`` to ``outbox/report.yaml``.
  The host validates it again and stages the one write batch.

No phase writes the memory store. The host owns the store and the bus.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState
from pm.records.mental_state import MentalStateSample, PersonalityProfile
from workflows.cognitive_control.schemas import FocusState, WaitRequest
from workflows.main_cognition.schemas import (
    ActionDispatch,
    ActionSelectionResult,
    AppraisalAffect,
    AppraisalResult,
    AppraisalView,
    AttentionResult,
    CodeletsResult,
    ContextResult,
    FocusProposalRef,
    GoalRecordView,
    GoalsResult,
    ManagedRunView,
    MemoryCandidate,
    MentalProposal,
    PhaseReceipt,
    RecordView,
    ReflectionGuidance,
    Stimulus,
    TriageResult,
    WorkflowOption,
    WorkspaceResult,
)
from workflows.public_speech.public_context import PublicContext

#: Path of the shared phase state in the working target.
WORK_FILE = "scratch/cycle_work.json"


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
    codelet prompts open their shared story prefix with it.
    """

    run_id: str = ""
    correlation_id: str = ""
    companion_name: str = ""
    user_name: str = ""
    user_entity_id: int | None = None
    state_summary: str = ""
    previous_state: CycleState | None = None
    cycle_time: datetime = Field(default_factory=datetime.utcnow)
    state_history: list[MentalStateSample] = Field(default_factory=list)
    personality: PersonalityProfile = Field(default_factory=PersonalityProfile)
    active_goals: list[str] = Field(default_factory=list)
    stimuli: list[Stimulus] = Field(default_factory=list)
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    memory_trace: dict = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    available_workflows: list[WorkflowOption] = Field(default_factory=list)
    public_context: PublicContext | None = None
    managed_runs: list[ManagedRunView] = Field(default_factory=list)
    # Inputs of the codelets phase. The module derives them from committed state.
    stimulus_revisions: dict[int, int] = Field(default_factory=dict)
    active_goal_records: list[GoalRecordView] = Field(default_factory=list)
    seconds_since_user_input: float | None = None
    character_card: str = ""
    # Path of a codelet definitions file. Empty selects codelets.yaml.
    codelet_definitions: str = ""
    # A view of every record that the CSM can hold in this cycle: the items
    # of the previous CSM, the stimuli, the recent dialogue, and the memory
    # candidates. JSON stores the keys as text; validation restores ints.
    records: dict[int, RecordView] = Field(default_factory=dict)
    # The last dialogue turns before the stimuli, oldest first.
    recent_dialogue_ids: list[int] = Field(default_factory=list)
    active_focus: FocusState | None = None
    focus_proposals: list[FocusProposalRef] = Field(default_factory=list)
    mental_proposals: list[MentalProposal] = Field(default_factory=list)
    reflection_guidance: list[ReflectionGuidance] = Field(default_factory=list)
    wait_default_seconds: float = 1800


class CycleWork(BaseModel):
    """The state that the phases of one cycle share.

    ``state`` is the previous cycle state. The phases change it in place:
    intake decays the CSM and sets the stimulus groups, appraise sets the
    EMA mental state, decide sets the decisions. The host sets the tick, the
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
    to a new percept (see :meth:`allocate_pending_id`).

    Each phase result field holds an empty result until its phase runs.
    """

    request: CycleRequest = Field(default_factory=CycleRequest)
    state: CycleState = Field(default_factory=lambda: CycleState(tick_id=0))
    state_summary: str = ""
    phases: list[PhaseReceipt] = Field(default_factory=list)
    triage: TriageResult = Field(default_factory=TriageResult)
    appraisal: AppraisalResult = Field(default_factory=AppraisalResult)
    affect: AppraisalAffect = Field(default_factory=AppraisalAffect)
    context: ContextResult = Field(default_factory=ContextResult)
    selected_memories: list[MemoryCandidate] = Field(default_factory=list)
    codelets: CodeletsResult = Field(default_factory=CodeletsResult)
    attention: AttentionResult = Field(default_factory=AttentionResult)
    goals: GoalsResult = Field(default_factory=GoalsResult)
    workspace: WorkspaceResult = Field(default_factory=WorkspaceResult)
    action_selection: ActionSelectionResult = Field(default_factory=ActionSelectionResult)
    dispatches: list[ActionDispatch] = Field(default_factory=list)
    wait_request: WaitRequest | None = None
    summary: str = ""
    records: dict[int, RecordView] = Field(default_factory=dict)
    last_pending_id: int = -1

    def appraisal_view(self) -> AppraisalView:
        """Return the appraisal with the accepted feeling in words, for prompts."""
        return AppraisalView(**self.appraisal.model_dump(), felt_state=self.affect.felt)

    def receipt(self, name: str, summary: str) -> None:
        """Record that phase ``name`` ran."""
        self.phases.append(PhaseReceipt(name=name, summary=summary))

    def allocate_pending_id(self) -> int:
        """Return a new pending record id for a record that the commit creates.

        The memory store assigns record ids only at commit. A new percept
        needs an id earlier, because the CSM holds it by id. Pending ids
        count down from -2: -1 is the store's "unassigned" id and 0 is no
        record. The commit replaces each pending id by the record id.
        """
        self.last_pending_id -= 1
        return self.last_pending_id


def primary_group(work: CycleWork) -> StimulusGroup | None:
    """Return the group that owns the appraisal and the decisions of this cycle.

    It is the group that carries the cycle correlation, or the first group.
    The result is None for a cycle without stimuli, for example a wake event
    that resolved to no record. Such a cycle records no decisions.
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


def read_work(environment) -> CycleWork:
    """Read and validate the shared phase state from the working target.

    A missing or invalid file is a bug in the phase order and raises.
    """
    return CycleWork.model_validate_json(environment.read_text(WORK_FILE))


def write_work(environment, work: CycleWork) -> None:
    """Write the shared phase state to the working target."""
    environment.write_text(WORK_FILE, work.model_dump_json())


__all__ = ["CycleRequest", "CycleWork", "WORK_FILE", "primary_group", "provisional_tick", "read_work", "write_work"]
