"""Admit accepted deliberate mental actions into the shared private slot and deliver their results.

Main cognition selects a mental action and dispatches it to this module. The
module owns that action from then on: its admission, its lifecycle receipts,
and the delivery of its result. It runs no model and no agent. The private
loop executes the admitted work on its own timer, through the same inquiry
steps as spontaneous thought. :mod:`workflows.inner_thought.host` is the
canonical overview of that loop and of the inquiry graph.

One action moves through these stages:

1. **Requested.** ``action.request`` arrives with the accepted action record.
   The request waits in memory while an accepted wait, unclassified input, or
   required Critical reconsideration holds private work.
2. **Admitted.** :meth:`PrivateInquiryOwner.admit_deliberate_action` opens the
   deliberate episode. Spontaneous work that holds the slot is parked first,
   with a receipt that names this action, so a later reader can see why that
   question was set aside. Deliberate work never parks other deliberate work;
   the later action waits.
3. **Executed.** The private loop runs one operation per timer tick. Nothing
   here executes a step.
4. **Closed.** When the episode leaves the slot, this module commits the
   action's result identity and status. A finding becomes the result. An
   episode parked at a bound, displaced by Critical reconsideration, or closed
   because a goal or expectation it rested on changed has no finding. The module then commits a short first-person notice that the
   thinking stopped and why, so main cognition always hears back about an
   action it selected.
5. **Delivered.** A finding of a deliberate episode and a notice both carry
   the action's identity. This module publishes them as ``thought.created``
   under the action's correlation until an accepted main cycle claims them.
   Main cognition still accepts or rejects every proposed goal and focus
   change. A notice proposes nothing.

The action catalogue also carries a second name, ``theory_of_mind``: private
thinking about another entity that main cognition named as the action's
partner. It is ordinary deliberate work with one extra field, the subject's
entity id, so this module admits and delivers it exactly like any other
deliberate action. The private loop turns that subject into perspective
evidence; none of that happens here.

Recovery: an unadmitted request is replayed by dispatch recovery, because its
action stays pending. An admitted action is found again by its action identity,
so a restart repeats no inference and opens no second episode. Results that
the earlier standalone agent sessions committed are delivered the same way.

Telemetry: ``action.receipt`` when an action closes, ``mental.deferred`` when
an admission has to wait for a reason other than an accepted hold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from pm import vclock
from pm.host.application import Application
from pm.bus.envelopes import Envelope, JsonObject, new_envelope
from pm.host.application import BusModule
from pm.memory.snapshot import GhostCancelled, GhostConflict, Snapshot
from pm.memory.store import MemoryStore
from pm.records.base import Provenance
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.lineage import RecordLineage
from pm.records.memory_records import ActionSchemaRecord, Feature
from workflows.cognitive_control.schemas import ControlConfig
from workflows.inner_thought.graph_host import (
    MENTAL_ACTION_KEY, MENTAL_CORRELATION_KEY, PrivateInquiryOwner, accepted_source_ids, deliberate_action_ids,
    deliberate_inquiry, private_observer,
)
from workflows.inner_thought.inquiry import InquiryRecord, inquiry_history, inquiry_origin
from workflows.inner_thought.nominations import capture_inquiry_grant
from workflows.inner_thought.operations import OPERATIONS
from workflows.inner_thought.schemas import DeliberateAdmission
from workflows.main_cognition.schemas import ActionMetadata, ActionRequest

if TYPE_CHECKING:
    from workflows.cognitive_control.host import CognitiveControl

#: First operation of deliberate work when the action names none. The workflow
#: is offered to main cognition as scoped recall, so recall comes first.
DEFAULT_FIRST_OPERATION = "memory_search"

#: Second entry of the mental-action catalogue: deliberate thinking about
#: another person or agent, named by main cognition as the action's partner.
PERSPECTIVE_WORKFLOW = "theory_of_mind"

#: Workflow names whose public ``action.request`` envelopes this module admits.
#: Both names route to the same private slot; only the subject field differs.
MENTAL_WORKFLOWS: frozenset[str] = frozenset({"focus_management", PERSPECTIVE_WORKFLOW})


class MentalActionLifecycle(BaseModel):
    """The lifecycle fields of a mental action record that this module reads."""

    model_config = ConfigDict(extra="ignore")
    dispatch_status: str = Field(default="", description="'pending' or 'admitted' while main cognition's dispatch stands; 'deferred' after main withdrew it.")
    dispatch_deferred: bool = Field(default=False, description="True while main cognition holds the dispatch back.")
    mental_result_id: int | None = Field(default=None, description="Record that closed the action: its finding, or the receipt that parked or displaced its inquiry.")
    mental_status: str = Field(default="", description="Empty while the action is open. 'proposed' with a finding, 'parked' or 'displaced' without one, 'failed' for an earlier agent session.")


@dataclass(frozen=True, slots=True)
class RequestedMentalAction:
    """One dispatched mental action that has not entered the private slot yet."""

    action_record_id: int
    correlation: str
    instruction: str
    first_operation: str
    reason: str
    evidence_ids: tuple[int, ...]
    subject_entity_id: int | None = None


@dataclass(frozen=True, slots=True)
class MentalOutcome:
    """How a deliberate episode closes its action.

    ``finding_id`` is the outcome operation, or None when the episode stopped
    without one. ``stop_reason`` is the episode's own recorded reason.
    """

    finding_id: int | None
    mental_status: Literal["proposed", "parked", "displaced"]
    execution_status: Literal["completed", "parked", "cancelled"]
    stop_reason: str
    #: What displaced the episode; None unless ``mental_status`` is ``displaced``.
    displacement_cause: Literal["critical_epoch", "governing_source_changed"] | None


@dataclass(frozen=True, slots=True)
class ClosedMentalAction:
    """The committed close of one action: its result record and how it closed."""

    result_record_id: int
    outcome: MentalOutcome


@dataclass(frozen=True, slots=True)
class ResultDelivery:
    """One unappraised mental-action result that is due for publication."""

    record_id: int
    correlation: str
    gwt_tick: int | None


def requested_action(request: ActionRequest, correlation: str) -> RequestedMentalAction:
    """Read the selected mental task from main cognition's action request.

    The task is ``arguments.instruction``, else ``arguments.instructions``,
    else the decision text. ``arguments.operation`` chooses the first inquiry
    operation when it names a delivered one; any other value is model wording
    for another workflow, and deliberate work then starts with recall.
    """
    if request.knoxel_id is None:
        raise ValueError("A mental action request requires its accepted action record.")
    instruction = request.arguments.instruction or request.arguments.instructions or request.content
    if not instruction.strip():
        raise ValueError("Mental work requires a nonempty instruction.")
    first_operation = DEFAULT_FIRST_OPERATION
    if request.arguments.operation in OPERATIONS:
        first_operation = request.arguments.operation
    reason = request.content.strip() or "Main cognition selected this private mental work."
    evidence_ids = tuple(dict.fromkeys(stimulus.knoxel_id for stimulus in request.stimuli))
    subject_entity_id: int | None = None
    if request.workflow_name == PERSPECTIVE_WORKFLOW:
        # The person or agent this part of the story is about. Main cognition
        # names it as the action's partner; a missing partner is a broken
        # contract, because the perspective inquiry has nothing to model.
        subject_entity_id = request.partner_entity_id
        if subject_entity_id is None:
            raise ValueError("A theory-of-mind action requires the entity it is about.")
    return RequestedMentalAction(action_record_id=request.knoxel_id, correlation=correlation,
        instruction=instruction, first_operation=first_operation, reason=reason, evidence_ids=evidence_ids,
        subject_entity_id=subject_entity_id)


def mental_outcome(view: Snapshot, head: InquiryRecord) -> MentalOutcome | None:
    """Return how a deliberate episode closes its action, or None while it holds the slot.

    The newest outcome operation is the result when one exists. An episode
    parked at a step, time, or stall bound has none; neither has an episode
    that Critical reconsideration displaced before it reached an outcome.
    """
    state = head.receipt.state
    if state.status == "active":
        return None
    for accepted in reversed(inquiry_history(view, state.episode_id)):
        step = accepted.receipt.step
        if step is None or step.outcome == "continue":
            continue
        finding_id = accepted.receipt.operation_record_id
        if finding_id is None:
            raise ValueError("An executed inquiry step requires its operation record.")
        return MentalOutcome(finding_id=finding_id, mental_status="proposed", execution_status="completed",
                             stop_reason=state.stop_reason, displacement_cause=None)
    if state.status == "displaced":
        displacement = head.receipt.displacement
        if displacement is None:
            raise ValueError("A displaced inquiry requires its displacement receipt.")
        return MentalOutcome(finding_id=None, mental_status="displaced", execution_status="cancelled",
                             stop_reason=state.stop_reason, displacement_cause=displacement.cause)
    if state.status == "paused":
        return MentalOutcome(finding_id=None, mental_status="parked", execution_status="parked",
                             stop_reason=state.stop_reason, displacement_cause=None)
    raise ValueError("A completed or abandoned inquiry must end at an outcome operation.")


def unfinished_notice(instruction: str, outcome: MentalOutcome) -> str:
    """Word the first-person notice for deliberate work that stopped without a finding.

    The text states only recorded facts: the accepted instruction, that no
    conclusion exists, and the episode's own stop reason. It invents no thought
    content.
    """
    if outcome.displacement_cause == "critical_epoch":
        return (f"I set out to think this through: {instruction} Something critical came up before I "
                "reached a conclusion, so that thinking was interrupted.")
    if outcome.displacement_cause == "governing_source_changed":
        return (f"I set out to think this through: {instruction} A goal or expectation it rested on changed "
                "before I reached a conclusion, so I stopped that line of thinking.")
    return (f"I set out to think this through: {instruction} I stopped before I reached a conclusion "
            f"({outcome.stop_reason}). The question is parked for now.")


class FocusManagement(BusModule):
    """Own selected mental actions: admission to the private slot, lifecycle, and result delivery.

    The module holds no model and changes no goal, emotion, or workspace. The
    private loop executes the work; main cognition appraises the result.
    """

    DATA_TOPICS = frozenset({"action.request"})
    WORKFLOW_NAME = "focus_management"

    def accepts(self, envelope: Envelope) -> bool:
        """Take ``action.request`` also when a route names the perspective workflow.

        A new menu name needs its own route target (audit A1 in the B06 plan),
        and the route target is the workflow name. Both catalogue names send
        their requests to this one module, which owns the shared private slot.
        """
        if envelope.kind != "control" and envelope.topic in self.DATA_TOPICS and envelope.target in MENTAL_WORKFLOWS:
            return True
        return super().accepts(envelope)

    def __init__(self, module_id: str, application: Application, tick_seconds: float | None = None, *, ghost: MemoryStore) -> None:
        """Bind the module to the shared memory store. The timer retries admission and delivery."""
        super().__init__(module_id, application, tick_seconds)
        self.ghost: MemoryStore = ghost
        self.cognitive_control: CognitiveControl | None = None
        # Results already put on the bus, so a timer tick never publishes twice.
        self._published_results: set[int] = set()
        # Dispatched actions that still wait for the private slot, by action ID.
        self._requested: dict[int, RequestedMentalAction] = {}
        # Admitted actions whose episode has not closed them yet. None until
        # the first tick rebuilds it from memory, so a restart finds them again.
        self._open_actions: set[int] | None = None
        # Last reported reason an admission had to wait, by action ID.
        self._deferral_reasons: dict[int, str] = {}

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Register new requests, then advance every action. A timer tick has an empty batch."""
        for envelope in batch:
            request = ActionRequest.model_validate(envelope.payload)
            requested = requested_action(request, envelope.correlation_id)
            self._requested[requested.action_record_id] = requested
        self._admit_requested()
        self._close_finished_actions()
        self.resume()

    def _limits(self) -> ControlConfig:
        """Return the operator episode limits, or the shipped defaults."""
        control = self.cognitive_control
        return control.config if control is not None else ControlConfig()

    def _admit_requested(self) -> None:
        """Give each requested action the private slot when current authority permits it.

        An accepted wait, unclassified input, and required Critical
        reconsideration hold every admission until a later tick. A withdrawn or
        already answered action is dropped. Other deliberate work in the slot
        makes the action wait, with one ``mental.deferred`` telemetry event.
        """
        control = self.cognitive_control
        for action_id in sorted(self._requested):
            requested = self._requested[action_id]
            with self.ghost.lock:
                if control is not None and (control.waiting() or control.reconsideration_required()):
                    return
                view = self.ghost.snapshot(capture=False)
                action = view.get(action_id)
                if not isinstance(action, ActionSchemaRecord):
                    raise TypeError("A mental action request must name an accepted action record.")
                lifecycle = MentalActionLifecycle.model_validate(action.metadata)
                if lifecycle.mental_result_id is not None:
                    del self._requested[action_id]
                    continue
                if deliberate_inquiry(view, action_id) is not None:
                    # A replayed dispatch: the episode already exists.
                    del self._requested[action_id]
                    self._tracked_actions().add(action_id)
                    continue
                if lifecycle.dispatch_status not in {"pending", "admitted"} or lifecycle.dispatch_deferred:
                    # Main cognition withdrew this dispatch for reconsideration.
                    del self._requested[action_id]
                    self._report_deferral(requested, "Main cognition withdrew the action before it entered the private slot.")
                    continue
                observer = private_observer(view, self.ghost.get_companion_name())
                # Evidence must lie in the observer's reality and come from an
                # accepted commit, because the entry is a recovery root.
                accepted_ids = accepted_source_ids(self.ghost)
                evidence_ids = tuple(source_id for source_id in requested.evidence_ids
                                     if source_id in accepted_ids and view.get(source_id).scope == observer.scope)
                grant = capture_inquiry_grant(view, observer_entity_id=observer.id, scope=observer.scope,
                    source_ids=[action_id, *evidence_ids],
                    critical_epoch=control.publication_epoch() if control is not None else 0)
                generation = control.generation if control is not None else 0

                from workflows.inner_thought.graph_host import PrivateCommitOwner
                owner = PrivateInquiryOwner(self.ghost,
                    commit_owner=PrivateCommitOwner(self.ghost, control, generation),
                    cancel=self.stop_event, limits=self._limits())
                admission = DeliberateAdmission(action_record_id=action_id, instruction=requested.instruction,
                    evidence_ids=evidence_ids, operation=requested.first_operation, reason=requested.reason,
                    subject_entity_id=requested.subject_entity_id)
                try:
                    owner.admit_deliberate_action(admission, grant=grant, now_utc=vclock.utcnow())
                except GhostCancelled:
                    # A wait or Critical reconsideration arrived during the commit.
                    return
                except GhostConflict as conflict:
                    self._report_deferral(requested, str(conflict))
                    continue
                del self._requested[action_id]
                self._deferral_reasons.pop(action_id, None)
                self._tracked_actions().add(action_id)

    def _report_deferral(self, requested: RequestedMentalAction, reason: str) -> None:
        """Publish why an admission waits, once for each new reason."""
        if self._deferral_reasons.get(requested.action_record_id) == reason:
            return
        self._deferral_reasons[requested.action_record_id] = reason
        self.bus.put(new_envelope(self.module_id, "telemetry", "mental.deferred", requested.correlation,
            {"workflow_name": self.WORKFLOW_NAME, "action_schema_id": requested.action_record_id, "reason": reason}))

    def _tracked_actions(self) -> set[int]:
        """Return the admitted actions that are not closed, rebuilding them once from memory."""
        if self._open_actions is None:
            with self.ghost.lock:
                view = self.ghost.snapshot(capture=False)
                open_actions: set[int] = set()
                for action_id in deliberate_action_ids(view):
                    lifecycle = MentalActionLifecycle.model_validate(view.get(action_id).metadata)
                    if lifecycle.mental_result_id is None:
                        open_actions.add(action_id)
                self._open_actions = open_actions
        return self._open_actions

    def _close_finished_actions(self) -> None:
        """Commit the result identity of every admitted action whose episode left the slot."""
        tracked = self._tracked_actions()
        for action_id in sorted(tracked):
            closed = self._close_action(action_id)
            if closed is None:
                continue
            tracked.discard(action_id)
            self._receipt(action_id, closed)

    def _close_action(self, action_id: int) -> ClosedMentalAction | None:
        """Commit one action's lifecycle when its episode has an outcome, else return None.

        The finding is committed by the private loop in an append-only graph
        commit, which cannot carry a lifecycle change. This separate commit is
        idempotent: an action that already names its result is left unchanged.

        Without a finding, the same commit creates the notice record that
        becomes the result. It links to the action and to the receipt that
        parked or displaced the episode.
        """
        with self.ghost.lock:
            view = self.ghost.snapshot(capture=False)
            head = deliberate_inquiry(view, action_id)
            if head is None:
                raise RuntimeError("An admitted mental action must own its private inquiry.")
            outcome = mental_outcome(view, head)
            if outcome is None:
                return None
            action = view.get(action_id)
            lifecycle = MentalActionLifecycle.model_validate(action.metadata)
            if lifecycle.mental_result_id is not None:
                return ClosedMentalAction(result_record_id=lifecycle.mental_result_id, outcome=outcome)
            changes = self.ghost.begin_batch(f"mental-result:{action_id}", embed=False, cancel=self.stop_event)
            if outcome.finding_id is not None:
                result_id = outcome.finding_id
                changes.depend_on([result_id], authority=action_id)
            else:
                state = head.receipt.state
                origin = inquiry_origin(view, state)
                correlation = ActionMetadata.model_validate(action.metadata).dispatch_correlation
                notice = changes.create(Feature(content=unfinished_notice(origin.text, outcome),
                    feature_type=FeatureType.Thought, entity_id=state.grant.observer_entity_id,
                    scope=state.grant.scope, provenance=Provenance(source_kind=SourceKind.Self),
                    interlocus=InterlocusType.PrivateInternal,
                    lineage=RecordLineage(episode_id=state.episode_id, operation="mental_unfinished",
                        parent_record_ids=[action_id, head.record.id]),
                    metadata={MENTAL_ACTION_KEY: action_id, MENTAL_CORRELATION_KEY: correlation,
                              "mental_status": outcome.mental_status, "memory_kind": "interpretation"}))
                result_id = notice.id
                changes.depend_on([head.record.id], authority=action_id)
            changes.set_lifecycle(action_id, dispatch_status="admitted", mental_result_id=result_id,
                                  mental_status=outcome.mental_status, execution_status=outcome.execution_status)
            self.ghost.commit(changes)
            return ClosedMentalAction(result_record_id=result_id, outcome=outcome)

    def resume(self) -> None:
        """Publish every unappraised mental-action result that is not on the bus yet.

        Accepted main cycles acknowledge result IDs. A wait delays publication.
        The process remembers queued IDs; a restart rebuilds the set from
        memory. A graph finding is published only after its inner turn exists,
        which the private loop derives or repairs.
        """
        control = self.cognitive_control
        with self.ghost.lock:
            if control is not None and control.waiting():
                return
            appraised = {key for cycle in self.ghost.states for key in cycle.claimed_ids()}
            deliveries: list[ResultDelivery] = []
            for item in self.ghost.all_features:
                if item.metadata.get(MENTAL_ACTION_KEY) is None:
                    continue
                if item.id in appraised or item.id in self._published_results:
                    continue
                gwt_tick: int | None = None
                if item.metadata.get("private_inquiry_finding") is True:
                    inner_ticks = [turn.tick_id for turn in self.ghost.gwt_turns.values()
                                   if turn.slot_id == "inner" and item.id in turn.knoxel_ids]
                    if not inner_ticks:
                        continue
                    gwt_tick = inner_ticks[0]
                correlation = str(item.metadata[MENTAL_CORRELATION_KEY])
                deliveries.append(ResultDelivery(record_id=item.id, correlation=correlation, gwt_tick=gwt_tick))
        for delivery in deliveries:
            # Main can discard an event during delivery. Mark it first so
            # that immediate release cannot leave a stale publication claim.
            self._published_results.add(delivery.record_id)
            payload: JsonObject = {"knoxel_id": delivery.record_id}
            if delivery.gwt_tick is not None:
                payload["gwt_tick"] = delivery.gwt_tick
            self.bus.put(new_envelope(self.module_id, "data", "thought.created", delivery.correlation, payload))

    def release_results(self, envelopes: list[Envelope]) -> None:
        """Permit replay after main cognition discards or cancels queued work."""
        for envelope in envelopes:
            if envelope.topic == "thought.created":
                self._published_results.discard(envelope.payload.get("knoxel_id"))

    def _receipt(self, action_id: int, closed: ClosedMentalAction) -> None:
        """Publish diagnostic closing evidence. This receipt requests no speech."""
        with self.ghost.lock:
            action = self.ghost.snapshot(capture=False).get(action_id)
        correlation = ActionMetadata.model_validate(action.metadata).dispatch_correlation
        self.bus.put(new_envelope(self.module_id, "telemetry", "action.receipt", correlation,
            {"workflow_name": self.WORKFLOW_NAME, "status": closed.outcome.mental_status,
             "action_schema_id": action_id, "result_id": closed.result_record_id,
             "public_reply_requested": False}))


__all__ = [
    "DEFAULT_FIRST_OPERATION", "ClosedMentalAction", "FocusManagement", "MentalActionLifecycle",
    "MentalOutcome", "RequestedMentalAction", "ResultDelivery", "mental_outcome", "requested_action",
    "unfinished_notice",
]
