"""Concrete situations the replay harness can record and replay.

Each scenario is a small, readable script over :class:`ScenarioRuntime`. It
states its own starting situation, drives the real modules in an explicit
order, and returns a typed outcome. The outcome is what a replay compares, so
it must contain the facts a reviewer would otherwise have to dig out of the
store.

Reading map: :data:`SCENARIOS` is the registry the CLI uses.
:class:`PrivateContinuityOutcome` and its helpers describe the private-loop
scenarios, which are the ones the beta continuity work needs.
:class:`DeliberateEntryOutcome` describes a selected mental action that takes
the private slot from spontaneous work.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from json import loads as load_json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pm.bus.envelopes import new_envelope
from pm.records.base import Provenance, RecordLineage
from pm.records.enums import ActionType, FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import ActionSchemaRecord
from pm.records.qualia_records import QualeRecord
from pm.runs.workflow_runner import WorkflowStatus

from tests.replay.runtime import ScenarioRuntime
from tests.replay.situations import (
    CONCERN_SOURCE, PERSPECTIVE_LINES, PerspectiveMorningSituation, PrivateConcernSituation,
    PrivateDwellingSituation, perspective_morning, private_concern, private_dwelling,
)
from workflows.cognitive_control.schemas import FocusProposal, MentalPlan
from workflows.entity_context.owner import EntityContextOwner
from workflows.entity_context.schemas import PerspectiveRevision
from workflows.focus_management.host import MentalActionLifecycle
from workflows.inner_thought.graph_execution import PrivateGraphResult
from workflows.inner_thought.graph_host import deliberate_inquiry, pending_findings
from workflows.inner_thought.inquiry import INQUIRY_EPISODE_KEY, InquiryNodeMetadata, inquiry_history
from workflows.inner_thought.nominations import NOMINATION_KEY, NominationMetadata
from workflows.main_cognition.entity_updates import prepare_perspective_revisions
from workflows.inner_thought.schemas import InnerRequest, ThoughtAttentionReceipt
from workflows.inner_thought.triggers import evaluate_triggers
from workflows.main_cognition.schemas import ActionArguments, ActionRequest, Stimulus


#: The ordinary message a scenario injects while private work is in flight.
#: It is unrelated to the fictional story, so it must not displace the concern.
ORDINARY_INPUT = (
    "Hey, unrelated thing: the kitchen tap is dripping again. I'll look at it "
    "tomorrow, no need to think about it now."
)


class ScenarioOutcome(BaseModel):
    """The comparable result of one deterministic scenario run.

    ``summary`` is the reviewer-facing one-line list the CLI prints per sample.
    Subclasses add the concrete identities and statuses that matter to them.
    """

    model_config = ConfigDict(extra="forbid")

    scenario: str = Field(description="Scenario name that produced this outcome.")
    summary: tuple[str, ...] = Field(description="Ordered human-readable facts, one per observation.")
    replay_blockers: tuple[str, ...] = Field(default=(),
        description="Reasons this run cannot be replayed, for example an aborted invocation.")


class CycleObservation(BaseModel):
    """What one private-thought timer tick actually did."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=1, description="One-based private cycle number in this session.")
    kernel_status: str = Field(description="Status of the parent kernel invocation.")
    summary: str = Field(description="The parent kernel's own report summary.")
    finding_id: int | None = Field(description="Accepted finding this cycle produced, if any.")
    episode_ids: tuple[str, ...] = Field(description="Every inquiry episode the store holds after this cycle.")
    node_ids: tuple[int, ...] = Field(description="Accepted inquiry node record ids after this cycle.")
    receipt_ids: tuple[int, ...] = Field(description="Accepted inquiry receipt record ids after this cycle.")
    provider_requests: int = Field(ge=0, description="Real provider requests made so far.")


class PrivateContinuityOutcome(ScenarioOutcome):
    """The private inquiry's fate across an injected ordinary message."""

    model_config = ConfigDict(extra="forbid")

    cycles: tuple[CycleObservation, ...] = Field(description="Each private cycle in order.")
    injected_input: str = Field(description="Exact text injected while private work was in flight.")
    triage_category: str | None = Field(description="Accepted early-triage category for the injected input.")
    finding_ids: tuple[int, ...] = Field(description="Accepted findings still awaiting main appraisal.")
    episode_status: str | None = Field(description="Status of the newest inquiry receipt's episode state.")
    inquiry_continued: bool = Field(description="True when the injected message left the inquiry advancing.")
    requests: int = Field(ge=0, description="Real provider requests this session made.")


def _episode_identities(runtime: ScenarioRuntime) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """Return the store's inquiry episodes, node ids, and receipt ids."""
    view = runtime.store.snapshot(capture=False)
    episode_ids = sorted({str(record.metadata[INQUIRY_EPISODE_KEY]) for record in view.all_knoxels.values()
                          if INQUIRY_EPISODE_KEY in record.metadata})
    node_ids: set[int] = set()
    receipt_ids: set[int] = set()
    for episode_id in episode_ids:
        for accepted in inquiry_history(view, episode_id):
            receipt_ids.add(accepted.record.id)
            node_ids.update(accepted.receipt.state.node_ids)
    return tuple(episode_ids), tuple(sorted(node_ids)), tuple(sorted(receipt_ids))


def _newest_episode_status(runtime: ScenarioRuntime) -> str | None:
    """Return the newest inquiry receipt's episode status, or None without one."""
    view = runtime.store.snapshot(capture=False)
    episode_ids = sorted({str(record.metadata[INQUIRY_EPISODE_KEY]) for record in view.all_knoxels.values()
                          if INQUIRY_EPISODE_KEY in record.metadata})
    if not episode_ids:
        return None
    newest = None
    for episode_id in episode_ids:
        history = inquiry_history(view, episode_id)
        if history:
            newest = history[-1]
    if newest is None:
        return None
    return str(newest.receipt.state.status)


def _observe_cycle(runtime: ScenarioRuntime, index: int) -> CycleObservation:
    """Read one private cycle from its kernel result and the store.

    A cycle that the control layer cancelled leaves no kernel result, which is
    itself the observation. It is reported as a status instead of raising, so a
    scenario about interruption can describe what happened.
    """
    result = runtime.private.last_result
    episode_ids, node_ids, receipt_ids = _episode_identities(runtime)
    if result is None:
        cancelled = runtime.private.run_stop.is_set()
        return CycleObservation(index=index,
                                kernel_status="cancelled" if cancelled else "no-result",
                                summary=("the invocation was cancelled before it produced a result"
                                         if cancelled else "the invocation produced no result"),
                                finding_id=None, episode_ids=episode_ids, node_ids=node_ids,
                                receipt_ids=receipt_ids, provider_requests=runtime.provider_requests())
    report = PrivateGraphResult.model_validate(result.report)
    return CycleObservation(index=index, kernel_status=str(result.status), summary=report.summary,
                            finding_id=report.finding_id, episode_ids=episode_ids, node_ids=node_ids,
                            receipt_ids=receipt_ids, provider_requests=runtime.provider_requests())


def _run_private_cycle(runtime: ScenarioRuntime, index: int, *, cancelled_is_expected: bool = False) -> CycleObservation:
    """Run one private timer tick and describe what it did."""
    runtime.private_cycle()
    observation = _observe_cycle(runtime, index)
    if observation.kernel_status == str(WorkflowStatus.Completed):
        return observation
    if cancelled_is_expected and observation.kernel_status == "cancelled":
        return observation
    raise RuntimeError(f"Private cycle {index} did not complete: {observation.kernel_status}: {observation.summary}")


class PrivateInvocationCounter:
    """Count model calls inside each private invocation.

    A scenario needs to inject outside input *while the private loop is
    thinking*, which is the situation that used to discard the work. Counting
    calls per invocation id puts the injection at the same logical place in a
    recording and in a replay, regardless of how many calls the model needs.
    """

    def __init__(self) -> None:
        self._calls: dict[str, int] = {}
        self._order: list[str] = []

    def note(self, invocation_id: str) -> int:
        """Record one model call and return how many this invocation has made."""
        if invocation_id not in self._calls:
            self._order.append(invocation_id)
        count = self._calls.get(invocation_id, 0) + 1
        self._calls[invocation_id] = count
        return count

    @property
    def invocations(self) -> tuple[str, ...]:
        """Return the private invocation ids in the order they first called a model."""
        return tuple(self._order)


def run_private_baseline(runtime: ScenarioRuntime) -> ScenarioOutcome:
    """Run three private cycles with no interruption at all.

    This is the control: it shows what the situation produces when nothing
    competes with it. A change that breaks the uninterrupted path is not a
    continuity improvement.
    """
    cycles = tuple(_run_private_cycle(runtime, index) for index in (1, 2, 3))
    findings = pending_findings(runtime.store)
    summary = tuple(f"cycle {item.index}: {item.summary}" for item in cycles)
    summary += (f"findings awaiting appraisal: {[record.id for record in findings]}",)
    return PrivateContinuityOutcome(
        scenario="private-baseline", summary=summary, cycles=cycles, replay_blockers=(), injected_input="",
        triage_category=None, finding_ids=tuple(record.id for record in findings),
        episode_status=_newest_episode_status(runtime),
        inquiry_continued=len(cycles[-1].node_ids) > len(cycles[0].node_ids),
        requests=runtime.provider_requests())


def run_private_continuity(runtime: ScenarioRuntime) -> ScenarioOutcome:
    """Inject one unrelated ordinary message while private work is in flight.

    Cycle one establishes the inquiry. Cycle two is interrupted by an ordinary
    message after its first model call. Cycles three and four then show whether
    the accepted inquiry survived, or whether the message discarded it.
    """
    counter = PrivateInvocationCounter()
    injected: list[str] = []

    def hook() -> None:
        """Count this call and, once, deliver the message mid-invocation.

        The message is committed and classified while the second private
        invocation is about to call the model, which is what happens in the
        running application: the triage module owns its own thread and the
        private loop is already inside its step. The harness states that
        interleaving explicitly instead of relying on thread timing.

        A continuing inquiry makes one model call per step, so the first call
        of the second invocation is the one that is in flight when the message
        arrives.
        """
        invocation = runtime.private.invocation_id or ""
        position = counter.note(invocation)
        if injected or len(counter.invocations) < 2:
            return
        if counter.invocations[1] == invocation and position == 1:
            runtime.submit(ORDINARY_INPUT)
            runtime.step_named("input_triage")
            injected.append(ORDINARY_INPUT)

    runtime.model.call_hook = hook
    cycles = [_run_private_cycle(runtime, 1), _run_private_cycle(runtime, 2, cancelled_is_expected=True)]
    cycles.append(_run_private_cycle(runtime, 3, cancelled_is_expected=True))
    runtime.model.call_hook = None

    source_ids = [record.id for record in runtime.store.all_features
                  if record.content == ORDINARY_INPUT]
    category = None
    if source_ids:
        assessment = runtime.control.triage_result(source_ids)
        category = None if assessment is None else assessment.items[0].category
    findings = pending_findings(runtime.store)
    continued = len(cycles[-1].node_ids) > len(cycles[1].node_ids)
    summary = tuple(f"cycle {item.index}: {item.summary}" for item in cycles)
    summary += (f"injected: {bool(injected)}", f"triage category: {category}",
                f"findings awaiting appraisal: {[record.id for record in findings]}",
                f"inquiry continued after the message: {continued}")
    # A cancelled invocation is a partial trace: the abort races the model call,
    # so the recorded call sequence cannot be reproduced. Say so instead of
    # letting a later replay fail as if the code had changed.
    blockers = tuple(f"cycle {item.index} was cancelled before it produced a result"
                     for item in cycles if item.kernel_status != str(WorkflowStatus.Completed))
    return PrivateContinuityOutcome(
        scenario="private-continuity", summary=summary, cycles=tuple(cycles), replay_blockers=blockers,
        injected_input=injected[0] if injected else "", triage_category=category,
        finding_ids=tuple(record.id for record in findings),
        episode_status=_newest_episode_status(runtime),
        inquiry_continued=continued, requests=runtime.provider_requests())


#: The mental task the deliberate scenario dispatches. It is synthetic input
#: standing in for a dispatch that main cognition accepted; the harness cannot
#: record a main cycle.
DELIBERATE_INSTRUCTION = (
    "Decide which of the two story endings I want to propose to Jesse first, and "
    "what I should keep in mind when we talk about the story tomorrow."
)

#: The deliberate instruction of the reflection scenario. It names the one
#: operation whose step reads recorded experiment and episode evidence.
DELIBERATE_REFLECTION_INSTRUCTION = (
    "Review what the last few private steps established about the story, and say what still needs "
    "checking before I propose one ending to Jesse."
)

#: The first inquiry operation of the deliberate scenario. Recall is the
#: default first operation, but it sends an embedding request, and the harness
#: does not record embedding requests. A replay must need no endpoint. The
#: operation is generative: a discriminatory first step has no accepted node to
#: compare, so it only spends one of the two stalled steps an episode has.
DELIBERATE_FIRST_OPERATION = "future_projection"

#: Private cycles the deliberate episode gets. The default episode limit is
#: six operations, so the episode has left the slot by then: with an outcome,
#: or parked at a bound.
DELIBERATE_CYCLE_LIMIT = 6


class DeliveredResult(BaseModel):
    """One ``thought.created`` envelope the session published."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Module that published the envelope.")
    correlation: str = Field(description="Correlation the result was delivered under.")
    record_id: int = Field(description="Delivered result record.")


class DeliberateEntryOutcome(ScenarioOutcome):
    """What happened when a selected mental action took the private slot."""

    model_config = ConfigDict(extra="forbid")

    cycles: tuple[CycleObservation, ...] = Field(description="Each private cycle in order.")
    action_record_id: int = Field(description="The dispatched mental action record.")
    parked_episode_id: str = Field(description="Spontaneous episode that held the slot when the action arrived.")
    parked_stop_reason: str = Field(description="Stop reason on the receipt that parked the spontaneous episode.")
    parked_by_action_id: int | None = Field(description="Action named by that receipt, or None without a preemption.")
    parked_nodes_kept: bool = Field(description="True when parking left the spontaneous graph and next operation unchanged.")
    deliberate_operations: tuple[str, ...] = Field(description="Operations the deliberate episode executed, in order.")
    deliberate_status: str = Field(description="Final status of the deliberate episode.")
    finding_id: int | None = Field(description="The deliberate episode's finding, if it reached one.")
    focus: FocusProposal | None = Field(description="Focus advice carried by the finding's proposal, if any.")
    mental_result_id: int | None = Field(description="Result identity committed on the action.")
    mental_status: str = Field(description="Mental status committed on the action; empty while it is open.")
    result_text: str = Field(description="Text of the action's result record: the finding, or the notice that the work stopped. Empty while open.")
    delivered: tuple[DeliveredResult, ...] = Field(description="Every published result envelope, in order.")
    private_finding_ids: tuple[int, ...] = Field(description="Findings the private loop itself still owes main appraisal.")
    parked_status_after: str = Field(description="Status of the spontaneous episode after one more private opportunity.")
    requests: int = Field(ge=0, description="Real provider requests this session made.")


def _dispatch_mental_action(runtime: ScenarioRuntime, *, instruction: str, operation: str,
                            correlation: str, workflow_name: str = "focus_management",
                            partner_entity_id: int | None = None,
                            source_ids: Sequence[int] | None = None) -> int:
    """Commit one accepted mental action and publish its request, as main cognition would.

    The record and the request are synthetic boundary input. Everything after
    the bus is the real deliberate owner, the real store, and the real private
    loop. ``workflow_name`` is the mental-workflow catalogue name; a
    perspective action names the entity it is about as the request's partner.
    Without explicit ``source_ids`` the situation must hold exactly one
    concern source, which the existing scenarios rely on.
    """
    if source_ids is None:
        source_ids = [record.id for record in runtime.store.all_features if record.content == CONCERN_SOURCE]
        if len(source_ids) != 1:
            raise LookupError("The situation must hold exactly one concern source.")
    view = runtime.store.snapshot(capture=False)
    source_texts = [str(view.get(source_id).content) for source_id in source_ids]
    request = ActionRequest(action_id="replay-mental-action", workflow_name=workflow_name,
        content="I want to settle this on my own before we talk again.",
        partner_entity_id=partner_entity_id,
        arguments=ActionArguments(instruction=instruction, operation=operation),
        stimuli=[Stimulus(knoxel_id=source_id, source="user_input", text=text)
                 for source_id, text in zip(source_ids, source_texts, strict=True)])
    batch = runtime.store.begin_batch("replay-mental-dispatch", embed=False)
    action = batch.create(ActionSchemaRecord(content=request.content, action_type=ActionType.Loopback,
        provenance=Provenance(source_kind=SourceKind.Workflow, source_ref="replay-synthetic-dispatch"),
        lineage=RecordLineage(episode_id="dispatch:replay-mental-action", operation="action_dispatch",
            parent_record_ids=list(source_ids)),
        metadata={"action_id": request.action_id, "workflow_name": workflow_name,
                  "dispatch_status": "pending", "dispatch_target": workflow_name,
                  "dispatch_correlation": correlation, "evidence_ids": list(source_ids)}))
    batch.preserve_on_recovery()
    runtime.store.commit(batch)
    payload = request.model_copy(update={"knoxel_id": action.id}).payload()
    runtime.private.bus.put(new_envelope("main_cog", "data", "action.request", correlation, payload,
                                         target=workflow_name))
    return action.id


def run_deliberate_entry(runtime: ScenarioRuntime) -> ScenarioOutcome:
    """Dispatch a mental action while a spontaneous inquiry holds the private slot.

    The first operation projects a future outcome. See
    :func:`run_deliberate_reflection` for the variant that reviews recorded
    evidence instead.
    """
    return _run_deliberate(runtime, name="deliberate-entry", instruction=DELIBERATE_INSTRUCTION,
                           operation=DELIBERATE_FIRST_OPERATION, correlation="replay-deliberate-entry")


def run_deliberate_reflection(runtime: ScenarioRuntime) -> ScenarioOutcome:
    """Dispatch a reflection review as the deliberate action's first operation.

    The step reads the scoped experiment and episode evidence the host captured
    under the observer's reality and frame. Nothing about the admission differs
    from :func:`run_deliberate_entry`; only the operation does.
    """
    return _run_deliberate(runtime, name="deliberate-reflection",
                           instruction=DELIBERATE_REFLECTION_INSTRUCTION,
                           operation="reflection_review", correlation="replay-deliberate-reflection")


def _run_deliberate(runtime: ScenarioRuntime, *, name: str, instruction: str, operation: str,
                    correlation: str) -> ScenarioOutcome:
    """Run one deliberate entry with the named first operation.

    Cycle one establishes the spontaneous inquiry. The action then arrives and
    the deliberate owner admits it. The following cycles execute the deliberate
    episode through the real thought and integrate stages until it leaves the
    slot. The owner then closes the action and delivers the finding. One more
    private cycle shows what shared attention does with the parked question.
    """
    cycles = [_run_private_cycle(runtime, 1)]
    view = runtime.store.snapshot(capture=False)
    episode_ids = sorted({str(record.metadata[INQUIRY_EPISODE_KEY]) for record in view.all_knoxels.values()
                          if INQUIRY_EPISODE_KEY in record.metadata})
    if len(episode_ids) != 1:
        raise RuntimeError(f"Cycle one must leave exactly one spontaneous episode, found {episode_ids}.")
    spontaneous_id = episode_ids[0]
    before = inquiry_history(view, spontaneous_id)[-1].receipt.state
    if before.status != "active":
        raise RuntimeError(f"The spontaneous inquiry must hold the slot before the action arrives: {before.status}.")

    action_id = _dispatch_mental_action(runtime, instruction=instruction, operation=operation,
                                         correlation=correlation)
    runtime.step_named("focus_management")
    view = runtime.store.snapshot(capture=False)
    parked = inquiry_history(view, spontaneous_id)[-1].receipt
    nodes_kept = (parked.state.node_ids, parked.state.next_operation, parked.state.sequence) == (
        before.node_ids, before.next_operation, before.sequence)
    if deliberate_inquiry(view, action_id) is None:
        raise RuntimeError("The deliberate owner did not admit the dispatched action.")

    for index in range(2, 2 + DELIBERATE_CYCLE_LIMIT):
        cycles.append(_run_private_cycle(runtime, index))
        head = deliberate_inquiry(runtime.store.snapshot(capture=False), action_id)
        assert head is not None  # Admitted above; an episode never disappears.
        if head.receipt.state.status != "active":
            break
    else:
        raise RuntimeError("The deliberate episode still held the slot after its operation limit.")
    # The deliberate owner now closes the action and delivers its result.
    runtime.step_named("focus_management")
    # One more opportunity: shared attention may now re-admit the parked question.
    cycles.append(_run_private_cycle(runtime, len(cycles) + 1))

    view = runtime.store.snapshot(capture=False)
    head = deliberate_inquiry(view, action_id)
    assert head is not None
    history = inquiry_history(view, head.receipt.state.episode_id)
    operations = tuple(item.receipt.step.operation for item in history if item.receipt.step is not None)
    lifecycle = MentalActionLifecycle.model_validate(view.get(action_id).metadata)
    finding_id: int | None = None
    focus: FocusProposal | None = None
    for item in history:
        if item.receipt.step is not None and item.receipt.step.outcome != "continue":
            finding_id = item.receipt.operation_record_id
    if finding_id is not None:
        focus = MentalPlan.model_validate(view.get(finding_id).metadata["mental_plan"]).focus
    delivered = tuple(DeliveredResult(source=item.source, correlation=item.correlation_id,
                                      record_id=int(str(item.payload["knoxel_id"])))
                      for item in runtime.envelopes("thought.created"))
    private_findings = tuple(record.id for record in pending_findings(runtime.store))
    parked_after = inquiry_history(view, spontaneous_id)[-1].receipt.state.status
    mental_status = lifecycle.mental_status
    result_text = ""
    if lifecycle.mental_result_id is not None:
        result_text = view.get(lifecycle.mental_result_id).content
    summary = tuple(f"cycle {item.index}: {item.summary}" for item in cycles)
    summary += (
        f"spontaneous episode parked by action: {parked.preemption is not None}",
        f"deliberate operations: {list(operations)}",
        f"deliberate status: {head.receipt.state.status}",
        f"finding: {finding_id}; focus advice: {focus is not None}",
        f"action result: {lifecycle.mental_result_id} ({mental_status or 'open'}): {result_text}",
        f"delivered: {[(item.source, item.record_id) for item in delivered]}",
        f"spontaneous episode afterwards: {parked_after}",
    )
    return DeliberateEntryOutcome(
        scenario=name, summary=summary, cycles=tuple(cycles), replay_blockers=(),
        action_record_id=action_id, parked_episode_id=spontaneous_id,
        parked_stop_reason=parked.state.stop_reason,
        parked_by_action_id=None if parked.preemption is None else parked.preemption.action_record_id,
        parked_nodes_kept=nodes_kept, deliberate_operations=operations,
        deliberate_status=str(head.receipt.state.status), finding_id=finding_id, focus=focus,
        mental_result_id=lifecycle.mental_result_id, mental_status=mental_status, result_text=result_text,
        delivered=delivered,
        private_finding_ids=private_findings, parked_status_after=str(parked_after),
        requests=runtime.provider_requests())


#: The ToM instruction of the morning-routine scenario (milestone 14.1). The
#: action asks one bounded question; nothing contacts Jesse.
PERSPECTIVE_INSTRUCTION = (
    "Understand when Jesse usually becomes available in the morning, and how uncertain that pattern is, "
    "so I can judge whether and when a greeting would be welcome."
)

#: First inquiry operation of the perspective scenario. Recall is the default
#: but sends an embedding request, and the harness does not record embedding
#: requests. Narrative stitching needs none and is generative, so a first step
#: has material to work from without a discrimination target.
PERSPECTIVE_FIRST_OPERATION = "narrative_stitching"

#: Maximum private opportunities the perspective episode may use. The shipped
#: episode limit is six operations, so the episode has left the slot.
PERSPECTIVE_CYCLE_LIMIT = 8


class RevisionClaimObservation(BaseModel):
    """One stored perspective-revision claim, exactly as the finding kept it."""

    model_config = ConfigDict(extra="forbid")

    claim_key: str = Field(description="Stable proposition key of the proposed entry.")
    statement: str = Field(description="The proposed attributed assertion.")
    epistemic_kind: str = Field(description="report, inference, or hypothetical; thinking cannot observe.")
    layer: str = Field(description="Entity-model layer the claim proposes.")
    confidence: float = Field(description="Proposed support for the attributed claim.")
    source_ids: tuple[int, ...] = Field(description="Original messages the claim cites.")


class ClaimDispositionObservation(BaseModel):
    """The real entity owner's answer to one proposed revision claim batch."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(description="Host-derived proposal identity of the staged batch.")
    disposition: str = Field(description="staged, deferred, or rejected.")
    reason: str = Field(description="accepted, stale_parent, changed_source, or unsupported_interpretation.")
    staged_receipt_id: int | None = Field(description="Receipt of a staged batch, else None.")


class PerspectiveInquiryOutcome(ScenarioOutcome):
    """What the morning-routine theory-of-mind episode produced.

    The scenario asserts no model answer. A sample without a finding or
    without a revision is a recorded quality result, kept for review.
    """

    model_config = ConfigDict(extra="forbid")

    cycles: tuple[CycleObservation, ...] = Field(description="Each private cycle in order.")
    action_record_id: int = Field(description="The dispatched theory-of-mind action record.")
    deliberate_operations: tuple[str, ...] = Field(description="Operations the episode executed, in order.")
    deliberate_status: str = Field(description="Final status of the deliberate episode.")
    finding_id: int | None = Field(description="The episode's finding, if it reached one.")
    mental_result_id: int | None = Field(description="Result identity committed on the action.")
    mental_status: str = Field(description="Mental status committed on the action; empty while it is open.")
    result_text: str = Field(description="Text of the finding or of the unfinished-work notice.")
    thought_requests: int = Field(ge=0, description="Thought-stage model requests the episode made.")
    thought_requests_with_perspective: int = Field(
        ge=0, description="Those thought requests whose instruction carried the perspective block.")
    revision_claims: tuple[RevisionClaimObservation, ...] = Field(
        description="Stored revision claims of the finding, if any.")
    has_observation_claim: bool = Field(description="True when any stored claim is an observation (must stay False).")
    dispositions: tuple[ClaimDispositionObservation, ...] = Field(
        description="The real owner's disposition for each proposed revision batch on the final store.")
    action_requests: int = Field(ge=0, description="action.request envelopes the session published.")
    public_utterances: tuple[str, ...] = Field(description="Public speech the session produced; a ToM action sends none.")
    requests: int = Field(ge=0, description="Real provider requests this session made.")


class NominationObservation(BaseModel):
    """One retained private nomination, with the evidence it cited."""

    model_config = ConfigDict(extra="forbid")

    record_id: int = Field(description="Record id of the retained nomination.")
    text: str = Field(description="The nomination's own text.")
    question: str = Field(description="The question the nomination proposed.")
    suggested_operation: str = Field(description="Operation the nomination suggested.")
    evidence_ids: tuple[int, ...] = Field(description="Records the nomination cited.")
    cites_private_thought: bool = Field(
        description="True when the nomination cites an earlier private thought or graph operation.")


class StepObservation(BaseModel):
    """One accepted inquiry operation, read from its own record."""

    model_config = ConfigDict(extra="forbid")

    record_id: int = Field(description="Record id of the executed operation.")
    operation: str = Field(description="Operation name the graph admitted.")
    outcome: str = Field(description="The step's own stated outcome.")
    progress_kind: str = Field(description="Accepted progress: new evidence, discrimination, or none.")
    evidence_ids: tuple[int, ...] = Field(description="Original sources the step cited.")
    reason: str = Field(description="Why the step continued or stopped.")
    information_gap: str = Field(description="Evidence the step says is missing; empty when it states none.")


class PrivateDwellingOutcome(ScenarioOutcome):
    """What one private loop did with a short ambiguous reply.

    The scenario asserts no model answer. The typed facts below are what the
    store actually holds after the cycles, so a model that declines to dwell is
    a recorded quality result, not a failed assertion.
    """

    model_config = ConfigDict(extra="forbid")

    cycles: tuple[CycleObservation, ...] = Field(description="Each private cycle in order.")
    nominations: tuple[NominationObservation, ...] = Field(description="Every retained nomination, in record order.")
    selected_nomination_id: int | None = Field(description="Nomination shared attention selected, if any.")
    steps: tuple[StepObservation, ...] = Field(description="Every accepted graph operation, in record order.")
    stop_reason: str = Field(description="Stop reason on the newest inquiry receipt.")
    repeated_concern: bool = Field(
        description="True when the real trigger policy reports repeated_concern over the retained thoughts at the end.")
    self_observation_nominated: bool = Field(
        description="True when any nomination cited an earlier private thought or graph operation.")
    quale_ids: tuple[int, ...] = Field(description="Felt-state records committed during the run.")
    requests: int = Field(ge=0, description="Real provider requests this session made.")


def _private_thought_ids(runtime: ScenarioRuntime) -> set[int]:
    """Return every committed private thought or graph operation id."""
    return {int(record.id) for record in runtime.store.all_features
            if record.feature_type == FeatureType.InnerMonologue
            and (record.metadata or {}).get("loop") in {"inner", "inquiry"}}


def _nomination_observations(runtime: ScenarioRuntime) -> tuple[NominationObservation, ...]:
    """Read every retained nomination and whether it cited private thinking."""
    private_ids = _private_thought_ids(runtime)
    rows: list[NominationObservation] = []
    for record in runtime.store.all_features:
        if NOMINATION_KEY not in record.metadata:
            continue
        receipt = NominationMetadata.model_validate(record.metadata).thought_nomination
        nomination = receipt.nomination
        rows.append(NominationObservation(
            record_id=int(record.id), text=nomination.text, question=nomination.question,
            suggested_operation=nomination.suggested_operation, evidence_ids=nomination.evidence_ids,
            cites_private_thought=bool(private_ids.intersection(nomination.evidence_ids))))
    rows.sort(key=lambda item: item.record_id)
    return tuple(rows)


def _selected_nomination(runtime: ScenarioRuntime) -> int | None:
    """Return the nomination shared attention selected, or None."""
    selected: int | None = None
    for record in runtime.store.all_features:
        raw = record.metadata.get("thought_attention")
        if raw is None:
            continue
        receipt = ThoughtAttentionReceipt.model_validate(raw)
        if receipt.decision.disposition == "select":
            selected = receipt.decision.nomination_id
    return selected


def _step_observations(runtime: ScenarioRuntime) -> tuple[StepObservation, ...]:
    """Read every accepted inquiry operation from its own record."""
    rows: list[StepObservation] = []
    for record in runtime.store.all_features:
        if record.feature_type != FeatureType.InnerMonologue:
            continue
        if (record.metadata or {}).get("loop") != "inquiry":
            continue
        step = InquiryNodeMetadata.model_validate(record.metadata).inquiry_step
        rows.append(StepObservation(record_id=int(record.id), operation=step.operation,
            outcome=step.outcome, progress_kind=step.progress.kind,
            evidence_ids=step.evidence_ids, reason=step.reason,
            information_gap=step.information_gap))
    rows.sort(key=lambda item: item.record_id)
    return tuple(rows)


def _newest_stop_reason(runtime: ScenarioRuntime) -> str:
    """Return the newest inquiry receipt's stop reason, or an empty string."""
    view = runtime.store.snapshot(capture=False)
    episode_ids = sorted({str(record.metadata[INQUIRY_EPISODE_KEY]) for record in view.all_knoxels.values()
                          if INQUIRY_EPISODE_KEY in record.metadata})
    newest = None
    for episode_id in episode_ids:
        history = inquiry_history(view, episode_id)
        if history:
            newest = history[-1]
    return "" if newest is None else str(newest.receipt.state.stop_reason)


def _repeated_concern(runtime: ScenarioRuntime) -> bool:
    """Recompute the real trigger policy over the thoughts the store retains now.

    The trigger candidates of an earlier request are not persisted, so the
    outcome reports the same policy's answer at the end rather than claiming
    which request fired it.
    """
    request = InnerRequest(recent_thoughts=runtime.private._recent_private_thoughts())
    return any(candidate.name == "repeated_concern" for candidate in evaluate_triggers(request))


#: How many private opportunities the dwelling scenario may use before it stops
#: waiting for the episode to leave the slot. The shipped step bound is six.
DWELLING_CYCLE_LIMIT = 8


def run_private_dwelling(runtime: ScenarioRuntime) -> ScenarioOutcome:
    """Run private cycles over one short ambiguous reply until the episode stops.

    Two more opportunities follow, so a parked question can be offered again.
    The outcome records what the store holds; it asserts no model answer.
    """
    cycles: list[CycleObservation] = []
    for index in range(1, DWELLING_CYCLE_LIMIT + 1):
        cycles.append(_run_private_cycle(runtime, index))
        status = _newest_episode_status(runtime)
        if status is not None and status != "active":
            break
    for _ in range(2):
        cycles.append(_run_private_cycle(runtime, len(cycles) + 1))
    nominations = _nomination_observations(runtime)
    steps = _step_observations(runtime)
    selected = _selected_nomination(runtime)
    stop_reason = _newest_stop_reason(runtime)
    repeated = _repeated_concern(runtime)
    self_observation = any(item.cites_private_thought for item in nominations)
    qualia = tuple(int(record.id) for record in runtime.store.all_knoxels.values()
                   if isinstance(record, QualeRecord))
    summary = tuple(f"cycle {item.index}: {item.summary}" for item in cycles)
    summary += (
        f"nominations: {[(item.record_id, item.suggested_operation) for item in nominations]}",
        f"selected: {selected}",
        f"steps: {[(item.operation, item.outcome, item.progress_kind) for item in steps]}",
        f"stop reason: {stop_reason}",
        f"repeated_concern: {repeated}",
        f"nominations citing private thinking: {self_observation}",
        f"felt-state records: {list(qualia)}",
    )
    return PrivateDwellingOutcome(
        scenario="private-dwelling", summary=summary, replay_blockers=(), cycles=tuple(cycles),
        nominations=nominations, selected_nomination_id=selected, steps=steps,
        stop_reason=stop_reason, repeated_concern=repeated,
        self_observation_nominated=self_observation, quale_ids=qualia,
        requests=runtime.provider_requests())


def _thought_evidence_requests(runtime: ScenarioRuntime) -> tuple[int, int]:
    """Count thought-stage calls and those whose instruction carried the perspective block.

    The rule block is rendered into the thought instruction before the call,
    so its first sentence is observable in the recorded request. The recording
    flushes every call as it happens, so the journal is current during the run.
    """
    journal = runtime.directory / "calls.jsonl"
    thought_calls = 0
    with_perspective = 0
    if not journal.exists():
        return (0, 0)
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        request = str(load_json(line).get("request", ""))
        if "Return one InquiryStep" not in request:
            continue
        thought_calls += 1
        if "This inquiry models" in request:
            with_perspective += 1
    return thought_calls, with_perspective


def _perspective_dispositions(runtime: ScenarioRuntime,
                              finding_id: int | None) -> tuple[ClaimDispositionObservation, ...]:
    """Ask the real entity owner how it would stage the finding's proposed revision.

    This is a rehearsal on the final store, in one batch that is never
    committed: main cognition performs the real staging at its own final
    commit boundary (decision D2 in the B06 plan). The outcome stays factual.
    """
    if finding_id is None:
        return ()
    view = runtime.store.snapshot(capture=False)
    record = view.get_knoxel_by_id(finding_id)
    if record is None or "perspective_revision" not in record.metadata:
        return ()
    companion_id = runtime.store.get_entity_id(runtime.store.get_companion_name())
    if companion_id is None:
        raise ValueError("The scenario store must hold the companion identity.")
    owner = EntityContextOwner(runtime.store)
    updates = prepare_perspective_revisions(owner, view, [finding_id], observer_entity_id=companion_id)
    rows: list[ClaimDispositionObservation] = []
    if updates.proposals:
        with runtime.store.lock:
            batch = runtime.store.begin_batch("perspective-rehearsal", embed=False)
            for proposal in updates.proposals:
                outcome = owner.stage_update(batch, proposal)
                rows.append(ClaimDispositionObservation(proposal_id=proposal.proposal_id,
                    disposition=outcome.disposition, reason=outcome.reason,
                    staged_receipt_id=outcome.staged_receipt_id))
    rows.extend(ClaimDispositionObservation(proposal_id="", disposition="rejected",
        reason=str(item["reason"]), staged_receipt_id=None) for item in updates.rejected)
    return tuple(rows)


def run_perspective_morning(runtime: ScenarioRuntime) -> ScenarioOutcome:
    """Dispatch one theory-of-mind action and execute its deliberate episode.

    The action is the only envelope this session publishes; Jesse is the
    partner, and the situation's synthetic morning lines are the action's
    evidence. The private loop executes the episode through the real thought
    and integrate stages; the deliberate owner closes and delivers the result.
    """
    expected: set[str] = {text for text, _ in PERSPECTIVE_LINES}
    source_ids = [record.id for record in runtime.store.all_features
                  if record.content in expected and record.interlocus == InterlocusType.Public]
    if len(source_ids) != len(PERSPECTIVE_LINES):
        raise LookupError("The situation must hold every synthetic morning line.")
    user_id = runtime.store.get_entity_id(runtime.store.get_user_name())
    if user_id is None:
        raise ValueError("The scenario store must hold the user identity.")
    action_id = _dispatch_mental_action(runtime, instruction=PERSPECTIVE_INSTRUCTION,
        operation=PERSPECTIVE_FIRST_OPERATION, correlation="replay-perspective-morning",
        workflow_name="theory_of_mind", partner_entity_id=user_id, source_ids=source_ids)
    runtime.step_named("focus_management")
    view = runtime.store.snapshot(capture=False)
    if deliberate_inquiry(view, action_id) is None:
        raise RuntimeError("The deliberate owner did not admit the perspective action.")

    cycles: list[CycleObservation] = []
    for index in range(1, PERSPECTIVE_CYCLE_LIMIT + 1):
        cycles.append(_run_private_cycle(runtime, index))
        head = deliberate_inquiry(runtime.store.snapshot(capture=False), action_id)
        assert head is not None  # Admitted above; an episode never disappears.
        if head.receipt.state.status != "active":
            break
    else:
        raise RuntimeError("The perspective episode still held the slot after its cycle limit.")
    # The deliberate owner closes the action and delivers its result.
    runtime.step_named("focus_management")

    view = runtime.store.snapshot(capture=False)
    head = deliberate_inquiry(view, action_id)
    assert head is not None
    history = inquiry_history(view, head.receipt.state.episode_id)
    operations = tuple(item.receipt.step.operation for item in history if item.receipt.step is not None)
    finding_id: int | None = next((item.receipt.operation_record_id for item in history
                                   if item.receipt.step is not None and item.receipt.step.outcome != "continue"),
                                  None)
    lifecycle = MentalActionLifecycle.model_validate(view.get(action_id).metadata)
    result_text = "" if lifecycle.mental_result_id is None else view.get(lifecycle.mental_result_id).content
    revision_claims: list[RevisionClaimObservation] = []
    has_observation = False
    if finding_id is not None:
        raw = view.get(finding_id).metadata.get("perspective_revision")
        if raw is not None:
            revision = PerspectiveRevision.model_validate(raw)
            for claim in revision.claims:
                revision_claims.append(RevisionClaimObservation(claim_key=claim.claim_key,
                    statement=claim.statement, epistemic_kind=claim.epistemic_kind, layer=claim.layer,
                    confidence=claim.confidence, source_ids=tuple(claim.source_ids)))
                has_observation = has_observation or claim.epistemic_kind == "observation"
    dispositions = _perspective_dispositions(runtime, finding_id)
    thought_calls, thought_with_perspective = _thought_evidence_requests(runtime)
    action_requests = len(runtime.envelopes("action.request"))
    public_utterances = tuple(str(item.text) for item in runtime.replies)
    summary = tuple(f"cycle {item.index}: {item.summary}" for item in cycles)
    summary += (
        f"deliberate operations: {list(operations)}",
        f"deliberate status: {head.receipt.state.status}",
        f"finding: {finding_id}",
        f"action result: {lifecycle.mental_result_id} ({lifecycle.mental_status or 'open'}): {result_text}",
        f"thought requests: {thought_calls}; with perspective evidence: {thought_with_perspective}",
        f"revision claims: {[(item.claim_key, item.epistemic_kind, item.source_ids) for item in revision_claims]}",
        f"observation claim: {has_observation}",
        f"owner dispositions: {[(item.disposition, item.reason) for item in dispositions]}",
        f"action requests: {action_requests}; public utterances: {list(public_utterances)}",
    )
    return PerspectiveInquiryOutcome(
        scenario="perspective-morning", summary=summary, replay_blockers=(), cycles=tuple(cycles),
        action_record_id=action_id, deliberate_operations=operations,
        deliberate_status=str(head.receipt.state.status), finding_id=finding_id,
        mental_result_id=lifecycle.mental_result_id, mental_status=lifecycle.mental_status,
        result_text=result_text, thought_requests=thought_calls,
        thought_requests_with_perspective=thought_with_perspective,
        revision_claims=tuple(revision_claims), has_observation_claim=has_observation,
        dispositions=tuple(dispositions), action_requests=action_requests,
        public_utterances=public_utterances, requests=runtime.provider_requests())


@dataclass(frozen=True, slots=True)
class Scenario:
    """One replayable script over the real modules.

    ``outcome_type`` names the typed outcome the script returns, so a recorded
    outcome reloads into its own shape instead of being flattened.
    """

    name: str
    description: str
    situation: Callable[[datetime, Path], PrivateConcernSituation | PrivateDwellingSituation]
    run: Callable[[ScenarioRuntime], ScenarioOutcome]
    outcome_type: type[ScenarioOutcome]


#: Every scenario the CLI can record or replay, keyed by name.
SCENARIOS: dict[str, Scenario] = {
    scenario.name: scenario for scenario in (
        Scenario(name="private-baseline",
                 description="Three uninterrupted private cycles over the synthetic story concern.",
                 situation=private_concern, run=run_private_baseline,
                 outcome_type=PrivateContinuityOutcome),
        Scenario(name="private-continuity",
                 description="One unrelated ordinary message injected while the private inquiry is in flight.",
                 situation=private_concern, run=run_private_continuity,
                 outcome_type=PrivateContinuityOutcome),
        Scenario(name="deliberate-entry",
                 description="A selected mental action takes the private slot from a spontaneous inquiry.",
                 situation=private_concern, run=run_deliberate_entry,
                 outcome_type=DeliberateEntryOutcome),
        Scenario(name="deliberate-reflection",
                 description="A deliberate reflection review runs in the private slot with scoped experiment and episode evidence.",
                 situation=private_concern, run=run_deliberate_reflection,
                 outcome_type=DeliberateEntryOutcome),
        Scenario(name="private-dwelling",
                 description="Private cycles over one short ambiguous reply, from a committed non-neutral mental state.",
                 situation=private_dwelling, run=run_private_dwelling,
                 outcome_type=PrivateDwellingOutcome),
        Scenario(name="perspective-morning",
                 description="A theory-of-mind action works out when Jesse usually becomes available in the morning.",
                 situation=perspective_morning, run=run_perspective_morning,
                 outcome_type=PerspectiveInquiryOutcome),
    )
}


def scenario_named(name: str) -> Scenario:
    """Return one scenario by name, or fail loudly listing the known names."""
    if name not in SCENARIOS:
        raise LookupError(f"Unknown scenario {name!r}. Known: {sorted(SCENARIOS)}")
    return SCENARIOS[name]


__all__ = ["DELIBERATE_INSTRUCTION", "ORDINARY_INPUT", "SCENARIOS", "CycleObservation",
           "DeliberateEntryOutcome", "DeliveredResult", "NominationObservation",
           "PerspectiveInquiryOutcome", "PrivateContinuityOutcome", "PrivateDwellingOutcome", "Scenario",
           "ScenarioOutcome", "StepObservation", "scenario_named"]
