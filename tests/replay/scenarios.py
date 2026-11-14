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

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pm.bus.envelopes import new_envelope
from pm.records.base import Provenance, RecordLineage
from pm.records.enums import ActionType, SourceKind
from pm.records.memory_records import ActionSchemaRecord
from pm.runs.workflow_runner import WorkflowStatus

from tests.replay.runtime import ScenarioRuntime
from tests.replay.situations import CONCERN_SOURCE, PrivateConcernSituation, private_concern
from workflows.cognitive_control.schemas import FocusProposal, MentalPlan
from workflows.focus_management.host import MentalActionLifecycle
from workflows.inner_thought.graph_execution import PrivateGraphResult
from workflows.inner_thought.graph_host import deliberate_inquiry, pending_findings
from workflows.inner_thought.inquiry import INQUIRY_EPISODE_KEY, inquiry_history
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


def _dispatch_mental_action(runtime: ScenarioRuntime) -> int:
    """Commit one accepted mental action and publish its request, as main cognition would.

    The record and the request are synthetic boundary input. Everything after
    the bus is the real deliberate owner, the real store, and the real private
    loop.
    """
    sources = [record.id for record in runtime.store.all_features if record.content == CONCERN_SOURCE]
    if len(sources) != 1:
        raise LookupError("The situation must hold exactly one concern source.")
    correlation = "replay-deliberate-entry"
    request = ActionRequest(action_id="replay-mental-action", workflow_name="focus_management",
        content="I want to settle this on my own before we talk again.",
        arguments=ActionArguments(instruction=DELIBERATE_INSTRUCTION, operation=DELIBERATE_FIRST_OPERATION),
        stimuli=[Stimulus(knoxel_id=sources[0], source="user_input", text=CONCERN_SOURCE)])
    batch = runtime.store.begin_batch("replay-mental-dispatch", embed=False)
    action = batch.create(ActionSchemaRecord(content=request.content, action_type=ActionType.Loopback,
        provenance=Provenance(source_kind=SourceKind.Workflow, source_ref="replay-synthetic-dispatch"),
        lineage=RecordLineage(episode_id="dispatch:replay-mental-action", operation="action_dispatch",
            parent_record_ids=[sources[0]]),
        metadata={"action_id": request.action_id, "workflow_name": "focus_management",
                  "dispatch_status": "pending", "dispatch_target": "focus_management",
                  "dispatch_correlation": correlation, "evidence_ids": [sources[0]]}))
    batch.preserve_on_recovery()
    runtime.store.commit(batch)
    payload = request.model_copy(update={"knoxel_id": action.id}).payload()
    runtime.private.bus.put(new_envelope("main_cog", "data", "action.request", correlation, payload,
                                         target="focus_management"))
    return action.id


def run_deliberate_entry(runtime: ScenarioRuntime) -> ScenarioOutcome:
    """Dispatch a mental action while a spontaneous inquiry holds the private slot.

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

    action_id = _dispatch_mental_action(runtime)
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
        scenario="deliberate-entry", summary=summary, cycles=tuple(cycles), replay_blockers=(),
        action_record_id=action_id, parked_episode_id=spontaneous_id,
        parked_stop_reason=parked.state.stop_reason,
        parked_by_action_id=None if parked.preemption is None else parked.preemption.action_record_id,
        parked_nodes_kept=nodes_kept, deliberate_operations=operations,
        deliberate_status=str(head.receipt.state.status), finding_id=finding_id, focus=focus,
        mental_result_id=lifecycle.mental_result_id, mental_status=mental_status, result_text=result_text,
        delivered=delivered,
        private_finding_ids=private_findings, parked_status_after=str(parked_after),
        requests=runtime.provider_requests())


@dataclass(frozen=True, slots=True)
class Scenario:
    """One replayable script over the real modules.

    ``outcome_type`` names the typed outcome the script returns, so a recorded
    outcome reloads into its own shape instead of being flattened.
    """

    name: str
    description: str
    situation: Callable[[datetime, Path], PrivateConcernSituation]
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
    )
}


def scenario_named(name: str) -> Scenario:
    """Return one scenario by name, or fail loudly listing the known names."""
    if name not in SCENARIOS:
        raise LookupError(f"Unknown scenario {name!r}. Known: {sorted(SCENARIOS)}")
    return SCENARIOS[name]


__all__ = ["DELIBERATE_INSTRUCTION", "ORDINARY_INPUT", "SCENARIOS", "CycleObservation",
           "DeliberateEntryOutcome", "DeliveredResult", "PrivateContinuityOutcome",
           "Scenario", "ScenarioOutcome", "scenario_named"]
