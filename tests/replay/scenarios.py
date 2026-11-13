"""Concrete situations the replay harness can record and replay.

Each scenario is a small, readable script over :class:`ScenarioRuntime`. It
states its own starting situation, drives the real modules in an explicit
order, and returns a typed outcome. The outcome is what a replay compares, so
it must contain the facts a reviewer would otherwise have to dig out of the
store.

Reading map: :data:`SCENARIOS` is the registry the CLI uses.
:class:`PrivateContinuityOutcome` and its helpers describe the private-loop
scenarios, which are the ones the beta continuity work needs.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from pm.runs.workflow_runner import WorkflowStatus

from tests.replay.runtime import ScenarioRuntime
from tests.replay.situations import PrivateConcernSituation, private_concern
from workflows.inner_thought.graph_execution import PrivateGraphResult
from workflows.inner_thought.graph_host import pending_findings
from workflows.inner_thought.inquiry import INQUIRY_EPISODE_KEY, inquiry_history


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
    )
}


def scenario_named(name: str) -> Scenario:
    """Return one scenario by name, or fail loudly listing the known names."""
    if name not in SCENARIOS:
        raise LookupError(f"Unknown scenario {name!r}. Known: {sorted(SCENARIOS)}")
    return SCENARIOS[name]


__all__ = ["ORDINARY_INPUT", "SCENARIOS", "CycleObservation", "PrivateContinuityOutcome",
           "Scenario", "ScenarioOutcome", "scenario_named"]
