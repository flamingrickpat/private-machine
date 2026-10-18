"""Commit phase: finish the cycle and hand it to the host.

The phase finishes the work that depends on all earlier phases and writes the
final ``CycleWork`` to ``outbox/report.yaml``. It does not write the memory
store. The host validates the report and stages one write batch with the
records, the cycle state, the turn record, and the dispatches. The host then
publishes the bus events.

The store write stays in the host for two reasons. The host owns the store,
the bus, and cognitive control. And the workflow module publishes only after
the kernel reports success and the run was not cancelled, so a cancelled run
never writes a partial cycle.
"""
from __future__ import annotations

from workflows.cognitive_control.schemas import WaitRequest
from workflows.main_cognition.affect import REGULATION_REACTIVE
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import write_report


def apply_reactive_override(work: CycleWork) -> None:
    """Make the frame reactive when the regulation mode of the cycle is reactive.

    Game danger preempts slow regulation. The experience phase set the
    frame fields first. This override changes only the committed attention
    frame.
    """
    if work.affect.regulation != REGULATION_REACTIVE:
        return
    frame = work.experience
    frame.control_mode = "reactive"
    reason = frame.frame_reason.strip()
    frame.frame_reason = (f"{reason} Danger preempts contemplation.".strip()
                          if "preempt" not in reason.lower() else reason)


def wait_request(work: CycleWork) -> WaitRequest | None:
    """Return the wait of the first decision that routing mapped to Sleep, or None.

    A Sleep route without a duration waits for the configured default. The
    reason of the wait is the decision text.
    """
    for item in work.routes:
        if item.route.action_type == "Sleep":
            seconds = item.route.arguments.timeout_seconds
            return WaitRequest(seconds=float(seconds if seconds is not None else work.request.wait_default_seconds),
                               reason=item.decision or "Wait for a new stimulus.")
    return None


def run(context, work: CycleWork) -> CycleWork:
    apply_reactive_override(work)
    work.wait_request = wait_request(work)
    released = sum(1 for item in work.dispatches if item.status == "dispatched")
    work.receipt("commit", "handed the cycle to the host")
    work.summary = f"ran {len(work.phases)} phases, dispatched {released} actions"
    write_report(context.runtime.environment, work)
    return work


__all__ = ["apply_reactive_override", "run", "wait_request"]
