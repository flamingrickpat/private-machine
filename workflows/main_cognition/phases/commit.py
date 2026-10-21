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
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import write_report


#: Action types that put words in front of the partner.
SPOKEN_ACTIONS = frozenset({"Reply", "ToolCallAndReply"})


def wait_request(work: CycleWork) -> WaitRequest | None:
    """Return the wait that ends this cycle, or None to run the next cycle at once.

    A Sleep route wins. Its duration comes from the route, or from the
    configured default when the route gives none, and its reason is the
    decision text.

    Otherwise a spoken reply arms the quiet window. The companion has just
    said something, so the next thing worth reading is her partner's answer,
    not another of her own thoughts. The window is the operator's
    ``post_reply_seconds``; zero disables it. A cycle that only spoke, and a
    cycle that spoke and started a managed run, both wait: a managed run is
    not suspended by a wait, so the window gives the worker time to finish.
    """
    for item in work.routes:
        if item.route.action_type == "Sleep":
            seconds = item.route.arguments.timeout_seconds
            return WaitRequest(seconds=float(seconds if seconds is not None else work.request.wait_default_seconds),
                               reason=item.decision or "Wait for a new stimulus.")
    if work.request.post_reply_seconds > 0 and any(
            item.status == "dispatched" and item.action_type in SPOKEN_ACTIONS for item in work.dispatches):
        return WaitRequest(seconds=float(work.request.post_reply_seconds),
                           reason="I said something. Wait for an answer before I think again.")
    return None


def run(context, work: CycleWork) -> CycleWork:
    work.wait_request = wait_request(work)
    released = sum(1 for item in work.dispatches if item.status == "dispatched")
    work.receipt("commit", "handed the cycle to the host")
    work.summary = f"ran {len(work.phases)} phases, dispatched {released} actions"
    write_report(context.runtime.environment, work)
    return work


__all__ = ["run", "wait_request", "SPOKEN_ACTIONS"]
