"""Discard executable leftovers when opening accepted checkpoint history.

Historical messages, actions, and receipts remain inspectable. Pending inboxes,
undelivered publications, and parked workers cannot become startup commands.
No external effect is reversed or retried. Raw checkpoint inspection bypasses
this operation. See pm.memory.store for checkpoint ownership.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from pm.runs.run_receipt import ManagedRunReceipt

if TYPE_CHECKING:
    from pm.memory.store import MemoryStore


class RestartLifecycle(BaseModel):
    """Operational fields retained on historical records."""

    model_config = ConfigDict(extra="ignore")
    exchange_status: str = ""
    publication_status: str = ""
    dispatch_status: str = ""
    execution_status: str = ""
    run_id: str = ""
    recovery_outcome: str = ""
    memory_action_id: int | None = None


@dataclass(frozen=True, slots=True)
class RestartSummary:
    """Counts of abandoned executable work, not deleted historical records."""

    records: int
    runs: int


def discard_unfinished_work(ghost: MemoryStore) -> RestartSummary:
    """Commit startup disposition once; reopening an already clean store is inert."""
    view = ghost.snapshot(capture=False)
    batch = ghost.begin_batch(f"restart-discard:{uuid4().hex}", embed=False)
    record_count = 0
    run_count = 0
    for record in view.all_knoxels.values():
        lifecycle = RestartLifecycle.model_validate(record.metadata)
        changed = False
        if (lifecycle.memory_action_id is not None and record.metadata.get("control_kind") == "inference_wait"
                and record.metadata["status"] == "waiting"):
            # The corresponding worker has no execution thread after reopening.
            batch.set_lifecycle(record.id, status="awake", wake_reason="memory worker discarded on restart")
            changed = True
        if lifecycle.exchange_status == "pending":
            batch.set_lifecycle(record.id, exchange_status="resolved", recovery_outcome="discarded_on_restart")
            changed = True
        if lifecycle.publication_status == "pending":
            batch.set_lifecycle(record.id, publication_status="discarded", recovery_outcome="discarded_on_restart")
            changed = True
        # Older checkpoints can already carry the restart cancellation on an
        # admitted action. Its own linked worker receipt is authoritative.
        if lifecycle.run_id and lifecycle.recovery_outcome == "discarded_on_restart" and lifecycle.execution_status == "cancelled":
            stored_run = ghost.managed_runs.get(lifecycle.run_id)
            if stored_run is not None:
                linked = ManagedRunReceipt.from_dict(stored_run)
                if linked.authority_record_id == record.id and linked.state in {"completed", "failed"}:
                    batch.set_lifecycle(record.id, dispatch_status="admitted", execution_status=linked.state,
                                        recovery_outcome="reconciled_from_run_receipt")
                    changed = True
        terminal = lifecycle.execution_status in {"completed", "failed", "cancelled"}
        dispatched = lifecycle.dispatch_status in {"pending", "admitted"}
        if not terminal and (dispatched or lifecycle.execution_status in {"pending", "running"}):
            batch.set_lifecycle(record.id, dispatch_status="cancelled", execution_status="cancelled",
                                recovery_outcome="discarded_on_restart")
            changed = True
        if changed:
            record_count += 1
    for run_id, row in ghost.managed_runs.items():
        receipt = ManagedRunReceipt.from_dict(row)
        unfinished = receipt.state in {"accepted", "running", "waiting_for_input", "interrupted"}
        owes_reply = receipt.arguments.get("_public_reply_requested") is True
        owes_event = any(not event["published"] for event in receipt.events)
        if not (unfinished or owes_reply or owes_event or receipt.parked_state):
            continue
        if unfinished:
            receipt.state = "cancelled"
            receipt.latest_event = "Unfinished work discarded on restart; external effects were not rolled back."
        receipt.arguments["_public_reply_requested"] = False
        receipt.parked_state = {}
        receipt.resume_phase = ""
        receipt.question_id = ""
        receipt.updates_open = False
        receipt.events = [{"message": event["message"], "published": True} for event in receipt.events]
        receipt.control_version += 1
        receipt.causal_commit_id = batch.commit_id
        batch.set_run_receipt(run_id, dict(receipt.to_dict()))
        run_count += 1
    if record_count or run_count:
        # The configured companion owns its operational inbox. Use its accepted
        # identity as authority so disposition survives later clean projection.
        companion_id = ghost.get_entity_id(ghost.get_companion_name())
        if companion_id is None:
            raise ValueError("Cannot discard pending work without the companion identity.")
        batch.depend_on([], authority=companion_id, manifest={"operation": "discard_on_restart"})
        ghost.commit(batch)
    else:
        batch.discard()
    return RestartSummary(records=record_count, runs=run_count)
