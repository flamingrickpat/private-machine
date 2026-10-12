"""Replay the exact accepted dispatch after a commit/publication interruption."""
from __future__ import annotations

from pm.bus.envelopes import Envelope, _new_id
from pm.records.memory_records import ActionSchemaRecord
from runtime.publication import exchange_record


def acknowledge_intent(ghost, action_id: int) -> None:
    """Mark durable admission without changing the selected action or its ID."""
    with ghost.lock:
        if action_id not in ghost.revisions:
            return
        action = ghost.snapshot({action_id}).get(action_id)
        if not isinstance(action, ActionSchemaRecord) or action.metadata.get("dispatch_status") != "pending":
            return
        batch = ghost.begin_batch(f"intent-ack:{action_id}")
        action.metadata["dispatch_status"] = "admitted"
        batch.replace(action)
        ghost.commit(batch)


def replay_intents(ghost, bus, supervisor) -> set[str]:
    """Replay frozen envelopes, or reconcile them with existing run receipts.

    An existing body receipt proves admission, even after interruption. Replay
    never resumes its worker. Speech uses the original exchange identity.
    """
    handled = set()
    receipts = supervisor.receipts()
    for item in list(ghost.all_knoxels.values()):
        if not isinstance(item, ActionSchemaRecord) or item.metadata.get("dispatch_status") != "pending":
            continue
        metadata = item.metadata
        correlation = metadata["dispatch_correlation"]
        handled.add(correlation)
        exchange = exchange_record(ghost, correlation)
        admitted = any(row.get("action_id") == metadata.get("action_id") for row in receipts.values())
        if admitted or (exchange is not None and exchange.metadata.get("exchange_status") == "resolved"):
            acknowledge_intent(ghost, item.id)
            continue
        bus.put(Envelope(id=_new_id(), source="intent_recovery", kind="data", topic="action.request",
                         correlation_id=correlation, target=metadata["dispatch_target"],
                         payload={**metadata["dispatch_payload"], "knoxel_id": item.id, "recovery_outcome": "intent_replayed"}))
    return handled
