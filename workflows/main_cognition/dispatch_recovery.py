"""Replay the exact accepted dispatch after a commit/publication interruption.

Main cognition stores each released action with ``dispatch_status:
pending`` and the published :class:`~workflows.main_cognition.schemas.ActionRequest`
in its metadata. A workflow that admits the action sets the status to
``admitted``. After a restart, a pending action is published again with the
same request, unless a run receipt or a resolved exchange proves that it was
already admitted.
"""
from __future__ import annotations

from pm.bus.envelopes import Envelope, new_id
from pm.records.memory_records import ActionSchemaRecord
from workflows.main_cognition.schemas import ActionMetadata, PendingDispatch, RecordMetadataView
from workflows.public_speech.publication import exchange_record


def acknowledge_intent(ghost, action_id: int) -> None:
    """Mark durable admission without changing the selected action or its ID.

    An unknown id, a record that is not an action, and an action that is not
    pending change nothing: an admission can arrive twice.
    """
    with ghost.lock:
        if action_id not in ghost.revisions:
            return
        action = ghost.snapshot({action_id}).get(action_id).copy_for_edit()
        if (not isinstance(action, ActionSchemaRecord)
                or ActionMetadata.model_validate(action.metadata).dispatch_status != "pending"):
            return
        batch = ghost.begin_batch(f"intent-ack:{action_id}")
        action.metadata["dispatch_status"] = "admitted"
        batch.replace(action)
        ghost.commit(batch)


def replay_intents(ghost, bus, supervisor) -> set[str]:
    """Replay frozen requests, or reconcile them with existing run receipts.

    An existing body receipt proves admission, even after interruption. Replay
    never resumes its worker. Speech uses the original exchange identity.
    Returns the correlations of the pending actions.
    """
    handled = set()
    receipts = supervisor.receipts()
    for item in list(ghost.all_knoxels.values()):
        if (not isinstance(item, ActionSchemaRecord)
                or ActionMetadata.model_validate(item.metadata).dispatch_status != "pending"):
            continue
        pending = PendingDispatch.model_validate(item.metadata)
        correlation = pending.dispatch_correlation
        handled.add(correlation)
        exchange = exchange_record(ghost, correlation)
        admitted = any(row["action_id"] == pending.action_id for row in receipts.values() if "action_id" in row)
        if admitted or (exchange is not None
                        and RecordMetadataView.model_validate(exchange.metadata).exchange_status == "resolved"):
            acknowledge_intent(ghost, item.id)
            continue
        request = pending.dispatch_payload.model_copy(update={"knoxel_id": item.id,
                                                              "recovery_outcome": "intent_replayed"})
        bus.put(Envelope(id=new_id(), source="intent_recovery", kind="data", topic="action.request",
                         correlation_id=correlation, target=pending.dispatch_target, payload=request.payload()))
    return handled
