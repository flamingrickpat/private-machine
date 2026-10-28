"""Durable public utterances and exchange resolution in the canonical MemoryStore.

Commit precedes bus publication. A restart replays unacknowledged utterances.
Delivery can repeat after a crash. Consumers deduplicate by the utterance ID.
"""
from __future__ import annotations

from pm import vclock

from datetime import datetime, timezone
from threading import Event
from typing import TYPE_CHECKING, Any

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, new_id
from pm.memory.snapshot import GhostConflict
from pm.records.memory_records import Feature
from pm.records.base import Provenance, RecordBase
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from workflows.main_cognition.experience import utterance_metadata

if TYPE_CHECKING:
    from pm.memory.store import MemoryStore
    from workflows.public_speech.public_context import PublicContext


def exchange_record(ghost: MemoryStore, correlation: str) -> RecordBase | None:
    """Return the accepted user record for a correlated exchange, if present."""
    source = next((item for item in ghost.all_features
                 if item.feature_type == FeatureType.Dialogue
                 and item.metadata.get("exchange_correlation") == correlation
                 and item.provenance is not None and item.provenance.source_kind == SourceKind.User), None)
    return ghost.snapshot({source.id}).get(source.id) if source is not None else None


def commit_public_reply(ghost: MemoryStore, text: str, correlation: str, *, commit_id: str,
                        action_id: int | None = None, context: PublicContext | None = None,
                        recovery: str = "", review_reason: str = "", review_judgment: str = "",
                        addressee_entity_id: int | None = None,
                        cancel: Event | None = None) -> int | None:
    """Commit speech and exchange resolution together, or reject stale input.

    ``addressee_entity_id`` is the entity that the reply answers, or None for
    the user. The reply record stores it, so the dialogue with a second
    entity stays in the shard of that entity. Completed exchanges return no
    new utterance. Changed evidence raises GhostConflict. New messages and
    later feelings do not cancel earlier speech.
    """
    with ghost.lock:
        exchange = exchange_record(ghost, correlation) if correlation else None
        action = ghost.snapshot({action_id}).get(action_id) if action_id is not None else None
        if action is not None and action.metadata.get("public_reply_id") is not None:
            return None
        if action_id is None and exchange is not None and exchange.metadata.get("exchange_status") == "resolved":
            return None
        batch = ghost.begin_batch(commit_id, cancel=cancel)
        inputs = set(context.revisions) if context is not None else set()
        if context is not None:
            inputs.update(context.input_record_ids)
            batch.depend_on_commits(context.input_commit_ids)
        if exchange is not None:
            inputs.add(exchange.id)
        authority_id = action_id if action_id is not None else context.action_schema_id if context is not None else None
        batch.depend_on(inputs, authority=authority_id,
                        manifest={"kind": "public_reply", "context": context.model_dump(mode="json") if context is not None else None})
        if context is not None:
            validate_public_context(ghost, context)
            for key, revision in context.revisions.items():
                batch.expect_revision(key, ghost.revisions[key])
        reply = batch.create(Feature(
            content=text, feature_type=FeatureType.Dialogue, interlocus=InterlocusType.Public,
            causal=True, override_source=ghost.get_companion_name(),
            addressee_entity_id=addressee_entity_id or 0,
            provenance=Provenance(source_kind=SourceKind.Self, source_ref=str(action_id or "public_recovery")),
            metadata={**utterance_metadata(batch.view(), action_id),
                      "reply_correlation": correlation, "publication_status": "pending",
                      "recovery_outcome": recovery, "review_reason": review_reason,
                      "review_judgment": review_judgment,
                      "generation_user_frontier": context.user_frontier if context is not None else None,
                      "generation_state_tick": context.state_tick if context is not None else None,
                      "generation_source_revisions": dict(context.revisions) if context is not None else {},
                      "responding_to_ids": list(context.stimulus_user_ids) if context is not None else [],
                      "executed_at": vclock.now_aware().isoformat()},
        ))
        resolved_ids = set(context.stimulus_user_ids) if context is not None else set()
        if exchange is not None:
            resolved_ids.add(exchange.id)
        for key in resolved_ids:
            source = ghost.snapshot({key}).get(key).copy_for_edit()
            if source.metadata.get("exchange_status") == "pending":
                batch.set_lifecycle(key, exchange_status="resolved", public_reply_id=reply.id, recovery_outcome=recovery)
        if action_id is not None and action_id in ghost.revisions:
            batch.set_lifecycle(action_id, dispatch_status="admitted", execution_status="completed", public_reply_id=reply.id)
        receipt = ghost.commit(batch)
        return receipt.added_ids[0]


def validate_public_context(ghost: MemoryStore, context: PublicContext) -> None:
    """Reject changed sources before publication or grounding recovery.

    The caller holds the MemoryStore lock. Recovery must not replace old revision
    guards with a fresh snapshot after their original evidence changed.
    """
    if any(key not in ghost.all_knoxels for key in context.head_ids):
        raise GhostConflict("Public evidence acquired a successor during generation.")
    for key, revision in context.revisions.items():
        if ghost.revisions.get(key) != revision:
            from workflows.public_speech.public_context import source_fingerprint
            current = ghost.get_knoxel_by_id(key)
            if (current is None or key not in context.source_fingerprints
                    or source_fingerprint(current) != context.source_fingerprints[key]):
                raise GhostConflict(f"Public evidence {key} changed during generation.")


def publish_reply(bus: BusSystem, source: str, correlation: str, reply_id: int, *,
                  cause: str | None = None, **fields: Any) -> None:
    """Publish a committed utterance. Its pending marker survives delivery failure."""
    bus.put(Envelope(id=new_id(), source=source, kind="data", topic="dialogue.reply",
                     correlation_id=correlation, payload={"knoxel_id": reply_id, **fields}, causation_id=cause))


def acknowledge_reply(ghost: MemoryStore, reply_id: int) -> None:
    """Mark a delivered utterance without changing its text or factual status."""
    with ghost.lock:
        reply = ghost.snapshot({reply_id}).get(reply_id) if reply_id in ghost.revisions else None
        if reply is None or reply.metadata.get("publication_status") != "pending":
            return
        batch = ghost.begin_batch(f"public-ack:{reply_id}")
        batch.depend_on([reply_id], authority=reply_id)
        batch.set_lifecycle(reply_id, publication_status="published")
        ghost.commit(batch)


__all__ = ["acknowledge_reply", "commit_public_reply", "exchange_record", "publish_reply", "validate_public_context"]
