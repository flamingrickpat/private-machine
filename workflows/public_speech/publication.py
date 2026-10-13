"""Durable public utterances and exchange resolution in the canonical MemoryStore.

Commit precedes bus publication. A restart replays unacknowledged utterances.
Delivery can repeat after a crash. Consumers deduplicate by the utterance ID.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pm.bus.envelopes import Envelope, new_id
from pm.memory.snapshot import GhostConflict
from pm.records.memory_records import Feature
from pm.records.base import Provenance
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from workflows.main_cognition.experience import utterance_metadata


def exchange_record(ghost, correlation: str):
    """Return the accepted user record for a correlated exchange, if present."""
    source = next((item for item in ghost.all_features
                 if item.feature_type == FeatureType.Dialogue
                 and item.metadata.get("exchange_correlation") == correlation
                 and item.provenance is not None and item.provenance.source_kind == SourceKind.User), None)
    return ghost.snapshot({source.id}).get(source.id) if source is not None else None


def commit_public_reply(ghost, text: str, correlation: str, *, commit_id: str,
                        action_id: int | None = None, context=None, recovery: str = "",
                        review_reason: str = "", review_judgment: str = "",
                        cancel=None) -> int | None:
    """Commit speech and exchange resolution together, or reject stale input.

    Completed exchanges return no new utterance. A revision conflict or newer
    user stimulus raises GhostConflict before any speech enters memory.
    """
    with ghost.lock:
        exchange = exchange_record(ghost, correlation) if correlation else None
        if exchange is not None and exchange.metadata.get("exchange_status") == "resolved":
            return None
        batch = ghost.begin_batch(commit_id, cancel=cancel)
        if context is not None:
            validate_public_context(ghost, context)
            for key, revision in context.revisions.items():
                batch.expect_revision(key, revision)
        reply = batch.create(Feature(
            content=text, feature_type=FeatureType.Dialogue, interlocus=InterlocusType.Public,
            causal=True, override_source=ghost.get_companion_name(),
            provenance=Provenance(source_kind=SourceKind.Self, source_ref=str(action_id or "public_recovery")),
            metadata={**utterance_metadata(batch.view(), action_id),
                      "reply_correlation": correlation, "publication_status": "pending",
                      "recovery_outcome": recovery, "review_reason": review_reason,
                      "review_judgment": review_judgment,
                      "executed_at": datetime.now(timezone.utc).isoformat()},
        ))
        if exchange is not None:
            exchange.metadata.update(exchange_status="resolved", public_reply_id=reply.id, recovery_outcome=recovery)
            batch.replace(exchange)
        if action_id is not None and action_id in ghost.revisions:
            action = ghost.snapshot({action_id}).get(action_id)
            if action.metadata.get("dispatch_status") == "pending":
                action.metadata["dispatch_status"] = "admitted"
                batch.replace(action)
        receipt = ghost.commit(batch)
        return receipt.added_ids[0]


def validate_public_context(ghost, context) -> None:
    """Reject changed sources before publication or grounding recovery.

    The caller holds the MemoryStore lock. Recovery must not replace old revision
    guards with a fresh snapshot after their original evidence changed.
    """
    for key, revision in context.revisions.items():
        if ghost.revisions.get(key) != revision:
            raise GhostConflict("Public evidence changed during generation.")
    users = [item.id for item in ghost.all_features if item.feature_type == FeatureType.Dialogue
             and item.override_source == ghost.get_user_name()]
    if max(users, default=None) != context.user_frontier:
        raise GhostConflict("A newer user stimulus superseded this reply.")
    state_tick = ghost.states[-1].tick_id if ghost.states else None
    if state_tick != context.state_tick:
        raise GhostConflict("Accepted internal state changed during public generation.")


def publish_reply(bus, source: str, correlation: str, reply_id: int, *, cause=None, **fields):
    """Publish a committed utterance. Its pending marker survives delivery failure."""
    bus.put(Envelope(id=new_id(), source=source, kind="data", topic="dialogue.reply",
                     correlation_id=correlation, payload={"knoxel_id": reply_id, **fields}, causation_id=cause))


def acknowledge_reply(ghost, reply_id: int) -> None:
    """Mark a delivered utterance without changing its text or factual status."""
    with ghost.lock:
        reply = ghost.snapshot({reply_id}).get(reply_id) if reply_id in ghost.revisions else None
        if reply is None or reply.metadata.get("publication_status") != "pending":
            return
        batch = ghost.begin_batch(f"public-ack:{reply_id}")
        reply.metadata["publication_status"] = "published"
        batch.replace(reply)
        ghost.commit(batch)


__all__ = ["acknowledge_reply", "commit_public_reply", "exchange_record", "publish_reply", "validate_public_context"]
