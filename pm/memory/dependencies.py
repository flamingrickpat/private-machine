"""Capture source meaning for atomic memory acceptance.

The package overview is :mod:`pm.memory`. A dependency keeps the exact source
identity and its meaning at capture time. Operational receipts can change its
revision without changing that meaning. :class:`pm.memory.store.MemoryStore`
compares these dependencies under its commit lock.
"""
from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field

from pm.memory.snapshot import Snapshot
from pm.records.base import RecordBase


_LIFECYCLE_METADATA_KEYS: frozenset[str] = frozenset({
    "exchange_status", "public_reply_id", "publication_status",
    "recovery_outcome", "dispatch_status", "execution_status",
})


class SourceDependency(BaseModel):
    """One frozen source identity, meaning, and requirement for current authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    record_id: int = Field(description="Exact source record ID in the captured memory view.")
    revision: int = Field(ge=1, description="Store revision observed when the source was captured.")
    fingerprint: str = Field(min_length=1, description="SHA-256 of source meaning before calculation starts.")
    current_head: bool = Field(description="Whether acceptance requires this exact record to remain a live head.")


def source_fingerprint(record: RecordBase) -> str:
    """Hash source meaning while excluding embeddings and operational receipts.

    The explicit metadata exclusion contains only exchange, publication,
    dispatch, recovery, and execution bookkeeping. Content, attribution,
    provenance, scope, timestamps, output references, and epistemic fields
    remain significant. This comparison does not authorize a status-dependent
    operation; such an operation also requires an exact revision guard.
    """
    semantic_metadata = {
        key: value for key, value in record.metadata.items()
        if key not in _LIFECYCLE_METADATA_KEYS
    }
    semantic_record = record.model_copy(update={"metadata": semantic_metadata})
    data = semantic_record.model_dump(mode="json", exclude={"embedding", "embedding_model"})
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def capture_source(view: Snapshot, record_id: int, *, current_head: bool) -> SourceDependency:
    """Freeze the selected source before calculation, including historical evidence.

    A missing source raises ``KeyError``. An explicitly historical source
    cannot receive current authority. Commit checks current membership again,
    because the source can acquire a successor after this capture.
    """
    record = view.get(record_id)
    if current_head and record_id not in view.all_knoxels:
        raise ValueError("A historical source cannot require current-head authority.")
    return SourceDependency(record_id=record_id, revision=view.revisions[record_id],
                            fingerprint=source_fingerprint(record), current_head=current_head)
