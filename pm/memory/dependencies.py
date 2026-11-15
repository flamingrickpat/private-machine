"""Capture source meaning for atomic memory acceptance.

The package overview is :mod:`pm.memory`. A dependency keeps the exact source
identity and its meaning at capture time. Operational receipts can change its
revision without changing that meaning. :class:`pm.memory.store.MemoryStore`
compares these dependencies under its commit lock.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any, cast

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


#: Record fields that describe when this process handled the record, not what
#: the record means. ``timestamp_creation`` and ``last_accessed`` are real wall
#: clock, so including them makes an unchanged source look changed after any
#: touch, and makes an identical record built in another process hash
#: differently. World timestamps stay significant: they are evidence.
_PROCESS_BOOKKEEPING_FIELDS: frozenset[str] = frozenset({"timestamp_creation", "last_accessed"})

#: The inquiry receipt field that records accumulated active computation time.
#: The private loop's budget check reads it, but it differs on every run and is
#: a receipt of how long the run took, never evidence a reader reasons about.
#: The replay harness normalizes the same field in a rendered request.
_RECEIPT_TIMING_FIELD: str = "active_elapsed_seconds"
#: Stable stand-in for that field when a receipt's meaning is hashed.
_RECEIPT_TIMING_PLACEHOLDER: float = 0.0


def _meaning_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a record's metadata without its process bookkeeping.

    Only fields that differ between two runs of the same accepted state are
    removed: the lifecycle keys, and the elapsed duration inside an inquiry
    receipt. Every other metadata field stays significant.
    """
    meaning = {key: value for key, value in metadata.items() if key not in _LIFECYCLE_METADATA_KEYS}
    # Stored metadata is heterogeneous JSON. The nested receipt is checked
    # structurally, then read with its named field boundary.
    stored_receipt = meaning.get("inquiry_receipt")
    if isinstance(stored_receipt, dict):
        receipt = cast(dict[str, Any], stored_receipt)
        stored_state = receipt.get("state")
        if isinstance(stored_state, dict) and _RECEIPT_TIMING_FIELD in stored_state:
            state = cast(dict[str, Any], stored_state)
            meaning["inquiry_receipt"] = {**receipt, "state": {
                **state, _RECEIPT_TIMING_FIELD: _RECEIPT_TIMING_PLACEHOLDER}}
    return meaning


def source_fingerprint(record: RecordBase) -> str:
    """Hash source meaning while excluding embeddings and operational receipts.

    The explicit metadata exclusion contains only exchange, publication,
    dispatch, recovery, and execution bookkeeping. Content, attribution,
    provenance, scope, world timestamps, output references, and epistemic
    fields remain significant. This comparison does not authorize a
    status-dependent operation; such an operation also requires an exact
    revision guard.
    """
    semantic_record = record.model_copy(update={"metadata": _meaning_metadata(record.metadata)})
    data = semantic_record.model_dump(mode="json",
        exclude={"embedding", "embedding_model", *_PROCESS_BOOKKEEPING_FIELDS})
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


def changed_dependencies(view: Snapshot, dependencies: Iterable[SourceDependency]) -> tuple[int, ...]:
    """Return the IDs of captured dependencies that acceptance would refuse now, ascending.

    This applies the commit's own source rule
    (:meth:`pm.memory.store.MemoryStore.commit`) to a view: a dependency is
    changed when its record is gone, when it required current-head authority
    and has an accepted successor, or when its meaning fingerprint differs.
    Use it before an expensive calculation whose commit would certainly be
    refused. The commit stays authoritative; a change after this check is
    still refused there.
    """
    changed: list[int] = []
    for dependency in dependencies:
        record_id = dependency.record_id
        current = view.get_knoxel_by_id(record_id)
        lost_authority = dependency.current_head and record_id not in view.all_knoxels
        if current is None or lost_authority or source_fingerprint(current) != dependency.fingerprint:
            changed.append(record_id)
    return tuple(sorted(changed))
