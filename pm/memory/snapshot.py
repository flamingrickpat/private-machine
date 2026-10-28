"""Typed membership and revisions over shared memory records."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator, Mapping
from typing import Any, Protocol

from pm.records.base import RecordBase
from pm.records.global_workspace import GwtTurnRecord
from pm.memory.vector_index import VectorIndex


class RecordView(Protocol):
    """Read-only record membership shared by a store and a frozen snapshot."""

    @property
    def all_knoxels(self) -> Mapping[int, RecordBase]: ...

    @property
    def revisions(self) -> Mapping[int, int]: ...


class GhostConflict(RuntimeError):
    """An inspected input changed before the caller's commit.

    The caller must refresh the view and reconsider its draft. Retrying the
    same commit cannot succeed; nothing of the rejected batch was accepted.
    """


class GhostCancelled(RuntimeError):
    """The owner revoked this read grant or cancelled the surrounding run."""



@dataclass(slots=True)
class RecordPayloads(Mapping[int, dict[str, Any]]):
    """Render tool payloads on demand over typed shared records."""

    records: Mapping[int, RecordBase]

    def __getitem__(self, key: int) -> dict[str, Any]:
        return self.records[key].as_payload(include_embedding=False)

    def __iter__(self) -> Iterator[int]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


@dataclass(slots=True)
class Snapshot:
    """Keep record references and captured revisions. Record values stay shared."""

    _records: dict[int, RecordBase]
    _gwt: dict[int, GwtTurnRecord]
    revisions: dict[int, int]
    embedding_index: VectorIndex | None = None
    commit_ids: frozenset[str] = frozenset()
    historical_records: dict[int, RecordBase] = field(default_factory=dict)

    @property
    def all_knoxels(self) -> dict[int, RecordBase]:
        """Return this view's members without copying their records."""
        return self._records

    @property
    def gwt_turns(self) -> Mapping[int, GwtTurnRecord]:
        """Expose the frozen workspace turns through a typed read interface."""
        return self._gwt

    def get(self, record_id: int) -> RecordBase:
        """Return the shared record. A missing ID raises KeyError."""
        if record_id in self._records:
            return self._records[record_id]
        return self.historical_records[record_id]

    def get_knoxel_by_id(self, record_id: int) -> RecordBase | None:
        """Return a shared member or None. Keep its central owner."""
        return self._records.get(record_id) or self.historical_records.get(record_id)

    def get_entity_name(self, entity_id: int) -> str:
        """Return the content of a visible entity."""
        return self.get(entity_id).content

    def serialize_knoxels(self) -> dict[int, dict[str, Any]]:
        """Render records for external tools and evidence capture.

        Each value is the ``__type__``/``data`` payload of one record, without
        its embedding. Record kind decides which fields ``data`` holds.
        """
        return {key: record.as_payload(include_embedding=False) for key, record in self._records.items()}

    def serialize_gwt(self) -> dict[int, dict[str, Any]]:
        """Render workspace turns for external tools and evidence capture."""
        return {key: record.model_dump(mode="json") for key, record in self._gwt.items()}

