"""Typed membership and revisions over shared memory records."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator, Mapping
from typing import Any

from pm.records.base import RecordBase
from pm.records.global_workspace import GwtTurnRecord
from pm.memory.vector_index import VectorIndex


class GhostConflict(RuntimeError):
    pass


class GhostCancelled(RuntimeError):
    pass


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

    def serialize_knoxels(self) -> dict[int, dict]:
        """Render records for external tools and evidence capture."""
        return {key: record.as_payload(include_embedding=False) for key, record in self._records.items()}

    def serialize_gwt(self) -> dict[int, dict]:
        """Render workspace turns for external tools and evidence capture."""
        return {key: record.model_dump(mode="json") for key, record in self._gwt.items()}

