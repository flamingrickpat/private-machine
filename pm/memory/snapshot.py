"""Immutable, scoped snapshots of the memory store, and commit failures.

A snapshot holds serialized records at known revisions. Readers never see
proposals that a write batch has not committed.
"""
from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType

from pm.records.base import RecordBase
from pm.memory.yaml_codec import _deserialize_knoxel


class GhostConflict(RuntimeError):
    pass


class GhostCancelled(RuntimeError):
    pass


class Snapshot:
    def __init__(self, records: dict[int, dict], gwt: dict[int, dict], revisions: dict[int, int]):
        self._records = records
        self._gwt = gwt
        self.revisions = MappingProxyType(revisions)

    @property
    def all_knoxels(self) -> dict[int, RecordBase]:
        return {record_id: self.get(record_id) for record_id in self._records}

    def get(self, record_id: int) -> RecordBase:
        record = _deserialize_knoxel(deepcopy(self._records[record_id]))
        record._owner = self
        return record

    def get_knoxel_by_id(self, record_id: int) -> RecordBase | None:
        return self.get(record_id) if record_id in self._records else None

    def get_entity_name(self, entity_id: int) -> str:
        return self.get(entity_id).content

    def serialize_knoxels(self) -> dict[int, dict]:
        return deepcopy(self._records)

    def serialize_gwt(self) -> dict[int, dict]:
        return deepcopy(self._gwt)

