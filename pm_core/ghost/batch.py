"""Private proposals over a scoped, stable Ghost read view.

Invalid proposals, stale commits, and cancellation are explicit failures.
Model and network calls belong outside these objects. The batch is a trusted
host API, not an unrestricted tool grant. Role-scoped write tools belong to W3.
"""
from __future__ import annotations

from copy import deepcopy
from threading import Event
from types import MappingProxyType

from pm_core.model.ghost_state import CycleState
from pm_core.model.gwt import GwtTurnRecord
from pm_core.model.knoxel_common import Feature
from pm_core.model.knoxel_core import KnoxelBase
from pm_core.persist.dataprovider_yaml import _deserialize_knoxel, _serialize_knoxel


class GhostConflict(RuntimeError):
    pass


class GhostCancelled(RuntimeError):
    pass


class GhostView:
    def __init__(self, records: dict[int, dict], gwt: dict[int, dict], revisions: dict[int, int]):
        self._records = records
        self._gwt = gwt
        self.revisions = MappingProxyType(revisions)

    @property
    def all_knoxels(self) -> dict[int, KnoxelBase]:
        return {record_id: self.get(record_id) for record_id in self._records}

    def get(self, record_id: int) -> KnoxelBase:
        record = _deserialize_knoxel(deepcopy(self._records[record_id]))
        record._owner = self
        return record

    def get_knoxel_by_id(self, record_id: int) -> KnoxelBase | None:
        return self.get(record_id) if record_id in self._records else None

    def get_entity_name(self, entity_id: int) -> str:
        return self.get(entity_id).content

    def serialize_knoxels(self) -> dict[int, dict]:
        return deepcopy(self._records)

    def serialize_gwt(self) -> dict[int, dict]:
        return deepcopy(self._gwt)


class GhostWriteBatch:
    def __init__(self, ghost, commit_id: str, snapshot: GhostView, tick_id: int, cancel: Event | None):
        self._ghost = ghost
        self.commit_id = commit_id
        self._snapshot = snapshot
        self.tick_id = tick_id
        self.cancel = cancel
        self._new: dict[int, dict] = {}
        self._replacements: dict[int, dict] = {}
        self._expected: dict[int, int] = {}
        self._initial_runs = deepcopy(ghost.managed_runs)
        self._runs: dict[str, dict] = {}
        self._initial_consolidations = deepcopy(ghost.consolidations)
        self._consolidations: dict[str, dict] = {}
        self._gwt: GwtTurnRecord | None = None
        self._cycle: CycleState | None = None
        self._closed = False

    def check_cancelled(self):
        if self._closed or (self.cancel is not None and self.cancel.is_set()):
            raise GhostCancelled("The Ghost batch is closed or cancelled.")

    def view(self, visible_ids=None) -> GhostView:
        records = {**self._snapshot._records, **self._new, **self._replacements}
        if visible_ids is not None:
            records = {record_id: records[record_id] for record_id in visible_ids}
        gwt = {tick: raw for tick, raw in self._snapshot._gwt.items()
               if set(raw["knoxel_ids"]).issubset(records)
               and (raw["attention_frame_id"] is None or raw["attention_frame_id"] in records)}
        revisions = {record_id: revision for record_id, revision in self._snapshot.revisions.items() if record_id in records}
        return GhostView(deepcopy(records), deepcopy(gwt), revisions)

    def create(self, knoxel: KnoxelBase) -> KnoxelBase:
        self.check_cancelled()
        record = _deserialize_knoxel(_serialize_knoxel(knoxel))
        if record.id != -1:
            raise ValueError("Create requires an unassigned knoxel ID.")
        if record.tick_id not in {-1, self.tick_id}:
            raise ValueError("New knoxels must use the batch tick.")
        with self._ghost.lock:
            self._ghost.current_knoxel_id += 1
            record.id = self._ghost.current_knoxel_id
        if record.tick_id == -1:
            record.tick_id = self.tick_id
        if isinstance(record, Feature) and not record.entity_id and record.override_source:
            for raw in self._snapshot._records.values():
                if raw["__type__"] == "Entity" and raw["data"]["content"] == record.override_source:
                    record.entity_id = raw["data"]["id"]
                    break
        self._new[record.id] = _serialize_knoxel(record)
        return record

    def replace(self, knoxel: KnoxelBase) -> None:
        self.check_cancelled()
        raw = _serialize_knoxel(_deserialize_knoxel(_serialize_knoxel(knoxel)))
        previous = self._snapshot._records[knoxel.id]
        if (raw["__type__"] != previous["__type__"]
                or raw["data"]["scope"] != previous["data"]["scope"]
                or raw["data"]["tick_id"] != previous["data"]["tick_id"]):
            raise ValueError("Replacement cannot change a knoxel's class, visibility scope, or creation tick.")
        self._replacements[knoxel.id] = raw
        self._expected[knoxel.id] = self._snapshot.revisions[knoxel.id]

    def set_gwt(self, record: GwtTurnRecord, cycle: CycleState | None = None) -> None:
        self.check_cancelled()
        if record.tick_id != self.tick_id or (cycle is not None and cycle.tick_id != self.tick_id):
            raise ValueError("GWT and cycle ticks must match the batch tick.")
        self._gwt = GwtTurnRecord.model_validate(record.model_dump())
        self._cycle = CycleState.model_validate(cycle.model_dump()) if cycle is not None else None

    def set_run_receipt(self, run_id: str, receipt: dict) -> None:
        self.check_cancelled()
        if receipt["state"] not in {"accepted", "running", "waiting_for_input", "completed", "failed", "cancelled", "interrupted"}:
            raise ValueError("Unknown managed-run state.")
        self._runs[run_id] = deepcopy(receipt)

    def set_consolidation_receipt(self, key: str, receipt: dict) -> None:
        """Stage one consolidation receipt. Repeated keys reuse accepted state."""
        self.check_cancelled()
        self._consolidations[key] = deepcopy(receipt)

    def discard(self) -> None:
        self._closed = True
        self._new.clear()
        self._replacements.clear()
        self._runs.clear()
        self._consolidations.clear()
        self._gwt = None
        self._cycle = None
