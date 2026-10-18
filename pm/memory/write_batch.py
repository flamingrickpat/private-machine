"""Private proposals over a scoped, stable MemoryStore read view.

Invalid proposals, stale commits, and cancellation are explicit failures.
Model and network calls belong outside these objects. The batch is a trusted
host API, not an unrestricted tool grant. Role-scoped write tools belong to W3.
"""
from __future__ import annotations

from copy import deepcopy
from threading import Event

from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Feature
from pm.records.base import RecordBase
from pm.records.lineage import RecordLineage
from pm.memory.yaml_codec import _deserialize_knoxel, _serialize_knoxel
from pm.memory.snapshot import GhostCancelled, Snapshot


class WriteBatch:
    def __init__(self, ghost, commit_id: str, snapshot: Snapshot, tick_id: int, cancel: Event | None):
        self._ghost = ghost
        self.commit_id = commit_id
        self._snapshot = snapshot
        self.tick_id = tick_id
        self.cancel = cancel
        self._new: dict[int, dict] = {}
        self._replacements: dict[int, dict] = {}
        self._expected: dict[int, int] = {}
        self._metadata_guards: list[tuple[str, object, dict[int, int]]] = []
        self._initial_runs = deepcopy(ghost.managed_runs)
        self._runs: dict[str, dict] = {}
        self._initial_consolidations = deepcopy(ghost.consolidations)
        self._consolidations: dict[str, dict] = {}
        self._consolidation_run_guards: set[str] = set()
        self._gwt: GwtTurnRecord | None = None
        self._cycle: CycleState | None = None
        self._closed = False

    def check_cancelled(self):
        if self._closed or (self.cancel is not None and self.cancel.is_set()):
            raise GhostCancelled("The Ghost batch is closed or cancelled.")

    def view(self, visible_ids=None) -> Snapshot:
        records = {**self._snapshot._records, **self._new, **self._replacements}
        if visible_ids is not None:
            records = {record_id: records[record_id] for record_id in visible_ids}
        gwt = {tick: raw for tick, raw in self._snapshot._gwt.items()
               if set(raw["knoxel_ids"]).issubset(records)}
        revisions = {record_id: revision for record_id, revision in self._snapshot.revisions.items() if record_id in records}
        return Snapshot(deepcopy(records), deepcopy(gwt), revisions)

    def create(self, knoxel: RecordBase) -> RecordBase:
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

    def replace(self, knoxel: RecordBase) -> None:
        self.check_cancelled()
        raw = _serialize_knoxel(_deserialize_knoxel(_serialize_knoxel(knoxel)))
        previous = self._snapshot._records[knoxel.id]
        if (raw["__type__"] != previous["__type__"]
                or raw["data"]["scope"] != previous["data"]["scope"]
                or raw["data"]["tick_id"] != previous["data"]["tick_id"]):
            raise ValueError("Replacement cannot change a knoxel's class, visibility scope, or creation tick.")
        self._replacements[knoxel.id] = raw
        # A retrieved revision can predate this write batch. Keep its earlier
        # read guard even when a later proposal also replaces the record.
        self._expected.setdefault(knoxel.id, self._snapshot.revisions[knoxel.id])

    def revise(self, knoxel: RecordBase) -> None:
        """Replace a record that this batch created and did not commit yet.

        Two new records can point at each other, for example a fact and its
        canonical semantic record. The second id exists only after the first
        ``create``, so the first record needs this update. The class, the
        scope, and the tick must stay the same. An id that this batch did not
        create raises ValueError; use ``replace`` for accepted records.
        """
        self.check_cancelled()
        if knoxel.id not in self._new:
            raise ValueError("Revise requires a record created in this batch.")
        raw = _serialize_knoxel(_deserialize_knoxel(_serialize_knoxel(knoxel)))
        previous = self._new[knoxel.id]
        if (raw["__type__"] != previous["__type__"]
                or raw["data"]["scope"] != previous["data"]["scope"]
                or raw["data"]["tick_id"] != previous["data"]["tick_id"]):
            raise ValueError("Revise cannot change a knoxel's class, visibility scope, or creation tick.")
        self._new[knoxel.id] = raw

    @property
    def created_ids(self) -> frozenset[int]:
        """Return IDs created in this batch, excluding replacements of old sources."""
        return frozenset(self._new)

    def annotate_lineage(self, record_id: int, lineage: RecordLineage) -> None:
        """Attach causal episode links to a new record in this batch only.

        Existing sources cannot acquire a new history through this API.
        Reference validation still occurs at the atomic commit boundary.
        """
        self.check_cancelled()
        if record_id not in self._new:
            raise ValueError("Lineage annotation requires a record created in this batch.")
        self._new[record_id]["data"]["lineage"] = lineage.model_dump(mode="json")

    def expect_revision(self, record_id: int, revision: int) -> None:
        """Reject the whole commit if retrieved evidence changed after its read."""
        self.check_cancelled()
        if record_id not in self._snapshot.revisions:
            raise ValueError("The evidence is outside the batch view.")
        self._expected[record_id] = revision

    def expect_metadata_version(self, key: str, value: object, revisions: dict[int, int]) -> None:
        """Guard a versioned record family, including an empty family.

        Pass the matching IDs and revisions from the invocation's frozen view.
        Commit rejects a concurrent append or replacement in that family.
        This guard closes the gap that a revision check on the old head leaves.
        """
        self.check_cancelled()
        self._metadata_guards.append((key, deepcopy(value), dict(revisions)))

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

    def guard_consolidation_run(self, run_id: str) -> None:
        """Require the run's artifact ledger to stay unchanged until commit."""
        self._consolidation_run_guards.add(run_id)

    def discard(self) -> None:
        self._closed = True
        self._new.clear()
        self._replacements.clear()
        self._runs.clear()
        self._consolidations.clear()
        self._gwt = None
        self._cycle = None

