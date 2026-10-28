"""Private proposals over a scoped, stable MemoryStore read view.

Invalid proposals, stale commits, and cancellation are explicit failures.
Model and network calls belong outside these objects. The batch is a trusted
host API, not an unrestricted tool grant. Role-scoped write tools belong to W3.
"""
from __future__ import annotations

from pm.records.copying import pickle_copy

from threading import Event
from typing import TYPE_CHECKING

from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Entity, Feature
from pm.records.base import RecordBase
from pm.records.base import get_causal_datetime, get_real_datetime
from pm.records.lineage import RecordLineage
from pm.memory.snapshot import GhostCancelled, Snapshot

if TYPE_CHECKING:
    from pm.memory.store import MemoryStore


class WriteBatch:
    """Stage typed records. Callers copy accepted records before an isolated edit."""

    def __init__(self, ghost: MemoryStore, commit_id: str, snapshot: Snapshot, tick_id: int, cancel: Event | None, *, embed: bool = True):
        self._ghost = ghost
        self.commit_id = commit_id
        self._snapshot = snapshot
        self.tick_id = tick_id
        self.cancel = cancel
        self.embed = embed
        self._new: dict[int, RecordBase] = {}
        self._replacements: dict[int, RecordBase] = {}
        self._superseded: set[int] = set()
        self._successor_ids: dict[int, int] = {}
        self._lifecycle: dict[int, dict] = {}
        self._input_ids: set[int] = set()
        self._input_commits: set[str] = set()
        self._authority: int | None = None
        self._manifest: dict = {}
        self._expected: dict[int, int] = {}
        self._heads: set[int] = set()
        self._metadata_guards: list[tuple[str, object, dict[int, int]]] = []
        self._initial_runs = dict(ghost.managed_runs)
        self._runs: dict[str, dict] = {}
        self._initial_consolidations = dict(ghost.consolidations)
        self._consolidations: dict[str, dict] = {}
        self._consolidation_run_guards: set[str] = set()
        self._gwt: GwtTurnRecord | None = None
        self._cycle: CycleState | None = None
        self._closed = False

    def check_cancelled(self):
        if self._closed or (self.cancel is not None and self.cancel.is_set()):
            raise GhostCancelled("The Ghost batch is closed or cancelled.")

    def view(self, visible_ids=None) -> Snapshot:
        records = {key: record for key, record in self._snapshot._records.items() if key not in self._superseded}
        records.update(self._new)
        records.update(self._replacements)
        if visible_ids is not None:
            records = {record_id: records[record_id] for record_id in visible_ids}
        gwt = {tick: raw for tick, raw in self._snapshot._gwt.items()
               if set(raw.knoxel_ids).issubset(set(records) | set(self._snapshot.historical_records))}
        history = {**self._snapshot.historical_records,
                   **{key: self._snapshot.get(key) for key in self._superseded}}
        if visible_ids is not None:
            history = {}
        revisions = {key: revision for key, revision in self._snapshot.revisions.items() if key in records or key in history}
        return Snapshot(records, gwt, revisions, self._snapshot.embedding_index, self._snapshot.commit_ids, history)

    def create(self, knoxel: RecordBase) -> RecordBase:
        self.check_cancelled()
        record = knoxel
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
            for entity in self._snapshot._records.values():
                if isinstance(entity, Entity) and entity.content == record.override_source:
                    record.entity_id = entity.id
                    break
        record._owner = self._ghost
        self._new[record.id] = record
        return record

    def replace(self, knoxel: RecordBase) -> RecordBase:
        """Append a successor with a new ID and guard the previous head.

        Existing payloads stay unchanged. Use lifecycle receipts for exchange
        and execution status. Those receipts do not create mental events.
        """
        self.check_cancelled()
        if knoxel.id in self._new:
            self.revise(knoxel)
            return knoxel
        if knoxel.id in self._successor_ids:
            knoxel.id = self._successor_ids[knoxel.id]
            knoxel.tick_id = self.tick_id
            staged = self._new[knoxel.id]
            knoxel.supersedes_id = staged.supersedes_id
            knoxel.timestamp_creation = staged.timestamp_creation
            knoxel.timestamp_world_begin = staged.timestamp_world_begin
            knoxel.timestamp_world_end = staged.timestamp_world_end
            self.revise(knoxel)
            return knoxel
        previous = self._snapshot.get(knoxel.id)
        if (type(knoxel) is not type(previous) or knoxel.scope != previous.scope or knoxel.tick_id != previous.tick_id):
            raise ValueError("Replacement cannot change a knoxel's class, visibility scope, or creation tick.")
        if knoxel.content != previous.content:
            knoxel.embedding = []
        previous_id = knoxel.id
        self._expected.setdefault(previous_id, self._snapshot.revisions[previous_id])
        self._input_ids.add(previous_id)
        self._superseded.add(previous_id)
        knoxel.id = -1
        knoxel.tick_id = self.tick_id
        knoxel.supersedes_id = previous_id
        # The successor starts now. Its predecessor retains the original dates.
        knoxel.timestamp_creation = get_real_datetime()
        knoxel.timestamp_world_begin = get_causal_datetime()
        knoxel.timestamp_world_end = knoxel.timestamp_world_begin
        created = self.create(knoxel)
        self._successor_ids[previous_id] = created.id
        return created

    def set_lifecycle(self, record_id: int, **fields) -> None:
        """Append operational status without changing the source knoxel.

        The live view overlays these fields for existing workflow readers.
        Raw history keeps the original payload and each separate receipt.
        """
        self.check_cancelled()
        if record_id not in self._snapshot._records and record_id not in self._new:
            raise ValueError("Lifecycle source is outside the batch view.")
        self._lifecycle.setdefault(record_id, {}).update(pickle_copy(fields))
        if record_id not in self._new:
            self._input_ids.add(record_id)

    def depend_on(self, record_ids, *, authority: int | None = None, manifest: dict | None = None) -> None:
        """Capture exact input identities and optional approved action authority."""
        self.check_cancelled()
        self._input_ids.update(record_ids)
        if authority is not None:
            self._authority = authority
            self._input_ids.add(authority)
        if manifest is not None:
            self._manifest.update(pickle_copy(manifest))

    def depend_on_commits(self, commit_ids) -> None:
        """Retain receipt and collection-view dependencies without a knoxel."""
        self._input_commits.update(commit_ids)

    def delete(self, record_id: int) -> RecordBase:
        """Append a tombstone. An excluded deletion cannot hide accepted data."""
        record = self._snapshot.get(record_id).copy_for_edit()
        record.tombstone = True
        return self.replace(record)

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
        previous = self._new[knoxel.id]
        if (type(knoxel) is not type(previous) or knoxel.scope != previous.scope or knoxel.tick_id != previous.tick_id):
            raise ValueError("Revise cannot change a knoxel's class, visibility scope, or creation tick.")
        self._new[knoxel.id] = knoxel

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
        self._new[record_id].lineage = lineage

    def expect_revision(self, record_id: int, revision: int) -> None:
        """Guard an exact historical payload. Supersession does not change its revision."""
        self.check_cancelled()
        if record_id not in self._snapshot.revisions:
            raise ValueError("The evidence is outside the batch view.")
        self._expected[record_id] = revision

    def expect_head(self, record_id: int, revision: int) -> None:
        """Require a cited version to remain current through this commit.

        Use this guard when a decision requires the current fact. Historical
        evidence only needs an exact ID and its revision guard.
        """
        self.expect_revision(record_id, revision)
        self._heads.add(record_id)

    def expect_metadata_version(self, key: str, value: object, revisions: dict[int, int]) -> None:
        """Guard a versioned record family, including an empty family.

        Pass the matching IDs and revisions from the invocation's frozen view.
        Commit rejects a concurrent append or replacement in that family.
        This guard closes the gap that a revision check on the old head leaves.
        """
        self.check_cancelled()
        self._metadata_guards.append((key, pickle_copy(value), dict(revisions)))

    def set_gwt(self, record: GwtTurnRecord, cycle: CycleState | None = None) -> None:
        self.check_cancelled()
        if record.tick_id != self.tick_id or (cycle is not None and cycle.tick_id != self.tick_id):
            raise ValueError("GWT and cycle ticks must match the batch tick.")
        self._gwt = record
        self._cycle = cycle

    def set_run_receipt(self, run_id: str, receipt: dict) -> None:
        self.check_cancelled()
        if receipt["state"] not in {"accepted", "running", "waiting_for_input", "completed", "failed", "cancelled", "interrupted"}:
            raise ValueError("Unknown managed-run state.")
        self._runs[run_id] = pickle_copy(receipt)

    def set_consolidation_receipt(self, key: str, receipt: dict) -> None:
        """Stage one consolidation receipt. Repeated keys reuse accepted state."""
        self.check_cancelled()
        self._consolidations[key] = pickle_copy(receipt)

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

