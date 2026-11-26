"""A finished memory proposal and its publication at a Ghost boundary.

Each unit holds its own source grant. A worker reserves record IDs through
the ordinary write batch. Its private projection lets a parent cite a
finished child before either has been published. Unfinished units never
enter the store. A ready unit carries its vectors and a recovery receipt.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from threading import Event

from pydantic import BaseModel, Field

from pm.memory.read_scope import ReadScope
from pm.memory.references import referenced_ids
from pm.memory.snapshot import GhostConflict, Snapshot
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import RecordBase


class MemoryUnitReceipt(BaseModel):
    """Recoverable record IDs of one accepted episode, period, or narrative."""

    run_id: str
    kind: str = "memory_reconsolidation"
    unit: str
    topical_ids: list[int] = Field(default_factory=list[int])
    temporal_ids: list[int] = Field(default_factory=list[int])
    fact_ids: list[int] = Field(default_factory=list[int])
    semantic_ids: list[int] = Field(default_factory=list[int])
    semantic_updated_ids: list[int] = Field(default_factory=list[int])
    narrative_ids: list[int] = Field(default_factory=list[int])


@dataclass(frozen=True, slots=True)
class CapturedCaption:
    image_id: str
    description: str


@dataclass(frozen=True, slots=True)
class PreparedMemoryUnit:
    batch: WriteBatch
    receipt: MemoryUnitReceipt
    captions: tuple[CapturedCaption, ...] = ()


class MemoryPreparation:
    """Own the private, advancing record view of one worker."""

    def __init__(self, store: MemoryStore, view: Snapshot, run_id: str, cancel: Event) -> None:
        self.store = store
        self.view = view
        self.run_id = run_id
        self.cancel = cancel

    def begin(self, unit: str, source_ids: Iterable[int]) -> WriteBatch:
        """Capture the exact source heads used by the model and the filing decisions."""
        records: dict[int, RecordBase] = {}  # pm-lint: allow PM004 -- accepted sources by knoxel ID
        revisions: dict[int, int] = {}  # pm-lint: allow PM004 -- revision of each accepted source
        for key in source_ids:
            records[key] = self.view.all_knoxels[key]
            revisions[key] = self.view.revisions[key]
        scoped = Snapshot(records, {}, revisions)  # pm-lint: allow PM005 -- snapshot's three native ID indexes
        with self.store.lock:
            batch = WriteBatch(self.store, f"reconsolidation:{self.run_id}:{unit}", scoped,
                               self.store.current_tick_id, self.cancel, embed=False)
        for key, revision in revisions.items():
            batch.expect_head(key, revision)
        return batch

    def finish(self, batch: WriteBatch, receipt: MemoryUnitReceipt,
               captions: tuple[CapturedCaption, ...] = ()) -> PreparedMemoryUnit:
        """Seal source links and expose outputs only to later private preparation."""
        batch.check_cancelled()
        prepared = batch.view()
        outputs = {key: prepared.get(key) for key in batch.created_ids}
        # Existing semantic successors retain their earlier historical links.
        # They must be visible to reference validation even if the model read
        # only the latest head. Keep queued grants proportional to their inputs.
        references: set[int] = set()
        for record in outputs.values():
            references.update(referenced_ids(record))
        for key in references.difference(outputs.keys()):
            batch.include_reference(self.view, key)
        batch.set_consolidation_receipt(batch.commit_id, receipt.model_dump(mode="json"))  # pm-lint: allow PM007 -- checkpoint receipt serialization
        batch.cancel = None  # Wake may cancel the worker after accepting this complete proposal.
        for key, record in outputs.items():
            if record.supersedes_id is not None:
                old_id = record.supersedes_id
                self.view.historical_records[old_id] = self.view.all_knoxels.pop(old_id)
            self.view.all_knoxels[key] = record
            self.view.revisions[key] = 1
        return PreparedMemoryUnit(batch, receipt, captions)

    def publish(self, unit: PreparedMemoryUnit) -> bool:
        """Commit a ready unit. A concurrent correction leaves the sources eligible."""
        # A main-cycle caller may have its own read scope and parked invocation.
        # This independent transaction already carries the complete unit grant.
        with self.store.lock, ReadScope():
            try:
                self.store.commit(unit.batch)
            except GhostConflict:
                unit.batch.discard()
                return False
            for caption in unit.captions:
                self.store.annotate_alt_text(caption.image_id, caption.description, "model", checkpoint=False)
        return True
