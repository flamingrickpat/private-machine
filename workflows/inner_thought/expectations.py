"""The expectation lifecycle helpers for the private loop.

An expectation is recorded before the outcome it anticipates. A later
observation resolves it to ``supported``, ``contradicted``, or ``unmet``. The
absence of a user reply is not evidence: a pending expectation stays pending
until an accepted record resolves it. The main loop's appraisal of the
resolution can then use surprise or disappointment without inventing intent.

All helpers stage records on the caller's open batch. The caller commits, so
the expectation lifecycle shares the loop's commit boundary and cancellation
behavior.
"""
from __future__ import annotations

from pm.memory.snapshot import RecordView
from pm.memory.write_batch import WriteBatch
from pm.records.cognitive_records import ExpectationRecord
from pm.records.base import Provenance
from pm.records.enums import SourceKind


def pending_expectations(record_source: RecordView) -> list[ExpectationRecord]:
    """Return all unresolved expectations, oldest first.

    ``record_source`` is any object with ``all_knoxels`` (a MemoryStore or a
    frozen snapshot view). A resolved record keeps its history and leaves the
    pending list.
    """
    records = [record for record in record_source.all_knoxels.values()
               if isinstance(record, ExpectationRecord) and record.status == "pending"]
    records.sort(key=lambda record: record.tick_id)
    return records


def open_expectation(batch: WriteBatch, *, anticipation: str, expected_outcome: str, expected_valence: float,
                     uncertainty: float, goal_ids: list[int], evidence_ids: list[int],
                     created_from_action_id: int | None = None, tick_id: int = 0) -> ExpectationRecord:
    """Stage one new expectation. The caller commits the batch."""
    record = ExpectationRecord(
        content=anticipation or expected_outcome,
        anticipation=anticipation,
        expected_outcome=expected_outcome,
        status="pending",
        expected_valence=expected_valence,
        uncertainty=uncertainty,
        goal_ids=list(goal_ids),
        created_from_action_id=created_from_action_id,
        evidence_ids=list(evidence_ids),
    )
    record.tick_id = tick_id or batch.tick_id
    record.provenance = Provenance(source_kind=SourceKind.Self, source_ref="private_loop")
    return batch.create(record)


def resolve_expectation(batch: WriteBatch, record: ExpectationRecord, *, status: str, observation_ids: list[int],
                        reason: str, revision: int) -> ExpectationRecord:
    """Stage one resolution. The caller commits the batch.

    ``status`` must be ``supported``, ``contradicted``, or ``unmet``. An empty
    ``observation_ids`` list is rejected: a resolution without an observation
    would invent an outcome. ``revision`` is the revision the private loop
    read; the commit rejects the update when another writer changed the
    record first. The original anticipation and evidence stay unchanged on
    the updated record.
    """
    if status not in {"supported", "contradicted", "unmet"}:
        raise ValueError(f"unknown expectation status: {status}")
    if not observation_ids:
        raise ValueError("an expectation resolution needs at least one observation id")
    batch.expect_revision(record.id, revision)
    # A view record carries its owner reference, which deep copying cannot
    # follow. The shallow copy keeps the scalars, and every nested value the
    # resolution changes is replaced with a new object, so the read record
    # stays untouched.
    updated = record.model_copy(deep=False)
    updated.status = status
    updated.observation_ids = list(observation_ids)
    updated.resolved_tick_id = batch.tick_id
    metadata = dict(updated.metadata or {})
    metadata["resolution_reason"] = reason
    metadata["resolved_by"] = "private_loop"
    updated.metadata = metadata
    batch.replace(updated)
    return updated


__all__ = ["open_expectation", "pending_expectations", "resolve_expectation"]
