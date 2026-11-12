"""Stage private nominations without selecting or executing them.

The subsystem overview is :mod:`workflows.inner_thought.host`. These functions
use an existing frozen memory grant. They perform no inference, retrieval,
commit, GWT update, or bus publication. The caller owns those boundaries.
Original source records retain attribution and reality. Following a thought's
lineage never turns its descendants into independent corroboration.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from pm.memory.dependencies import capture_source
from pm.memory.snapshot import Snapshot
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordBase, RecordScope
from pm.records.cognitive_records import BeliefRecord
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.lineage import RecordLineage
from pm.records.memory_records import (
    CauseEffectRecord, DeclarativeFactRecord, Entity, Feature, MemoryClusterRecord, Narrative, SemanticRecord,
)
from workflows.inner_thought.operations import validate_operation
from workflows.inner_thought.schemas import InquiryGrant, NominationReceipt, ThoughtNomination


NOMINATION_KEY = "thought_nomination"


class NominationMetadata(BaseModel):
    """Read the typed portion of a heterogeneous Feature metadata boundary."""

    model_config = ConfigDict(extra="ignore")
    thought_nomination: NominationReceipt


@dataclass(frozen=True, slots=True)
class NominationRecord:
    """A staged or accepted nomination with its actual memory identity."""

    record: Feature
    receipt: NominationReceipt


def _source_parents(record: RecordBase) -> tuple[int, ...] | None:
    """Return internal derivation sources, or None for an original record."""
    if isinstance(record, MemoryClusterRecord):
        ids: list[int] = []
        for encoded in (record.included_event_ids, record.included_cluster_ids):
            if encoded is not None:
                ids.extend(int(value.strip()) for value in encoded.split(",") if value.strip())
        return tuple(dict.fromkeys(ids))
    if isinstance(record, (DeclarativeFactRecord, CauseEffectRecord)):
        return (record.source_cluster_id,) if record.source_cluster_id is not None else ()
    if isinstance(record, SemanticRecord):
        # Canonical back-pointers are not causal parents and would form cycles.
        return tuple(record.source_knoxel_ids)
    if isinstance(record, BeliefRecord):
        return tuple(dict.fromkeys([*record.supporting_knoxel_ids, *record.contradicting_knoxel_ids]))
    if isinstance(record, Narrative):
        # A tick range alone identifies no original event. Retain the prior as
        # internal context without manufacturing independent source roots.
        return tuple(record.lineage.source_event_ids) if record.lineage is not None else ()
    if (not isinstance(record, Feature)
            or record.provenance is None or record.provenance.source_kind != SourceKind.Self):
        return None
    derived_thought = record.feature_type in {FeatureType.Thought, FeatureType.InnerMonologue}
    derived_receipt = (record.feature_type == FeatureType.StateUpdate and record.lineage is not None
                       and record.lineage.operation in {"inquiry_transition", "private_attention"})
    if not (derived_thought or derived_receipt):
        return None
    if record.lineage is None:
        raise ValueError("A derived private thought requires explicit source lineage.")
    return tuple(dict.fromkeys([*record.lineage.source_event_ids, *record.lineage.parent_record_ids]))


def source_roots(view: Snapshot, source_ids: Sequence[int], *, scope: RecordScope,
                 permitted_ids: frozenset[int]) -> tuple[int, ...]:
    """Resolve original roots inside the frozen grant, rejecting cycles and gaps.

    A source-free internal thought has no external roots. Its actual event ID
    can still be cited as internal evidence. Neither branches nor repeated
    summaries create more original roots. Legacy summaries without exact
    source links remain context with no independently attributable roots.
    """
    roots: set[int] = set()
    visited: set[int] = set()
    visiting: set[int] = set()

    def visit(record_id: int) -> None:
        if record_id not in permitted_ids:
            raise ValueError(f"Inquiry source {record_id} is outside its captured grant.")
        if record_id in visiting:
            raise ValueError("Private source lineage contains a cycle.")
        if record_id in visited:
            return
        record = view.get(record_id)
        if record.scope != scope:
            raise ValueError("Private source lineage cannot cross its captured reality and frame.")
        visiting.add(record_id)
        parents = _source_parents(record)
        if parents is None:
            roots.add(record_id)
        else:
            for parent_id in parents:
                visit(parent_id)
        visiting.remove(record_id)
        visited.add(record_id)

    for source_id in source_ids:
        visit(source_id)
    return tuple(sorted(roots))


def capture_inquiry_grant(view: Snapshot, *, observer_entity_id: int, scope: RecordScope,
                          source_ids: Sequence[int], governing_ids: Sequence[int] = (),
                          critical_epoch: int) -> InquiryGrant:
    """Capture source fingerprints and derivation closure before generation.

    The supplied snapshot already owns the visibility grant. Governing IDs
    require current authority; historical sources retain their original IDs.
    Missing source parents fail instead of widening the snapshot.
    """
    observer = view.get(observer_entity_id)
    if not isinstance(observer, Entity):
        raise ValueError("An inquiry observer must name an existing entity.")
    captured_ids: set[int] = {observer_entity_id, *source_ids, *governing_ids}
    pending = list(captured_ids)
    while pending:
        record_id = pending.pop()
        record = view.get(record_id)
        if record.scope != scope:
            raise ValueError("An inquiry source lies outside the captured reality and frame.")
        parents = _source_parents(record)
        if parents is not None:
            for parent_id in parents:
                if parent_id not in captured_ids:
                    captured_ids.add(parent_id)
                    pending.append(parent_id)
    # Traversal also rejects a cyclic derivation before any model call.
    source_roots(view, tuple(captured_ids), scope=scope, permitted_ids=frozenset(captured_ids))
    governing = set(governing_ids)
    return InquiryGrant(observer_entity_id=observer_entity_id, scope=scope, critical_epoch=critical_epoch,
        sources=tuple(capture_source(view, source_id, current_head=source_id in governing)
                      for source_id in sorted(captured_ids)))


def expect_inquiry_grant(batch: WriteBatch, grant: InquiryGrant) -> None:
    """Stage the exact captured dependencies, without refreshing fingerprints."""
    view = batch.view()
    observer = view.get(grant.observer_entity_id)
    if not isinstance(observer, Entity):
        raise ValueError("An inquiry grant requires its captured observer identity.")
    source_ids = {source.record_id for source in grant.sources}
    if observer.id not in source_ids:
        raise ValueError("The inquiry grant must retain its observer dependency.")
    for source in grant.sources:
        if view.get(source.record_id).scope != grant.scope:
            raise ValueError("An inquiry grant contains a different reality or frame.")
        batch.expect_source(source)
    batch.depend_on(source_ids)


def extend_inquiry_grant(view: Snapshot, retained: InquiryGrant, source_ids: Sequence[int]) -> InquiryGrant:
    """Add newly inspected evidence while preserving every original fingerprint.

    An operational revision cannot refresh a retained dependency. A changed
    original meaning still fails at commit. New current authority requires
    fresh admission rather than an upgrade of historical evidence here.
    """
    added = capture_inquiry_grant(view, observer_entity_id=retained.observer_entity_id,
        scope=retained.scope, source_ids=source_ids, critical_epoch=retained.critical_epoch)
    sources = {source.record_id: source for source in added.sources}
    for source in retained.sources:
        sources[source.record_id] = source
    return InquiryGrant(observer_entity_id=retained.observer_entity_id, scope=retained.scope,
        sources=tuple(sources[record_id] for record_id in sorted(sources)), critical_epoch=retained.critical_epoch)


def read_nomination(view: Snapshot, record_id: int) -> NominationRecord:
    """Read a proposal and reject contradictory identity or epistemic metadata."""
    record = view.get(record_id)
    if (not isinstance(record, Feature) or record.feature_type != FeatureType.Thought
            or record.provenance is None or record.provenance.source_kind != SourceKind.Self
            or record.interlocus != InterlocusType.PrivateInternal):
        raise ValueError("An inquiry origin must be a private self-origin nomination.")
    receipt = NominationMetadata.model_validate(record.metadata).thought_nomination
    if (record.entity_id != receipt.grant.observer_entity_id or record.scope != receipt.grant.scope
            or record.content != receipt.nomination.text or record.lineage is None
            or record.lineage.operation != "nomination" or record.lineage.reality != receipt.nomination.reality
            or record.lineage.source_event_ids != list(receipt.nomination.evidence_ids)):
        raise ValueError("A nomination record contradicts its retained proposal.")
    roots = source_roots(view, receipt.nomination.evidence_ids, scope=receipt.grant.scope,
                         permitted_ids=frozenset(source.record_id for source in receipt.grant.sources))
    if roots != receipt.root_source_ids:
        raise ValueError("A nomination contradicts its original source roots.")
    return NominationRecord(record, receipt)


def stage_nominations(batch: WriteBatch, nominations: Sequence[ThoughtNomination], *,
                      grant: InquiryGrant, batch_id: str) -> tuple[NominationRecord, ...]:
    """Stage up to three proposals; leave attention, GWT, and commit to callers.

    An empty batch is valid. This function returns no fabricated thought.
    The parent host must retain its empty-stage receipt to avoid regeneration.
    """
    if len(nominations) > 3:
        raise ValueError("One private nomination batch permits at most three proposals.")
    if not batch_id.strip():
        raise ValueError("A nomination batch requires its invocation identity.")
    expect_inquiry_grant(batch, grant)
    view = batch.view()
    permitted = frozenset(source.record_id for source in grant.sources)
    receipts: list[NominationReceipt] = []
    # Validate the whole batch before allocating any proposal records.
    for nomination in nominations:
        validate_operation(nomination.suggested_operation)
        roots = source_roots(view, nomination.evidence_ids, scope=grant.scope, permitted_ids=permitted)
        receipts.append(NominationReceipt(nomination=nomination, grant=grant, root_source_ids=roots))
    staged: list[NominationRecord] = []
    for receipt in receipts:
        nomination = receipt.nomination
        record = batch.create(Feature(content=nomination.text, feature_type=FeatureType.Thought,
            entity_id=grant.observer_entity_id, scope=grant.scope,
            provenance=Provenance(source_kind=SourceKind.Self), interlocus=InterlocusType.PrivateInternal,
            lineage=RecordLineage(episode_id=batch_id, operation="nomination",
                source_event_ids=list(nomination.evidence_ids), reality=nomination.reality),
            metadata={NOMINATION_KEY: receipt.model_dump(mode="json")}))
        staged.append(NominationRecord(record, receipt))
    return tuple(staged)
