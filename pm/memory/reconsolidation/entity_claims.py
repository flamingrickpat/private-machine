"""Resolve entity-linked memory against the observer's accepted claim ledger.

The episode keeps the original assertion and its receipt. A later correction
changes its eligibility for current recall. Historical citations keep resolving
to the original records. The entity-context workflow remains the writer.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from pm.memory.entity_history import (
    ENTITY_CONTEXT_FAMILY, EntityEpistemicKind, EntityReceiptRecord, claim_views,
    fold_receipts, order_receipts, read_entity_receipt,
)
from pm.memory.snapshot import Snapshot
from pm.records.base import RecordBase, RecordScope
from pm.records.memory_records import Feature

EntityMemoryStatus = Literal["current", "due_for_review", "future", "expired", "retracted", "contested"]


@dataclass(frozen=True, slots=True)
class EntityMemoryClaim:
    """One historical claim, the family head consulted, and its present authority."""

    claim_id: str
    receipt_id: int
    head_receipt_id: int
    statement: str
    claim_key: str
    epistemic_kind: EntityEpistemicKind
    status: EntityMemoryStatus
    subject_entity_id: int
    observer_entity_id: int
    scope: RecordScope
    source_ids: tuple[int, ...]
    quotes: tuple[str, ...]


class FiledEntityClaim(BaseModel):
    """The authority recorded on an occurrence or its canonical memory."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    entity_claim_id: str
    entity_claim_receipt_id: int
    entity_claim_head_id: int
    entity_claim_status: EntityMemoryStatus


type ReceiptFamily = dict[int, EntityReceiptRecord]  # pm-lint: allow PM004 -- receipts indexed by knoxel ID


@dataclass(slots=True)
class EntityMemoryClaims:
    """Look up attributed interpretations by the source event they interpret."""

    by_source: dict[int, list[EntityMemoryClaim]] = field(default_factory=dict[int, list[EntityMemoryClaim]])  # pm-lint: allow PM004 -- interpretations indexed by evidence knoxel ID
    by_claim_id: dict[str, EntityMemoryClaim] = field(default_factory=dict[str, EntityMemoryClaim])  # pm-lint: allow PM004 -- ledger entries by immutable claim ID

    @classmethod
    def from_snapshot(cls, view: Snapshot, observer_id: int, now: datetime) -> EntityMemoryClaims:
        """Read all accepted families of one observer at the frozen world time."""
        result = cls()
        families: dict[str, ReceiptFamily] = {}  # pm-lint: allow PM004 -- receipt chains by subject-observer family digest
        for record in view.all_knoxels.values():
            if ENTITY_CONTEXT_FAMILY not in record.metadata:
                continue
            row = read_entity_receipt(record)
            if row.receipt.proposal.key.observer_entity_id == observer_id:
                family = record.metadata[ENTITY_CONTEXT_FAMILY]
                if family not in families:
                    families[family] = {}
                families[family][record.id] = row
        for records in families.values():
            result.add_family(view, records, now)
        return result

    def add_family(self, view: Snapshot, records: ReceiptFamily, now: datetime) -> None:
        """Interpret one complete family with the same conflict rules used by cognition."""
        state = fold_receipts(order_receipts(records))
        head = state.receipts[-1].record.id
        active = claim_views(view, list(state.claims.values()), state.challenged_claim_ids, now)
        statuses: dict[str, EntityMemoryStatus] = {}  # pm-lint: allow PM004 -- applicability indexed by immutable claim ID
        for item in active:
            status = item.applicability
            statuses[item.claim.claim_id] = "contested" if item.contested and status == "current" else status
        for row in state.receipts:
            family = row.receipt.proposal.key
            for claim in row.receipt.proposal.claims:
                source_ids = tuple(evidence.source.record_id for evidence in claim.evidence)
                quotes = tuple(evidence.exact_quote for evidence in claim.evidence if evidence.exact_quote is not None)
                memory = EntityMemoryClaim(
                    claim_id=claim.claim_id, receipt_id=row.record.id, head_receipt_id=head,
                    statement=claim.statement, claim_key=claim.claim_key, epistemic_kind=claim.epistemic_kind,
                    status=statuses.get(claim.claim_id, "retracted"), subject_entity_id=family.subject_entity_id,
                    observer_entity_id=family.observer_entity_id, scope=family.scope, source_ids=source_ids, quotes=quotes,
                )
                self.by_claim_id[memory.claim_id] = memory
                for record_id in source_ids:
                    self.by_source.setdefault(record_id, []).append(memory)

    def for_sources(self, source_ids: Sequence[int], scope: RecordScope, subject_id: int) -> tuple[EntityMemoryClaim, ...]:
        """Return this subject's interpretations citing any of these episode sources."""
        matches: dict[str, EntityMemoryClaim] = {}  # pm-lint: allow PM004 -- distinct interpretations by claim ID
        for record_id in source_ids:
            for claim in self.by_source.get(record_id, ()):
                if claim.subject_entity_id == subject_id and claim.scope == scope:
                    matches[claim.claim_id] = claim
        return tuple(matches.values())


def family_is_visible(records: ReceiptFamily, eligible: Mapping[int, RecordBase]) -> bool:  # pm-lint: allow PM004 -- allowed source records by knoxel ID
    """Decide whether this read grant contains the complete ancestry and its evidence."""
    for row in records.values():
        proposal = row.receipt.proposal
        if proposal.parent_receipt_id is not None and proposal.parent_receipt_id not in records:
            return False
        for claim in proposal.claims:
            for evidence in claim.evidence:
                source = eligible.get(evidence.source.record_id)
                if source is None:
                    return False
                # Attribution reads the source speaker's identity, which may be
                # outside a narrower grant even when the quoted line is visible.
                if isinstance(source, Feature) and source.entity_id and source.entity_id not in eligible:
                    return False
    return True


def outdated_entity_memories(view: Snapshot, eligible: Mapping[int, RecordBase], now: datetime) -> set[int]:  # pm-lint: allow PM004 -- original read grant by knoxel ID
    """Hide derived assertions whose authority changed within the permitted read frontier.

    A partial read grant may omit a receipt, a parent, or the source used by a
    sibling assertion. In that case this grant cannot establish current authority.
    The raw historical episode and dialogue remain available through their own grants.
    """
    linked: dict[int, FiledEntityClaim] = {}  # pm-lint: allow PM004 -- recorded authority by derived memory ID
    families: dict[str, ReceiptFamily] = {}  # pm-lint: allow PM004 -- receipt chains by subject-observer family digest
    outdated: set[int] = set()
    for record in eligible.values():
        if "entity_claim_id" not in record.metadata:
            continue
        recorded = FiledEntityClaim.model_validate(record.metadata)
        receipt = eligible.get(recorded.entity_claim_receipt_id)
        if receipt is None or recorded.entity_claim_head_id not in eligible:
            outdated.add(record.id)
            continue
        family = receipt.metadata[ENTITY_CONTEXT_FAMILY]
        if family not in families:
            families[family] = {}
        linked[record.id] = recorded
    if not linked:
        return outdated
    for record in eligible.values():
        family = record.metadata.get(ENTITY_CONTEXT_FAMILY)
        if family in families:
            families[family][record.id] = read_entity_receipt(record)
    claims = EntityMemoryClaims()
    for records in families.values():
        if family_is_visible(records, eligible):
            claims.add_family(view, records, now)
    for record_id, recorded in linked.items():
        claim = claims.by_claim_id.get(recorded.entity_claim_id)
        if claim is None or claim.status in {"expired", "future", "retracted"} or claim.status != recorded.entity_claim_status:
            outdated.add(record_id)
    return outdated
