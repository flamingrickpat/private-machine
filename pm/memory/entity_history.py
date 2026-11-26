"""Read the accepted entity-claim ledger shared by cognition and memory consolidation.

A receipt belongs to one subject, observer, reality and frame. Its parent chain
preserves explicit corrections and challenges. The workflow owns acceptance.
This module owns the persisted read contracts and deterministic interpretation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pm.memory.dependencies import SourceDependency
from pm.memory.snapshot import Snapshot
from pm.records.base import RecordBase, RecordScope
from pm.records.enums import FeatureType, SourceKind
from pm.records.memory_records import Entity, Feature


EntityClaimLayer = Literal["background", "relationship", "pattern", "situation", "perspective", "open_question"]
EntityEpistemicKind = Literal["observation", "report", "inference", "hypothetical"]
EntityProducer = Literal["immediate", "main", "thought", "tom", "migration"]
EntityApplicability = Literal["current", "due_for_review", "expired", "future"]
ENTITY_CONTEXT_FAMILY = "entity_context_family"
ENTITY_CONTEXT_RECEIPT = "entity_context_receipt"


class EntityContextKey(BaseModel):
    """One subject as understood by one observer in one reality and frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    subject_entity_id: int = Field(gt=0, description="Identity anchor of the entity this model describes.")
    observer_entity_id: int = Field(gt=0, description="Identity anchor of the observer holding this interpretation.")
    scope: RecordScope = Field(description="Reality and frame in which these claims apply.")


class EntityClaimEvidence(BaseModel):
    """Original source meaning and an optional verbatim report quotation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: SourceDependency
    exact_quote: str | None = Field(default=None, min_length=1, description="Exact source substring when this evidence is a report.")


class InteractionClaim(BaseModel):
    """An evidence-qualified ability estimate, never permission to contact."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    dimension: Literal["receive", "respond"]
    ability: Literal["can", "cannot", "unknown"]


class EntityContextClaim(BaseModel):
    """One attributed claim whose historical truth and current applicability differ."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    claim_id: str = Field(default="", description="Owner-assigned immutable claim identity; empty only before staging.")
    layer: EntityClaimLayer
    claim_key: str = Field(min_length=1, description="Exact proposition key within this observer and subject family.")
    statement: str = Field(min_length=1, description="The attributed assertion, without promoting inference to observation.")
    epistemic_kind: EntityEpistemicKind
    perspective_entity_id: int | None = Field(default=None, gt=0, description="Entity whose knowledge is attributed, distinct from the observer.")
    confidence: float = Field(ge=0.0, le=1.0, description="Support for this epistemic assertion, not proof of a report's truth.")
    evidence: list[EntityClaimEvidence] = Field(min_length=1)
    observed_at_utc: datetime = Field(description="Latest cited source world time, as naive UTC.")
    applicable_from_utc: datetime = Field(description="Beginning of the asserted applicable interval, as naive UTC.")
    review_after_utc: datetime | None = Field(default=None, description="Time to weaken current applicability, never infer the opposite.")
    applicable_until_utc: datetime | None = Field(default=None, description="Explicit end of the asserted interval, if evidence supplies one.")
    interaction: InteractionClaim | None = None

    @model_validator(mode="after")
    def valid_times(self) -> Self:
        """Reject mixed clock conventions and impossible explicit intervals."""
        if not self.claim_key.strip() or not self.statement.strip():
            raise ValueError("Entity claims require a nonempty proposition key and assertion.")
        for timestamp in (self.observed_at_utc, self.applicable_from_utc, self.review_after_utc, self.applicable_until_utc):
            if timestamp is not None and timestamp.tzinfo is not None:
                raise ValueError("Entity-context timestamps must use naive UTC world time.")
        if self.applicable_until_utc is not None and self.applicable_until_utc < self.applicable_from_utc:
            raise ValueError("An entity claim cannot end before its applicable interval starts.")
        return self


class EntityContextProposal(BaseModel):
    """Frozen host-hydrated interpretation with an explicit expected family parent."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    proposal_id: str = Field(min_length=1)
    key: EntityContextKey
    parent_receipt_id: int | None
    producer: EntityProducer
    input_commit_ids: list[str] = Field(default_factory=list[str])
    claims: list[EntityContextClaim] = Field(default_factory=list[EntityContextClaim], max_length=4)
    retract_claim_ids: list[str] = Field(default_factory=list[str])
    challenge_claim_ids: list[str] = Field(default_factory=list[str])
    correction_evidence: list[EntityClaimEvidence] = Field(default_factory=list[EntityClaimEvidence])


class EntityContextReceipt(BaseModel):
    """Validated payload of one append-only private state-update feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["pm.entity-context.v1"] = "pm.entity-context.v1"
    proposal: EntityContextProposal


class EntitySourceAttribution(BaseModel):
    """The actual source speaker and addressee, independent of the claim's subject."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    record_id: int
    source_kind: SourceKind | None
    speaker_entity_id: int | None
    speaker: str
    addressee_entity_id: int | None


class EntityClaimView(BaseModel):
    """A retained claim with time-sensitive applicability and competing-claim state."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    claim: EntityContextClaim
    applicability: EntityApplicability
    contested: bool
    attributions: list[EntitySourceAttribution] = Field(default_factory=list[EntitySourceAttribution])


class EntityReceiptMetadata(BaseModel):
    """The validated entity-context portion of heterogeneous record metadata."""

    model_config = ConfigDict(extra="ignore")
    entity_context_family: str
    entity_context_receipt: EntityContextReceipt


@dataclass(frozen=True, slots=True)
class EntityReceiptRecord:
    """One accepted or staged receipt with its actual memory identity."""

    record: Feature
    receipt: EntityContextReceipt


@dataclass(slots=True)
class EntityModelState:
    """Derived current claims and explicit challenges, never a second authority."""

    receipts: list[EntityReceiptRecord]
    claims: dict[str, EntityContextClaim]  # pm-lint: allow PM004 -- accepted claims by immutable claim ID
    challenged_claim_ids: set[str]


def family_key(key: EntityContextKey) -> str:
    """Return the stable metadata guard value for this exact family."""
    return _digest(key)


def receipt_chain(view: Snapshot, key: EntityContextKey) -> list[EntityReceiptRecord]:
    """Read one complete family chain and fail on forks or missing parents."""
    expected_family = family_key(key)
    records: dict[int, EntityReceiptRecord] = {}  # pm-lint: allow PM004 -- receipt records by knoxel ID
    for record in view.all_knoxels.values():
        if record.metadata.get(ENTITY_CONTEXT_FAMILY) != expected_family:
            continue
        item = read_entity_receipt(record)
        if item.receipt.proposal.key != key:
            raise ValueError("An entity-context receipt contradicts its expected family.")
        records[record.id] = item
    return order_receipts(records)  # pm-lint: allow PM005 -- receipt records indexed by their knoxel IDs


def read_entity_receipt(record: RecordBase) -> EntityReceiptRecord:
    """Validate the receipt's subject, scope, record kind and persisted family key."""
    metadata = EntityReceiptMetadata.model_validate(record.metadata)
    receipt = metadata.entity_context_receipt
    key = receipt.proposal.key
    if (not isinstance(record, Feature) or record.feature_type != FeatureType.StateUpdate
            or record.scope != key.scope or record.entity_id != key.subject_entity_id
            or metadata.entity_context_family != family_key(key)):
        raise ValueError(f"Entity-context receipt {record.id} contradicts its family identity.")
    return EntityReceiptRecord(record, receipt)


def order_receipts(records: dict[int, EntityReceiptRecord]) -> list[EntityReceiptRecord]:  # pm-lint: allow PM004 -- parent pointers reference receipt knoxel IDs
    """Return one validated parent chain in publication order."""
    if not records:
        return []
    parent_ids = {item.receipt.proposal.parent_receipt_id for item in records.values()}
    heads = [item for record_id, item in records.items() if record_id not in parent_ids]
    if len(heads) != 1:
        raise ValueError("An entity-context family must have exactly one receipt head.")
    chain: list[EntityReceiptRecord] = []
    visited: set[int] = set()
    current = heads[0]
    while True:
        if current.record.id in visited:
            raise ValueError("An entity-context receipt chain contains a cycle.")
        visited.add(current.record.id)
        chain.append(current)
        parent_id = current.receipt.proposal.parent_receipt_id
        if parent_id is None:
            break
        current = records[parent_id]
    if len(visited) != len(records):
        raise ValueError("An entity-context family contains a disconnected receipt.")
    chain.reverse()
    return chain


def model_state(view: Snapshot, key: EntityContextKey) -> EntityModelState:
    """Fold explicit corrections and challenges without rewriting old evidence."""
    return fold_receipts(receipt_chain(view, key))


def fold_receipts(receipts: list[EntityReceiptRecord]) -> EntityModelState:
    """Apply a complete ordered family history, including explicit retractions."""
    claims: dict[str, EntityContextClaim] = {}  # pm-lint: allow PM004 -- active claims indexed by claim ID
    challenged: set[str] = set()
    known_ids: set[str] = set()
    for item in receipts:
        proposal = item.receipt.proposal
        for claim_id in proposal.retract_claim_ids:
            del claims[claim_id]
            challenged.discard(claim_id)
        for claim_id in proposal.challenge_claim_ids:
            if claim_id not in claims:
                raise ValueError("An entity challenge must name an active claim in its own family.")
            challenged.add(claim_id)
        for claim in proposal.claims:
            if not claim.claim_id or claim.claim_id in known_ids:
                raise ValueError("Accepted entity claims require unique owner-assigned identities.")
            known_ids.add(claim.claim_id)
            claims[claim.claim_id] = claim
    return EntityModelState(receipts, claims, challenged)  # pm-lint: allow PM005 -- active claims indexed by claim ID


class _AttributionMeaning(BaseModel):
    """Source identity semantics without incidental support-record identity."""

    model_config = ConfigDict(frozen=True)
    source_kind: SourceKind | None
    speaker_entity_id: int | None
    speaker: str
    addressee_entity_id: int | None


class _ClaimMeaning(BaseModel):
    """Fields whose change can alter the supplied assertion or its authority."""

    model_config = ConfigDict(frozen=True)
    claim_key: str
    layer: EntityClaimLayer
    statement: str
    epistemic_kind: EntityEpistemicKind
    perspective_entity_id: int | None
    confidence: float
    applicability: EntityApplicability
    applicable_from_utc: datetime | None
    applicable_until_utc: datetime | None
    interaction: InteractionClaim | None
    attributions: list[_AttributionMeaning]


def _digest(value: BaseModel) -> str:
    """Hash a named meaning contract at the serialization boundary."""
    serialized = json.dumps(value.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)  # pm-lint: allow PM007 -- persisted family and meaning hashes require canonical JSON
    return hashlib.sha256(serialized.encode()).hexdigest()


def claim_applicability(claim: EntityContextClaim, as_of_utc: datetime) -> EntityApplicability:
    """Classify the stated interval without inferring an unobserved replacement."""
    if as_of_utc < claim.applicable_from_utc:
        return "future"
    if claim.applicable_until_utc is not None and as_of_utc >= claim.applicable_until_utc:
        return "expired"
    if claim.review_after_utc is not None and as_of_utc >= claim.review_after_utc:
        return "due_for_review"
    return "current"


def _attribution(view: Snapshot, record: RecordBase) -> EntitySourceAttribution:
    """Read known record shapes without treating the subject as the source speaker."""
    source_kind = record.provenance.source_kind if record.provenance is not None else None
    if isinstance(record, Feature):
        speaker = record.override_source or "unknown source"
        if record.entity_id:
            identity = view.get(record.entity_id)
            if not isinstance(identity, Entity):
                raise ValueError("An entity source speaker must refer to an Entity identity.")
            speaker = record.override_source or identity.content
        return EntitySourceAttribution(record_id=record.id, source_kind=source_kind,
            speaker_entity_id=record.entity_id or None, speaker=speaker,
            addressee_entity_id=record.addressee_entity_id or None)
    return EntitySourceAttribution(record_id=record.id, source_kind=source_kind,
        speaker_entity_id=None, speaker=source_kind.value if source_kind is not None else "unknown source",
        addressee_entity_id=None)


def _meaning(item: EntityClaimView) -> _ClaimMeaning:
    """Exclude repeat-support IDs and derived review clocks, while keeping attribution."""
    attributions: dict[str, _AttributionMeaning] = {}  # pm-lint: allow PM004 -- unique source attributions by meaning digest
    for source in item.attributions:
        attribution = _AttributionMeaning(source_kind=source.source_kind, speaker_entity_id=source.speaker_entity_id,
            speaker=source.speaker, addressee_entity_id=source.addressee_entity_id)
        attributions[_digest(attribution)] = attribution
    ordered_attributions = [attributions[digest] for digest in sorted(attributions.keys())]
    claim = item.claim
    return _ClaimMeaning(claim_key=claim.claim_key, layer=claim.layer,
        statement=" ".join(claim.statement.split()), epistemic_kind=claim.epistemic_kind,
        perspective_entity_id=claim.perspective_entity_id, confidence=claim.confidence,
        applicability=item.applicability,
        applicable_from_utc=claim.applicable_from_utc if item.applicability == "future" else None,
        applicable_until_utc=claim.applicable_until_utc, interaction=claim.interaction,
        attributions=ordered_attributions)


def claim_meaning_id(item: EntityClaimView) -> str:
    """Identify equivalent support independently of bookkeeping and challenge status."""
    return _digest(_meaning(item))


def claim_views(view: Snapshot, claims: Sequence[EntityContextClaim], challenged_ids: set[str],
                as_of_utc: datetime) -> list[EntityClaimView]:
    """Resolve applicability and competing current assertions for this family and clock."""
    views: list[EntityClaimView] = []
    current_meanings: dict[str, set[str]] = {}  # pm-lint: allow PM004 -- competing meaning IDs by proposition key
    for claim in claims:
        sources = [_attribution(view, view.get(item.source.record_id)) for item in claim.evidence]
        entry = EntityClaimView(claim=claim, applicability=claim_applicability(claim, as_of_utc),
                                contested=claim.claim_id in challenged_ids, attributions=sources)
        views.append(entry)
        if entry.applicability == "current":
            current_meanings.setdefault(claim.claim_key, set()).add(claim_meaning_id(entry))
    for index, entry in enumerate(views):
        if entry.applicability == "current" and len(current_meanings[entry.claim.claim_key]) > 1:
            views[index] = EntityClaimView(claim=entry.claim, applicability=entry.applicability,
                                           contested=True, attributions=entry.attributions)
    return views
