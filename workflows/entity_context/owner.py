"""Accept attributed entity claims through one append-only staging boundary.

The owner validates proposals from immediate maintenance, main cognition, and
later inquiry clients. It owns interpretation acceptance, not identity, model
inference, action selection, or speech permission. MemoryStore owns all durable
records. A caller holds the store lock through staging and commit, without a
model call inside that boundary.

Each private StateUpdate receipt names one observer/subject/scope family and
its preceding receipt. The receipt stream retains reports, inferred claims,
corrections, and challenges. Immediate hosts can preserve their append-only
batch on recovery; main hosts can retain valid appraisal while recording an
explicit deferred entity proposal. No record mutation or independent commit
occurs here. Source changes and stale parents have typed outcomes. Corrupt
chains and invalid internal contracts fail loudly.

Reading map: schemas defines model drafts and accepted contracts; projection
folds this stream into compact current context and compares relevant meaning.
The conceptual manual is docs/subsystems/entity-context.md. Engineering choices
and verification remain in dev/work_items/beta-b04.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from pydantic import BaseModel, ConfigDict

from pm.memory.dependencies import source_fingerprint
from pm.memory.evidence import EvidenceMetadata
from pm.memory.snapshot import Snapshot
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordBase
from pm.records.enums import FeatureType, FrameEnum, InterlocusType, SourceKind
from pm.records.lineage import RecordLineage
from pm.records.memory_records import Entity, Feature
from pm.records.world_records import BodyEpisodeRecord, WorldObjectRecord, WorldPlaceRecord
from workflows.entity_context.schemas import (
    ENTITY_CONTEXT_FAMILY, ENTITY_CONTEXT_RECEIPT, EntityClaimEvidence,
    EntityContextClaim, EntityContextKey, EntityContextProposal,
    EntityContextReceipt, EntityUpdateResult,
)


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
    claims: dict[str, EntityContextClaim]
    challenged_claim_ids: set[str]


def family_key(key: EntityContextKey) -> str:
    """Return the stable metadata guard value for this exact family."""
    serialized = json.dumps(key.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def receipt_chain(view: Snapshot, key: EntityContextKey) -> list[EntityReceiptRecord]:
    """Read one complete family chain and fail on forks or missing parents."""
    expected_family = family_key(key)
    records: dict[int, EntityReceiptRecord] = {}
    for record in view.all_knoxels.values():
        if record.metadata.get(ENTITY_CONTEXT_FAMILY) != expected_family:
            continue
        metadata = EntityReceiptMetadata.model_validate(record.metadata)
        if (not isinstance(record, Feature) or record.feature_type != FeatureType.StateUpdate
                or record.scope != key.scope or record.entity_id != key.subject_entity_id
                or metadata.entity_context_receipt.proposal.key != key):
            raise ValueError("An entity-context receipt contradicts its family identity.")
        records[record.id] = EntityReceiptRecord(record, metadata.entity_context_receipt)
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
    receipts = receipt_chain(view, key)
    claims: dict[str, EntityContextClaim] = {}
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
    return EntityModelState(receipts, claims, challenged)


def active_claims(view: Snapshot, *, key: EntityContextKey) -> list[EntityContextClaim]:
    """Return the complete active family for exact correction-target resolution."""
    return list(model_state(view, key).claims.values())


def claim_identity(proposal_id: str, position: int) -> str:
    """Assign a claim identity on the host, rather than ask a model to invent one."""
    return hashlib.sha256(f"{proposal_id}:{position}".encode()).hexdigest()


class EntityContextOwner:
    """Validate and stage accepted entity interpretation in its caller's transaction."""

    def __init__(self, store: MemoryStore) -> None:
        self.store: MemoryStore = store

    def current_head(self, view: Snapshot, key: EntityContextKey) -> int | None:
        """Return the exact family parent, including receipts staged in this view."""
        chain = receipt_chain(view, key)
        return chain[-1].record.id if chain else None

    def stage_update(self, batch: WriteBatch, proposal: EntityContextProposal) -> EntityUpdateResult:
        """Stage a private append, or return an explicit expected rejection.

        The host must keep the store lock through this call and commit when
        partial main acceptance matters. The family guard also prevents an
        unlocked caller from silently appending a competing branch.
        """
        batch.check_cancelled()
        if batch.store is not self.store:
            raise ValueError("An entity owner cannot stage updates in another memory store.")
        with self.store.lock:
            live_view = self.store.snapshot(capture=False)
            batch_view = batch.view()
            records = dict(live_view.all_knoxels)
            for record_id in batch.created_ids:
                records[record_id] = batch_view.get(record_id)
            effective_view = Snapshot(records, {}, dict(live_view.revisions),
                                      commit_ids=live_view.commit_ids,
                                      historical_records=live_view.historical_records)
            state = model_state(effective_view, proposal.key)
            parent = state.receipts[-1].record.id if state.receipts else None
            for previous in state.receipts:
                if previous.receipt.proposal.proposal_id == proposal.proposal_id:
                    normalized = self._assign_claim_ids(proposal)
                    if normalized != previous.receipt.proposal:
                        raise ValueError("An entity proposal identity cannot name different interpretations.")
                    return EntityUpdateResult(proposal_id=proposal.proposal_id, disposition="staged", reason="accepted",
                                              current_parent_receipt_id=parent, staged_receipt_id=previous.record.id)
            if proposal.parent_receipt_id != parent:
                return EntityUpdateResult(proposal_id=proposal.proposal_id, disposition="deferred", reason="stale_parent",
                                          current_parent_receipt_id=parent, staged_receipt_id=None)
            if not self._valid_observer(effective_view, proposal.key):
                return self._unsupported(proposal, parent)
            targets = set(proposal.retract_claim_ids) | set(proposal.challenge_claim_ids)
            if not targets.issubset(state.claims):
                return self._unsupported(proposal, parent)
            if proposal.retract_claim_ids and not proposal.correction_evidence:
                return self._unsupported(proposal, parent)
            evidence = list(proposal.correction_evidence)
            for claim in proposal.claims:
                evidence.extend(claim.evidence)
            for item in evidence:
                current = effective_view.get_knoxel_by_id(item.source.record_id)
                if (current is None
                        or item.source.current_head and item.source.record_id not in effective_view.all_knoxels
                        or source_fingerprint(current) != item.source.fingerprint):
                    return EntityUpdateResult(proposal_id=proposal.proposal_id, disposition="deferred", reason="changed_source",
                                              current_parent_receipt_id=parent, staged_receipt_id=None)
                if current.scope != proposal.key.scope:
                    return self._unsupported(proposal, parent)
                if (isinstance(current, Feature) and current.feature_type == FeatureType.Dialogue
                        and current.entity_id != proposal.key.observer_entity_id
                        and current.addressee_entity_id not in {0, proposal.key.observer_entity_id}):
                    return self._unsupported(proposal, parent)
                if item.exact_quote is not None and item.exact_quote not in current.content:
                    return self._unsupported(proposal, parent)
            if proposal.retract_claim_ids and not any(
                    self._report_evidence(effective_view, item, proposal.key)
                    or self._observation_source(effective_view.get(item.source.record_id))
                    for item in proposal.correction_evidence):
                return self._unsupported(proposal, parent)
            for claim in proposal.claims:
                if not self._valid_claim(effective_view, proposal.key, claim, state):
                    return self._unsupported(proposal, parent)
            normalized = self._assign_claim_ids(proposal)
            # No guard or record is staged before all expected dispositions are
            # known. A stale proposal therefore cannot poison valid appraisal.
            family_revisions = {item.record.id: live_view.revisions[item.record.id]
                                for item in receipt_chain(live_view, proposal.key)}
            batch.expect_metadata_version(ENTITY_CONTEXT_FAMILY, family_key(proposal.key), family_revisions)
            for item in evidence:
                batch.expect_source(item.source)
            batch.depend_on_commits(proposal.input_commit_ids)
            source_ids = sorted({item.source.record_id for item in evidence})
            batch.depend_on(source_ids)
            if parent is not None:
                batch.depend_on([parent])
            receipt = EntityContextReceipt(proposal=normalized)
            record = batch.create(Feature(
                content="Entity context: " + "; ".join(claim.statement for claim in normalized.claims),
                feature_type=FeatureType.StateUpdate, interlocus=InterlocusType.PrivateReportable,
                causal=True, entity_id=proposal.key.subject_entity_id, scope=proposal.key.scope,
                provenance=Provenance(source_kind=SourceKind.Workflow, source_ref="entity_context"),
                metadata={ENTITY_CONTEXT_FAMILY: family_key(proposal.key),
                          ENTITY_CONTEXT_RECEIPT: receipt.model_dump(mode="json")},
                lineage=RecordLineage(episode_id=family_key(proposal.key), operation="entity_context.update",
                    parent_record_ids=[parent] if parent is not None else [], source_event_ids=source_ids),
            ))
            return EntityUpdateResult(proposal_id=proposal.proposal_id, disposition="staged", reason="accepted",
                                      current_parent_receipt_id=parent, staged_receipt_id=record.id)

    def _assign_claim_ids(self, proposal: EntityContextProposal) -> EntityContextProposal:
        """Keep host-derived claim identities stable across acceptance and replay."""
        claims: list[EntityContextClaim] = []
        for position, claim in enumerate(proposal.claims):
            expected = claim_identity(proposal.proposal_id, position)
            if claim.claim_id and claim.claim_id != expected:
                raise ValueError("A staged claim must use its host-derived proposal identity.")
            claims.append(claim.model_copy(update={"claim_id": expected}))
        return proposal.model_copy(update={"claims": claims})

    def _unsupported(self, proposal: EntityContextProposal, parent: int | None) -> EntityUpdateResult:
        """Return the explicit interpretation rejection without staging side effects."""
        return EntityUpdateResult(proposal_id=proposal.proposal_id, disposition="rejected", reason="unsupported_interpretation",
                                  current_parent_receipt_id=parent, staged_receipt_id=None)

    def _valid_observer(self, view: Snapshot, key: EntityContextKey) -> bool:
        """The initial producer writes only this runtime's configured observer."""
        subject = view.get(key.subject_entity_id)
        observer = view.get(key.observer_entity_id)
        if not isinstance(subject, Entity) or not isinstance(observer, Entity):
            raise ValueError("Entity-context subject and observer must be identity anchors.")
        return observer.content == self.store.ghost_config.companion_name

    def _report_evidence(self, view: Snapshot, evidence: EntityClaimEvidence, key: EntityContextKey) -> bool:
        """A live attributed utterance supports a report, never independent truth."""
        record = view.get(evidence.source.record_id)
        return (isinstance(record, Feature) and record.feature_type == FeatureType.Dialogue
                and record.causal and record.interlocus == InterlocusType.Public
                and record.provenance is not None and record.provenance.source_kind in {SourceKind.User, SourceKind.Agent}
                and record.entity_id != key.observer_entity_id
                and record.addressee_entity_id in {0, key.observer_entity_id}
                and evidence.exact_quote is not None)

    def _valid_claim(self, view: Snapshot, key: EntityContextKey, claim: EntityContextClaim,
                     state: EntityModelState) -> bool:
        """Validate epistemic role and support without a lexical situation classifier."""
        sources = [view.get(item.source.record_id) for item in claim.evidence]
        if claim.observed_at_utc != max(record.timestamp_world_begin for record in sources):
            raise ValueError("The host must derive entity observation time from its original sources.")
        if claim.perspective_entity_id is not None and not isinstance(view.get(claim.perspective_entity_id), Entity):
            raise ValueError("An attributed perspective must identify an Entity.")
        if claim.epistemic_kind == "report" and not any(
                self._report_evidence(view, item, key) for item in claim.evidence):
            return False
        if claim.epistemic_kind == "observation":
            for source in sources:
                if not self._observation_source(source):
                    return False
        if claim.interaction is not None and claim.epistemic_kind == "hypothetical":
            return False
        source_ids = {item.source.record_id for item in claim.evidence}
        for previous in state.claims.values():
            if (previous.claim_key == claim.claim_key and previous.statement == claim.statement
                    and claim.epistemic_kind == "inference" and claim.confidence > previous.confidence
                    and source_ids.issubset({item.source.record_id for item in previous.evidence})):
                return False
        return True

    def _observation_source(self, source: RecordBase) -> bool:
        """Require runtime observation provenance before accepting an observed claim."""
        if (source.scope.frame != FrameEnum.InWorld or source.provenance is None
                or source.provenance.source_kind not in {SourceKind.Workflow, SourceKind.System}):
            return False
        if isinstance(source, (BodyEpisodeRecord, WorldObjectRecord, WorldPlaceRecord)):
            return True
        return (isinstance(source, Feature) and source.feature_type == FeatureType.WorldEvent and source.causal
                and EvidenceMetadata.model_validate(source.metadata).memory_kind == "executed_event")
