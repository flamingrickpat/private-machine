"""Stage bounded inquiry graphs through the existing private thought owner.

See :mod:`workflows.inner_thought.host` for the subsystem overview. This module
owns validation and staging, not scheduling, inference, attention policy,
Critical admission, GWT publication, or commit. Its callers must use the
control boundary before committing. The private host binds these functions
through graph_host and graph_execution. Retained linear episodes use the
separate explicit legacy reader.

Each accepted transition appends a typed StateUpdate receipt. Executed
operations use InnerMonologue; branches use private Thought features. Exact
parent guards reject competing continuations. Captured source fingerprints
guard meaning without a global ordinary-input veto. Original sources remain
distinct from internal derivations. No graph node grants external action,
entity-update, or affect authority.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from pydantic import BaseModel, ConfigDict

from pm.memory.snapshot import GhostConflict, Snapshot
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.lineage import RecordLineage
from pm.records.memory_records import Feature
from workflows.cognitive_control.schemas import ControlConfig
from workflows.inner_thought.nominations import expect_inquiry_grant, read_nomination, source_roots
from workflows.inner_thought.operations import validate_operation
from workflows.inner_thought.schemas import (
    InquiryAlternative, InquiryGrant, InquiryReceipt, InquiryState, InquiryStep, LegacyEpisodeState,
    ThoughtAttentionDecision, ThoughtAttentionReceipt,
)


INQUIRY_EPISODE_KEY = "inquiry_episode"
INQUIRY_RECEIPT_KEY = "inquiry_receipt"
PRIVATE_SLOT_KEY = "private_inquiry_slot"
ATTENTION_BATCH_KEY = "thought_attention_batch"


class InquiryMetadata(BaseModel):
    """Deserialize only the named inquiry fields of a memory record."""

    model_config = ConfigDict(extra="ignore")
    inquiry_episode: str
    private_inquiry_slot: str
    inquiry_receipt: InquiryReceipt


class AttentionMetadata(BaseModel):
    """Deserialize a persisted attention decision without untyped internal maps."""

    model_config = ConfigDict(extra="ignore")
    thought_attention_batch: str
    thought_attention: ThoughtAttentionReceipt


class LegacyThoughtMetadata(BaseModel):
    """Validate the old host's episode payload at the storage boundary."""

    model_config = ConfigDict(extra="ignore")
    episode: LegacyEpisodeState


class InquiryNodeMetadata(BaseModel):
    """Read the executed operation payload from its own private record."""

    model_config = ConfigDict(extra="ignore")
    inquiry_step: InquiryStep


class InquiryAlternativeMetadata(BaseModel):
    """Read a branch's independent assumptions, judgments, and disposition."""

    model_config = ConfigDict(extra="ignore")
    inquiry_alternative: InquiryAlternative


@dataclass(frozen=True, slots=True)
class InquiryRecord:
    """The actual receipt record and its validated graph state."""

    record: Feature
    receipt: InquiryReceipt


@dataclass(frozen=True, slots=True)
class AttentionRecord:
    """An attention receipt, optionally allocating the private episode slot."""

    record: Feature
    receipt: ThoughtAttentionReceipt
    inquiry: InquiryRecord | None


@dataclass(frozen=True, slots=True)
class LegacyInquiry:
    """The actual old chain, without synthetic branches or confidence values."""

    records: tuple[Feature, ...]
    state: LegacyEpisodeState


def _slot(grant: InquiryGrant) -> str:
    return str(grant.observer_entity_id)


def read_legacy_inquiry(view: Snapshot, thought_id: int) -> LegacyInquiry:
    """Read old linear lineage without rewriting or executing its checkpoint.

    The returned variant retains text alternatives as text. Missing parents,
    contradictory sequences, and absent episode fields fail visibly. No
    offline activity or active computation duration is inferred.
    """
    records: list[Feature] = []
    visited: set[int] = set()
    current_id: int | None = thought_id
    newest: LegacyEpisodeState | None = None
    while current_id is not None:
        if current_id in visited:
            raise ValueError("A legacy private episode contains a cycle.")
        visited.add(current_id)
        record = view.get(current_id)
        if (not isinstance(record, Feature) or record.feature_type != FeatureType.InnerMonologue
                or record.provenance is None or record.provenance.source_kind != SourceKind.Self
                or record.metadata.get("loop") != "inner" or record.lineage is None):
            raise ValueError("A legacy inquiry requires the original private thought lineage.")
        state = LegacyThoughtMetadata.model_validate(record.metadata).episode
        if newest is None:
            newest = state
        if (state.episode_id != newest.episode_id or record.lineage.episode_id != state.episode_id
                or record.lineage.sequence != state.sequence
                or state.sequence != newest.sequence - len(records)):
            raise ValueError("A legacy private episode contradicts its retained sequence.")
        expected_parents = [state.parent_thought_id] if state.parent_thought_id is not None else []
        if record.lineage.parent_record_ids != expected_parents:
            raise ValueError("A legacy private episode contradicts its retained parent.")
        if records and (record.scope != records[0].scope or record.entity_id != records[0].entity_id):
            raise ValueError("A legacy private episode crosses its observer or reality.")
        records.append(record)
        current_id = state.parent_thought_id
    assert newest is not None
    if len(records) != newest.sequence + 1:
        raise ValueError("A legacy private episode is missing an earlier operation.")
    records.reverse()
    return LegacyInquiry(tuple(records), newest)


def inquiry_history(view: Snapshot, episode_id: str) -> tuple[InquiryRecord, ...]:
    """Read one ordered append-only chain, rejecting forks and disconnected data."""
    records: dict[int, InquiryRecord] = {}
    for record in view.all_knoxels.values():
        if record.metadata.get(INQUIRY_EPISODE_KEY) != episode_id:
            continue
        payload = InquiryMetadata.model_validate(record.metadata)
        receipt = payload.inquiry_receipt
        state = receipt.state
        if (not isinstance(record, Feature) or record.feature_type != FeatureType.StateUpdate
                or record.provenance is None or record.provenance.source_kind != SourceKind.Self
                or record.interlocus != InterlocusType.PrivateInternal
                or state.episode_id != episode_id or state.grant.scope != record.scope
                or state.grant.observer_entity_id != record.entity_id
                or payload.private_inquiry_slot != _slot(state.grant)):
            raise ValueError("An inquiry receipt contradicts its episode or observer.")
        records[record.id] = InquiryRecord(record, receipt)
    if not records:
        return ()
    parent_ids = {item.receipt.parent_receipt_id for item in records.values()}
    heads = [item for record_id, item in records.items() if record_id not in parent_ids]
    if len(heads) != 1:
        raise ValueError("An inquiry must have exactly one receipt head.")
    chain: list[InquiryRecord] = []
    visited: set[int] = set()
    current = heads[0]
    while True:
        if current.record.id in visited:
            raise ValueError("An inquiry receipt chain contains a cycle.")
        visited.add(current.record.id)
        chain.append(current)
        parent = current.receipt.parent_receipt_id
        if parent is None:
            break
        current = records[parent]
    if len(visited) != len(records):
        raise ValueError("An inquiry contains disconnected receipts.")
    chain.reverse()
    for sequence, item in enumerate(chain):
        state = item.receipt.state
        if state.sequence != sequence or len(set(state.node_ids)) != len(state.node_ids):
            raise ValueError("Inquiry sequence and node identities must be unique and ordered.")
        if not set(state.frontier_ids).issubset(state.node_ids):
            raise ValueError("An inquiry frontier must reference accepted nodes.")
        if not set(state.pending_generation_ids).issubset(state.node_ids):
            raise ValueError("Pending discrimination must reference accepted nodes.")
        if not set(state.inspected_root_ids).issubset(state.root_source_ids):
            raise ValueError("Inspected evidence must retain its original roots.")
        receipt = item.receipt
        if sequence == 0:
            if (receipt.step is not None or receipt.operation_record_id is not None or receipt.alternative_record_ids
                    or state.node_ids or receipt.attention is None or receipt.attention.disposition != "select"
                    or receipt.attention.nomination_id != state.origin_record_id):
                raise ValueError("Inquiry selection cannot claim an executed operation.")
        else:
            prior = chain[sequence - 1].receipt.state
            if (receipt.step is None or receipt.operation_record_id is None or receipt.attention is not None
                    or state.origin_record_id != prior.origin_record_id or state.grant.scope != prior.grant.scope
                    or state.grant.observer_entity_id != prior.grant.observer_entity_id
                    or state.grant.critical_epoch != prior.grant.critical_epoch
                    or state.node_ids != (*prior.node_ids, receipt.operation_record_id, *receipt.alternative_record_ids)
                    or state.active_elapsed_seconds < prior.active_elapsed_seconds):
                raise ValueError("An inquiry transition contradicts its retained operation or origin.")
            operation = view.get(receipt.operation_record_id)
            if (not isinstance(operation, Feature) or operation.feature_type != FeatureType.InnerMonologue
                    or InquiryNodeMetadata.model_validate(operation.metadata).inquiry_step != receipt.step
                    or operation.content != receipt.step.thought or operation.lineage is None
                    or operation.lineage.sequence != sequence or operation.lineage.reality != receipt.step.reality):
                raise ValueError("An inquiry operation contradicts its original receipt.")
            if len(receipt.alternative_record_ids) != len(receipt.step.alternatives):
                raise ValueError("An inquiry operation lost one of its independent branches.")
            for branch_id, alternative in zip(receipt.alternative_record_ids, receipt.step.alternatives, strict=True):
                branch = view.get(branch_id)
                if (not isinstance(branch, Feature) or branch.feature_type != FeatureType.Thought
                        or InquiryAlternativeMetadata.model_validate(branch.metadata).inquiry_alternative != alternative
                        or branch.content != alternative.text or branch.lineage is None
                        or branch.lineage.parent_record_ids != [operation.id] or branch.lineage.reality != alternative.reality):
                    raise ValueError("An inquiry branch contradicts its original generation parent.")
        for node_id in state.node_ids:
            node = view.get(node_id)
            if (not isinstance(node, Feature) or node.lineage is None
                    or node.lineage.episode_id != episode_id or node.scope != state.grant.scope
                    or node.entity_id != state.grant.observer_entity_id):
                raise ValueError("An inquiry node contradicts its retained graph.")
    return tuple(chain)


def _guard_family(batch: WriteBatch, key: str, value: str) -> None:
    """Guard committed family membership; require one transition per batch."""
    view = batch.view()
    members = {record.id: view.revisions[record.id]
               for record in view.all_knoxels.values()
               if record.id not in batch.created_ids and record.metadata.get(key) == value}
    if any(view.get(record_id).metadata.get(key) == value for record_id in batch.created_ids):
        raise ValueError("An inquiry transaction stages at most one transition per family.")
    batch.expect_metadata_version(key, value, members)


def _stage_receipt(batch: WriteBatch, receipt: InquiryReceipt) -> InquiryRecord:
    state = receipt.state
    parent_ids = [receipt.parent_receipt_id] if receipt.parent_receipt_id is not None else [state.origin_record_id]
    record = batch.create(Feature(content=state.stop_reason or f"Inquiry {state.episode_id}, step {state.sequence}",
        feature_type=FeatureType.StateUpdate, entity_id=state.grant.observer_entity_id, scope=state.grant.scope,
        provenance=Provenance(source_kind=SourceKind.Self), interlocus=InterlocusType.PrivateInternal,
        lineage=RecordLineage(episode_id=state.episode_id, operation="inquiry_transition",
            sequence=state.sequence, parent_record_ids=parent_ids, source_event_ids=list(state.root_source_ids)),
        metadata={INQUIRY_EPISODE_KEY: state.episode_id, PRIVATE_SLOT_KEY: _slot(state.grant),
                  INQUIRY_RECEIPT_KEY: receipt.model_dump(mode="json")}))
    return InquiryRecord(record, receipt)


def stage_attention(batch: WriteBatch, nomination_ids: Sequence[int], decision: ThoughtAttentionDecision,
                    *, episode_id: str, now_utc: datetime) -> AttentionRecord:
    """Persist the separate attention decision and allocate at most one episode.

    The caller supplies a decision from the shared attention owner. This
    function enforces its identities and bounds; it does not choose a thought.
    Selection creates no executed node or GWT turn.
    """
    if not nomination_ids or len(nomination_ids) > 3 or len(set(nomination_ids)) != len(nomination_ids):
        raise ValueError("Private attention requires one to three unique offered nomination IDs.")
    if now_utc.tzinfo is not None:
        raise ValueError("Internal inquiry timestamps require naive UTC.")
    view = batch.view()
    nominations = [read_nomination(view, record_id) for record_id in nomination_ids]
    first = nominations[0]
    assert first.record.lineage is not None
    batch_id = first.record.lineage.episode_id
    grant = first.receipt.grant
    for nomination in nominations:
        if (nomination.record.lineage is None or nomination.record.lineage.episode_id != batch_id
                or nomination.receipt.grant != grant):
            raise ValueError("One attention decision must use one captured nomination batch.")
        batch.expect_revision(nomination.record.id, view.revisions[nomination.record.id])
    if decision.nomination_id is not None and decision.nomination_id not in nomination_ids:
        raise ValueError("Private attention selected an identity outside its offered nominations.")
    expect_inquiry_grant(batch, grant)
    _guard_family(batch, ATTENTION_BATCH_KEY, batch_id)
    for record in view.all_knoxels.values():
        if record.metadata.get(ATTENTION_BATCH_KEY) != batch_id:
            continue
        earlier = AttentionMetadata.model_validate(record.metadata).thought_attention.decision
        if (earlier.disposition in {"select", "dismiss"}
                and (earlier.nomination_id is None or earlier.nomination_id == decision.nomination_id)):
            raise ValueError("A selected or dismissed nomination cannot be selected again.")
    inquiry: InquiryRecord | None = None
    if decision.disposition == "select":
        if not episode_id.strip():
            raise ValueError("Selection requires a new episode identity.")
        if inquiry_history(view, episode_id):
            raise ValueError("A new inquiry cannot reuse an existing episode identity.")
        _guard_family(batch, INQUIRY_EPISODE_KEY, episode_id)
        _guard_family(batch, PRIVATE_SLOT_KEY, _slot(grant))
        episode_ids = {InquiryMetadata.model_validate(record.metadata).inquiry_episode
                       for record in view.all_knoxels.values()
                       if record.metadata.get(PRIVATE_SLOT_KEY) == _slot(grant)}
        for existing_id in episode_ids:
            if inquiry_history(view, existing_id)[-1].receipt.state.status == "active":
                raise GhostConflict("The private inquiry slot already holds an active episode.")
        selected = next(item for item in nominations if item.record.id == decision.nomination_id)
        state = InquiryState(episode_id=episode_id, origin_record_id=selected.record.id, grant=grant,
            node_ids=(), frontier_ids=(), pending_generation_ids=(), root_source_ids=selected.receipt.root_source_ids,
            inspected_root_ids=selected.receipt.root_source_ids, sequence=0, status="active", stop_reason="",
            started_at_utc=now_utc, active_elapsed_seconds=0, stalled_steps=0,
            next_operation=selected.receipt.nomination.suggested_operation, resume_after_utc=None, resume_source_ids=())
        inquiry = _stage_receipt(batch, InquiryReceipt(parent_receipt_id=None, state=state, attention=decision,
            step=None, operation_record_id=None, alternative_record_ids=()))
        record = inquiry.record
    else:
        record = batch.create(Feature(content=decision.reason, feature_type=FeatureType.StateUpdate,
            entity_id=grant.observer_entity_id, scope=grant.scope, provenance=Provenance(source_kind=SourceKind.Self),
            interlocus=InterlocusType.PrivateInternal,
            lineage=RecordLineage(episode_id=batch_id, operation="private_attention",
                                 parent_record_ids=list(nomination_ids))))
    attention = ThoughtAttentionReceipt(nomination_ids=tuple(nomination_ids), decision=decision, grant=grant)
    record.metadata.update({ATTENTION_BATCH_KEY: batch_id, "thought_attention": attention.model_dump(mode="json")})
    batch.revise(record)
    batch.depend_on(nomination_ids)
    return AttentionRecord(record, attention, inquiry)


def stage_inquiry_step(batch: WriteBatch, *, parent_receipt_id: int, step: InquiryStep,
                       grant: InquiryGrant, active_seconds: float, now_utc: datetime,
                       limits: ControlConfig) -> InquiryRecord:
    """Stage one connected operation, branches, and its next state atomically.

    ``active_seconds`` comes from monotonic computation timing in the caller.
    The caller must validate Critical and wait authority through control at
    commit. This boundary checks source meaning, graph parents, and finite
    limits. It performs no inference and sends no finding to main cognition.
    """
    if not math.isfinite(active_seconds) or active_seconds < 0 or now_utc.tzinfo is not None:
        raise ValueError("Inquiry computation needs finite active seconds and naive UTC chronology.")
    view = batch.view()
    parent_record = view.get(parent_receipt_id)
    parent = InquiryMetadata.model_validate(parent_record.metadata).inquiry_receipt
    prior = parent.state
    history = inquiry_history(view, prior.episode_id)
    if history[-1].record.id != parent_receipt_id:
        raise GhostConflict("An inquiry continuation requires its exact current receipt parent.")
    if prior.status != "active":
        raise ValueError("Only an admitted active inquiry can execute another operation.")
    if prior.sequence >= limits.episode_steps or prior.active_elapsed_seconds >= limits.episode_seconds:
        raise ValueError("An inquiry cannot execute beyond its accepted step or time bound.")
    if prior.stalled_steps >= limits.stalled_steps:
        raise ValueError("An inquiry cannot execute beyond its accepted stall bound.")
    if (grant.observer_entity_id != prior.grant.observer_entity_id or grant.scope != prior.grant.scope
            or grant.critical_epoch != prior.grant.critical_epoch):
        raise ValueError("A continuation cannot replace its observer, reality, or Critical epoch.")
    dependencies = {source.record_id: source for source in grant.sources}
    for source in prior.grant.sources:
        if source.record_id not in dependencies or dependencies[source.record_id] != source:
            raise ValueError("An inquiry continuation cannot refresh its original source fingerprints.")
    expect_inquiry_grant(batch, grant)
    _guard_family(batch, INQUIRY_EPISODE_KEY, prior.episode_id)
    _guard_family(batch, PRIVATE_SLOT_KEY, _slot(grant))
    batch.expect_revision(parent_receipt_id, view.revisions[parent_receipt_id])
    batch.depend_on([parent_receipt_id])
    operation = validate_operation(step.operation)
    if step.operation != prior.next_operation:
        raise ValueError("An inquiry must execute its admitted next operation.")
    if step.next_operation:
        validate_operation(step.next_operation)
    targets = {relation.target_record_id for relation in step.relations}
    if not targets.issubset(prior.node_ids):
        raise ValueError("Inquiry relations must name already accepted nodes in this episode.")
    if prior.sequence and not targets:
        raise ValueError("An inquiry continuation requires an explicit relation to its earlier graph.")
    if operation.mode != "generative" and step.alternatives:
        raise ValueError("Only a generative operation can create alternative branches.")
    permitted = frozenset(dependencies)
    if not targets.issubset(permitted):
        raise ValueError("An inquiry relation must use the captured grant of inspected graph nodes.")
    roots = set(source_roots(view, step.evidence_ids, scope=grant.scope, permitted_ids=permitted))
    for alternative in step.alternatives:
        roots.update(source_roots(view, alternative.evidence_ids, scope=grant.scope, permitted_ids=permitted))
    compared = set(step.progress.compared_node_ids)
    if not compared.issubset(targets):
        raise ValueError("Discrimination must retain explicit relations to every compared node.")
    meaningful = False
    if step.progress.kind == "new_evidence":
        inspected = set(step.progress.inspected_source_ids)
        if not inspected or not inspected.issubset(step.evidence_ids):
            raise ValueError("Evidence progress requires newly inspected original sources cited by this operation.")
        for source_id in inspected:
            if source_roots(view, [source_id], scope=grant.scope, permitted_ids=permitted) != (source_id,):
                raise ValueError("Internal derivations cannot count as acquired original evidence.")
        already_supplied = {source.record_id for source in prior.grant.sources}
        meaningful = bool(inspected - set(prior.inspected_root_ids) - already_supplied)
    elif step.progress.kind == "discrimination":
        if operation.mode != "discriminatory" or len(compared) < 2:
            raise ValueError("Discrimination progress requires an actual comparison of accepted graph nodes.")
        meaningful = step.progress.meaningful_difference
    pending = set(prior.pending_generation_ids)
    if operation.mode == "discriminatory":
        pending.difference_update(compared)
    if step.outcome in {"finding", "completed"} and (operation.mode == "generative" or pending):
        raise ValueError("A generated branch requires discrimination before a proposed conclusion.")
    parents = sorted(targets) if targets else [prior.origin_record_id]
    thought = batch.create(Feature(content=step.thought, feature_type=FeatureType.InnerMonologue,
        entity_id=grant.observer_entity_id, scope=grant.scope, provenance=Provenance(source_kind=SourceKind.Self),
        interlocus=InterlocusType.PrivateInternal,
        lineage=RecordLineage(episode_id=prior.episode_id, operation=operation.name, sequence=prior.sequence + 1,
            parent_record_ids=parents, source_event_ids=list(step.evidence_ids), reality=step.reality),
        metadata={"inquiry_step": step.model_dump(mode="json"), "memory_kind": "interpretation"}))
    alternatives: list[int] = []
    for alternative in step.alternatives:
        branch = batch.create(Feature(content=alternative.text, feature_type=FeatureType.Thought,
            entity_id=grant.observer_entity_id, scope=grant.scope, provenance=Provenance(source_kind=SourceKind.Self),
            interlocus=InterlocusType.PrivateInternal,
            lineage=RecordLineage(episode_id=prior.episode_id, operation="inquiry_alternative", sequence=prior.sequence + 1,
                parent_record_ids=[thought.id], source_event_ids=list(alternative.evidence_ids), reality=alternative.reality),
            metadata={"inquiry_alternative": alternative.model_dump(mode="json"), "memory_kind": "interpretation"}))
        alternatives.append(branch.id)
    if operation.mode == "generative":
        pending.update([thought.id, *alternatives])
    elapsed = prior.active_elapsed_seconds + active_seconds
    stalled = 0 if meaningful else prior.stalled_steps + 1
    status = "active"
    stop_reason = ""
    if step.outcome == "abandoned":
        status, stop_reason = "abandoned", step.reason
    elif step.outcome in {"finding", "completed", "information_gap"}:
        status, stop_reason = "completed", step.reason
    elif step.outcome in {"wait", "pause"}:
        status, stop_reason = "paused", step.reason
    elif prior.sequence + 1 >= limits.episode_steps:
        status, stop_reason = "paused", "step limit"
    elif elapsed >= limits.episode_seconds:
        status, stop_reason = "paused", "active computation time limit"
    elif stalled >= limits.stalled_steps:
        status, stop_reason = "paused", "stalled progress"
    dismissed = {relation.target_record_id for relation in step.relations if relation.kind == "dismisses"}
    frontier = [node_id for node_id in prior.frontier_ids if node_id not in dismissed]
    frontier.extend([thought.id, *alternatives])
    next_state = InquiryState(episode_id=prior.episode_id, origin_record_id=prior.origin_record_id,
        grant=grant, node_ids=(*prior.node_ids, thought.id, *alternatives), frontier_ids=tuple(frontier),
        pending_generation_ids=tuple(sorted(pending)), root_source_ids=tuple(sorted(set(prior.root_source_ids) | roots)),
        inspected_root_ids=tuple(sorted(set(prior.inspected_root_ids) | roots)), sequence=prior.sequence + 1,
        status=status, stop_reason=stop_reason, started_at_utc=prior.started_at_utc,
        active_elapsed_seconds=elapsed, stalled_steps=stalled,
        next_operation=step.next_operation if status in {"active", "paused"} else "",
        resume_after_utc=now_utc + timedelta(seconds=limits.parked_seconds) if status == "paused" else None,
        resume_source_ids=tuple(sorted(roots)) if status == "paused" else (), reflection_source_id=prior.reflection_source_id)
    return _stage_receipt(batch, InquiryReceipt(parent_receipt_id=parent_receipt_id, state=next_state, attention=None,
        step=step, operation_record_id=thought.id, alternative_record_ids=tuple(alternatives)))
