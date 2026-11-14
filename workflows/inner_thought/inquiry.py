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

Accepted Critical reconsideration ends an inquiry's authority, not its
question. :func:`stage_inquiry_displacement` closes the episode with a terminal
``displaced`` receipt that frees the private slot. :func:`displaced_offer`
names the retained question for shared attention, and
:func:`stage_displaced_inquiry` opens a new episode for it under the current
grant. The earlier grant, nodes, and budgets never carry over.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from pydantic import BaseModel, ConfigDict, JsonValue

from pm.memory.snapshot import GhostConflict, Snapshot
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance
from pm.records.enums import FeatureType, FrameEnum, InterlocusType, SourceKind
from pm.records.lineage import RecordLineage
from pm.records.memory_records import ActionSchemaRecord, Feature
from workflows.cognitive_control.schemas import ControlConfig
from workflows.inner_thought.nominations import expect_inquiry_grant, read_nomination, source_roots
from workflows.inner_thought.operations import validate_operation
from workflows.inner_thought.schemas import (
    DeliberateAdmission, DisplacedInquiryOffer, InquiryAlternative, InquiryDisplacement, InquiryGrant, InquiryOrigin,
    InquiryReceipt, InquiryResumeDecision, InquiryState, InquiryStep, LegacyEpisodeState, ParkedInquiryOffer,
    RetainedOfferSelection, ThoughtAttentionDecision, ThoughtAttentionReceipt,
)
from workflows.main_cognition.schemas import ActionMetadata


INQUIRY_EPISODE_KEY = "inquiry_episode"
INQUIRY_RECEIPT_KEY = "inquiry_receipt"
PRIVATE_SLOT_KEY = "private_inquiry_slot"
ATTENTION_BATCH_KEY = "thought_attention_batch"
DELIBERATE_ACTION_KEY = "deliberate_inquiry_action"
#: On the entry receipt of a new episode: the displaced receipt its question came from.
DISPLACED_ORIGIN_KEY = "displaced_inquiry_origin"
DISPLACED_REASON = "Accepted Critical reconsideration superseded the authority this inquiry was admitted under."


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
    for transition_index, item in enumerate(chain):
        state = item.receipt.state
        if len(set(state.node_ids)) != len(state.node_ids):
            raise ValueError("Inquiry sequence and node identities must be unique and ordered.")
        if not set(state.frontier_ids).issubset(state.node_ids):
            raise ValueError("An inquiry frontier must reference accepted nodes.")
        if not set(state.pending_generation_ids).issubset(state.node_ids):
            raise ValueError("Pending discrimination must reference accepted nodes.")
        if not set(state.inspected_root_ids).issubset(state.root_source_ids):
            raise ValueError("Inspected evidence must retain its original roots.")
        receipt = item.receipt
        if transition_index == 0:
            _validate_entry_transition(state, receipt)
        else:
            prior = chain[transition_index - 1].receipt.state
            if (state.origin_record_id != prior.origin_record_id or state.grant.scope != prior.grant.scope
                    or state.grant.observer_entity_id != prior.grant.observer_entity_id
                    or state.grant.critical_epoch != prior.grant.critical_epoch):
                raise ValueError("An inquiry transition contradicts its retained operation or origin.")
            if prior.status == "displaced":
                raise ValueError("A displaced inquiry accepts no further transition.")
            if receipt.displacement is not None:
                _validate_displacement_transition(prior, receipt)
                continue
            if receipt.resume is not None:
                _validate_resume_transition(prior, receipt)
                continue
            if (receipt.step is None or receipt.operation_record_id is None or receipt.attention is not None
                    or state.sequence != prior.sequence + 1
                    or state.node_ids != (*prior.node_ids, receipt.operation_record_id, *receipt.alternative_record_ids)
                    or state.active_elapsed_seconds < prior.active_elapsed_seconds):
                raise ValueError("An executed inquiry transition contradicts its retained operation.")
            operation = view.get(receipt.operation_record_id)
            if (not isinstance(operation, Feature) or operation.feature_type != FeatureType.InnerMonologue
                    or InquiryNodeMetadata.model_validate(operation.metadata).inquiry_step != receipt.step
                    or operation.content != receipt.step.thought or operation.lineage is None
                    or operation.lineage.sequence != state.sequence or operation.lineage.reality != receipt.step.reality):
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


def outside_source(view: Snapshot, record_id: int, *, grant: InquiryGrant) -> bool:
    """Return whether a captured record is original evidence supplied from outside.

    An early parked reconsideration and its attention offer must name the same
    kind of independently supplied evidence: an outside report addressed to the
    observer, or a system-origin observed event. Self-origin content, internal
    derivations, and anything resolving to another root are never outside
    evidence. This one definition backs both the offer and the durable guard.
    """
    permitted = frozenset(source.record_id for source in grant.sources)
    if record_id not in permitted:
        return False
    record = view.get(record_id)
    if (not isinstance(record, Feature) or not record.causal or record.scope.frame != FrameEnum.InWorld
            or record.provenance is None or record.provenance.source_kind == SourceKind.Self):
        return False
    external_report = (record.feature_type == FeatureType.Dialogue
        and record.interlocus == InterlocusType.Public
        and record.provenance.source_kind in {SourceKind.User, SourceKind.Agent}
        and record.addressee_entity_id in {0, grant.observer_entity_id})
    observed_event = (record.feature_type == FeatureType.WorldEvent
        and record.provenance.source_kind in {SourceKind.System, SourceKind.Workflow})
    if not (external_report or observed_event):
        return False
    return source_roots(view, [record_id], scope=grant.scope, permitted_ids=permitted) == (record_id,)


def inquiry_origin(view: Snapshot, state: InquiryState) -> InquiryOrigin:
    """Read the question an inquiry works on from any of its admission routes.

    A spontaneous episode works on a private proposal with its own judgments. A
    deliberate episode works on main cognition's accepted instruction, which the
    proposal stage never scored, so its judgments stay null rather than invented.
    An episode opened for a displaced question works on the earlier episode's
    question. The earlier judgments were made under the superseded authority,
    so they stay null too.
    """
    entry = inquiry_history(view, state.episode_id)[0].receipt
    if entry.deliberate is not None:
        admission = entry.deliberate
        if admission.action_record_id != state.origin_record_id:
            raise ValueError("A deliberate inquiry contradicts its accepted action identity.")
        return InquiryOrigin(record_id=admission.action_record_id, kind="deliberate_action",
            text=admission.instruction, question=admission.instruction, evidence_ids=admission.evidence_ids,
            judgments=None, suggested_operation="")
    if entry.attention is not None and entry.attention.displaced_receipt_id is not None:
        earlier = read_displaced_inquiry(view, state.origin_record_id)
        earlier_origin = inquiry_origin(view, earlier.receipt.state)
        return InquiryOrigin(record_id=state.origin_record_id, kind="displaced_inquiry",
            text=earlier_origin.text, question=earlier_origin.question, evidence_ids=earlier_origin.evidence_ids,
            judgments=None, suggested_operation="")
    nomination = read_nomination(view, state.origin_record_id).receipt.nomination
    return InquiryOrigin(record_id=state.origin_record_id, kind="nomination", text=nomination.text,
        question=nomination.question, evidence_ids=nomination.evidence_ids, judgments=nomination.judgments,
        suggested_operation=nomination.suggested_operation)


def parked_offer(view: Snapshot, parked: InquiryRecord, *, grant: InquiryGrant, now_utc: datetime,
                 limits: ControlConfig) -> ParkedInquiryOffer | None:
    """Name retained parked work that shared attention can re-admit now, or None.

    Elapsed parking permits reconsideration only; it establishes no new evidence
    and demands no automatic resumption. Fresh original outside evidence permits
    reconsideration before that deadline. Work that already exhausted a finite
    step or active-computation bound stays unofferable, because admission cannot
    reset either bound. A different Critical epoch means required
    reconsideration already supersedes the retained graph, so nothing is offered
    here. :func:`stage_inquiry_displacement` closes such work, and its question
    returns through :func:`displaced_offer` instead.
    """
    if now_utc.tzinfo is not None:
        raise ValueError("Internal inquiry timestamps require naive UTC.")
    state = parked.receipt.state
    if state.status != "paused":
        raise ValueError("Only a paused inquiry can be offered for re-admission.")
    if grant.observer_entity_id != state.grant.observer_entity_id or grant.scope != state.grant.scope:
        raise ValueError("A parked inquiry offer cannot replace its observer or reality.")
    if grant.critical_epoch != state.grant.critical_epoch:
        return None
    if state.sequence >= limits.episode_steps or state.active_elapsed_seconds >= limits.episode_seconds:
        return None
    _retains_original_grant(state.grant, grant)
    supplied = {source.record_id for source in grant.sources}
    already_supplied = {source.record_id for source in state.grant.sources}
    fresh_evidence = tuple(source_id for source_id in sorted(supplied - already_supplied)
                           if outside_source(view, source_id, grant=grant))
    deadline_elapsed = state.resume_after_utc is not None and now_utc >= state.resume_after_utc
    if not deadline_elapsed and not fresh_evidence:
        return None
    return ParkedInquiryOffer(receipt_id=parked.record.id, episode_id=state.episode_id,
        question=inquiry_origin(view, state).question, stop_reason=state.stop_reason,
        resume_after_utc=state.resume_after_utc, deadline_elapsed=deadline_elapsed,
        available_evidence_ids=fresh_evidence)


def read_displaced_inquiry(view: Snapshot, receipt_id: int) -> InquiryRecord:
    """Read the terminal displaced receipt of a closed episode.

    The identity must be the current head of its episode. An earlier receipt of
    that episode, or a head in any other status, is not a displaced question.
    """
    record = view.get(receipt_id)
    episode_id = InquiryMetadata.model_validate(record.metadata).inquiry_episode
    head = inquiry_history(view, episode_id)[-1]
    if head.record.id != receipt_id or head.receipt.state.status != "displaced" or head.receipt.displacement is None:
        raise ValueError("A displaced question requires the terminal displaced receipt of its earlier episode.")
    return head


def displaced_successor(view: Snapshot, displaced_receipt_id: int) -> str | None:
    """Return the episode that already took over a displaced question, or None."""
    episode_ids = {InquiryMetadata.model_validate(record.metadata).inquiry_episode
                   for record in view.all_knoxels.values()
                   if record.metadata.get(DISPLACED_ORIGIN_KEY) == str(displaced_receipt_id)}
    if len(episode_ids) > 1:
        raise ValueError("One displaced question cannot open multiple episodes.")
    return next(iter(episode_ids)) if episode_ids else None


def displaced_offer(view: Snapshot, displaced: InquiryRecord) -> DisplacedInquiryOffer | None:
    """Name a displaced question that shared attention can admit now, or None.

    A displaced question needs no deadline and no new evidence: Critical
    reconsideration interrupted it, the inquiry itself did not stop. It stays
    offered at every qualifying opportunity until an episode takes it over.
    No decision dismisses it. An offer grants nothing; admission opens a new
    episode through :func:`stage_displaced_inquiry`.
    """
    state = displaced.receipt.state
    if state.status != "displaced":
        raise ValueError("Only a displaced inquiry can be offered as a displaced question.")
    if displaced_successor(view, displaced.record.id) is not None:
        return None
    return DisplacedInquiryOffer(receipt_id=displaced.record.id, episode_id=state.episode_id,
        question=inquiry_origin(view, state).question, displaced_reason=state.stop_reason,
        earlier_operation_count=state.sequence)


def _validate_entry_transition(state: InquiryState, receipt: InquiryReceipt) -> None:
    """Accept one entry identity: a nomination, an accepted action, or a displaced question."""
    if (receipt.step is not None or receipt.operation_record_id is not None or receipt.alternative_record_ids
            or receipt.resume is not None or receipt.displacement is not None
            or state.sequence != 0 or state.node_ids):
        raise ValueError("An inquiry entry cannot claim an executed operation.")
    if receipt.deliberate is not None:
        if receipt.attention is not None or receipt.deliberate.action_record_id != state.origin_record_id:
            raise ValueError("A deliberate inquiry entry cannot also claim a selection decision.")
        return
    attention = receipt.attention
    if attention is None or attention.disposition != "select" or attention.parked_receipt_id is not None:
        raise ValueError("Inquiry selection requires its committed nomination identity.")
    if attention.displaced_receipt_id is not None:
        if (attention.displaced_receipt_id != state.origin_record_id
                or attention.admitted_operation != state.next_operation):
            raise ValueError("A displaced question entry requires its exact attention admission.")
        return
    if attention.nomination_id != state.origin_record_id:
        raise ValueError("Inquiry selection requires its committed nomination identity.")


def _validate_displacement_transition(prior: InquiryState, receipt: InquiryReceipt) -> None:
    """Validate a closing displacement: no thought, no progress, no changed graph."""
    displacement = receipt.displacement
    assert displacement is not None
    if (prior.status not in {"active", "paused"} or receipt.step is not None
            or receipt.operation_record_id is not None or receipt.alternative_record_ids
            or receipt.attention is not None or receipt.resume is not None or receipt.deliberate is not None):
        raise ValueError("A displacement closes active or parked work without an executed operation or admission.")
    if displacement.superseding_critical_epoch <= prior.grant.critical_epoch:
        raise ValueError("Only a newer accepted Critical epoch can displace an inquiry.")
    expected = prior.model_copy(update={"status": "displaced", "stop_reason": DISPLACED_REASON,
        "next_operation": "", "resume_after_utc": None, "resume_source_ids": ()})
    if receipt.state != expected:
        raise ValueError("A displaced inquiry must retain its original grant, graph, operation count, and active duration.")


def _retains_original_grant(prior: InquiryGrant, grant: InquiryGrant) -> None:
    """Reject a continuation that refreshes or drops original captured meaning."""
    dependencies = {source.record_id: source for source in grant.sources}
    for source in prior.sources:
        if source.record_id not in dependencies or dependencies[source.record_id] != source:
            raise ValueError("An inquiry continuation cannot refresh its original source fingerprints.")


def _validate_resume_transition(prior: InquiryState, receipt: InquiryReceipt) -> None:
    """Validate explicit admission without counting it as executed progress."""
    decision = receipt.resume
    assert decision is not None
    attention = receipt.attention
    if (prior.status != "paused" or receipt.step is not None or receipt.operation_record_id is not None
            or receipt.alternative_record_ids or receipt.deliberate is not None):
        raise ValueError("A parked inquiry resume cannot claim an executed operation or a deliberate entry.")
    if (attention is None or attention.disposition != "select" or attention.nomination_id is not None
            or attention.parked_receipt_id != receipt.parent_receipt_id
            or attention.admitted_operation != decision.operation or attention.reason != decision.reason):
        raise ValueError("A parked inquiry resume requires the exact attention admission for its parked receipt.")
    _retains_original_grant(prior.grant, receipt.state.grant)
    expected = prior.model_copy(update={"grant": receipt.state.grant, "status": "active", "stop_reason": "",
        "stalled_steps": 0, "next_operation": decision.operation, "resume_after_utc": None, "resume_source_ids": ()})
    if receipt.state != expected:
        raise ValueError("A resumed inquiry must retain its original graph, operation count, and active duration.")
    validate_operation(decision.operation)


def _guard_family(batch: WriteBatch, key: str, value: str) -> None:
    """Guard committed family membership; require one transition per batch."""
    view = batch.view()
    members = {record.id: view.revisions[record.id]
               for record in view.all_knoxels.values()
               if record.id not in batch.created_ids and record.metadata.get(key) == value}
    if any(view.get(record_id).metadata.get(key) == value for record_id in batch.created_ids):
        raise ValueError("An inquiry transaction stages at most one transition per family.")
    batch.expect_metadata_version(key, value, members)


def _stage_receipt(batch: WriteBatch, receipt: InquiryReceipt,
                   *, extra_metadata: Mapping[str, JsonValue] | None = None) -> InquiryRecord:
    """Append one typed transition record and return its actual memory identity."""
    state = receipt.state
    summary = state.stop_reason
    if not summary and receipt.deliberate is not None:
        summary = receipt.deliberate.reason
    if not summary and receipt.resume is not None:
        summary = receipt.resume.reason
    if not summary:
        summary = f"Inquiry {state.episode_id}, step {state.sequence}"
    parent_ids = [receipt.parent_receipt_id] if receipt.parent_receipt_id is not None else [state.origin_record_id]
    metadata: dict[str, JsonValue] = {INQUIRY_EPISODE_KEY: state.episode_id, PRIVATE_SLOT_KEY: _slot(state.grant),
                                      INQUIRY_RECEIPT_KEY: receipt.model_dump(mode="json")}
    if extra_metadata is not None:
        metadata.update(extra_metadata)
    record = batch.create(Feature(content=summary,
        feature_type=FeatureType.StateUpdate, entity_id=state.grant.observer_entity_id, scope=state.grant.scope,
        provenance=Provenance(source_kind=SourceKind.Self), interlocus=InterlocusType.PrivateInternal,
        lineage=RecordLineage(episode_id=state.episode_id, operation="inquiry_transition",
            sequence=state.sequence, parent_record_ids=parent_ids, source_event_ids=list(state.root_source_ids)),
        metadata=metadata))
    return InquiryRecord(record, receipt)


def stage_attention(batch: WriteBatch, nomination_ids: Sequence[int], decision: ThoughtAttentionDecision,
                    *, episode_id: str, now_utc: datetime,
                    displaced_selection: RetainedOfferSelection | None = None) -> AttentionRecord:
    """Persist the separate attention decision and allocate at most one episode.

    The caller supplies a decision from the shared attention owner. This
    function enforces its identities and bounds; it does not choose a thought.
    Selection creates no executed node or GWT turn. ``displaced_selection``
    is retained beside the decision when a displaced question was offered in
    the same opportunity, so a reader can see why that question was the one.
    """
    if not nomination_ids or len(nomination_ids) > 3 or len(set(nomination_ids)) != len(nomination_ids):
        raise ValueError("Private attention requires one to three unique offered nomination IDs.")
    if decision.parked_receipt_id is not None:
        raise ValueError("Parked re-admission uses the explicit inquiry resume boundary, not a nomination batch.")
    if decision.displaced_receipt_id is not None:
        raise ValueError("A displaced question enters through stage_displaced_inquiry, not a nomination batch.")
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
    attention = ThoughtAttentionReceipt(nomination_ids=tuple(nomination_ids), decision=decision, grant=grant,
                                        displaced_selection=displaced_selection)
    _attach_attention(record, attention, batch_id=batch_id)
    batch.revise(record)
    batch.depend_on(nomination_ids)
    return AttentionRecord(record, attention, inquiry)


def _attach_attention(record: Feature, attention: ThoughtAttentionReceipt, *, batch_id: str) -> None:
    """Bind one attention decision to its opportunity identity before revision."""
    record.metadata.update({ATTENTION_BATCH_KEY: batch_id, "thought_attention": attention.model_dump(mode="json")})


def _stage_attention_record(batch: WriteBatch, nomination_ids: Sequence[int], decision: ThoughtAttentionDecision,
                            *, batch_id: str, grant: InquiryGrant,
                            displaced_selection: RetainedOfferSelection | None) -> Feature:
    """Append the durable attention outcome that allocates no inquiry episode."""
    record = batch.create(Feature(content=decision.reason, feature_type=FeatureType.StateUpdate,
        entity_id=grant.observer_entity_id, scope=grant.scope, provenance=Provenance(source_kind=SourceKind.Self),
        interlocus=InterlocusType.PrivateInternal,
        lineage=RecordLineage(episode_id=batch_id, operation="private_attention",
                             parent_record_ids=list(nomination_ids))))
    attention = ThoughtAttentionReceipt(nomination_ids=tuple(nomination_ids), decision=decision, grant=grant,
                                        displaced_selection=displaced_selection)
    _attach_attention(record, attention, batch_id=batch_id)
    batch.revise(record)
    batch.depend_on(nomination_ids)
    return record


def stage_attention_outcome(batch: WriteBatch, *, batch_id: str, nomination_ids: Sequence[int],
                            decision: ThoughtAttentionDecision, grant: InquiryGrant,
                            displaced_selection: RetainedOfferSelection | None = None) -> Feature:
    """Retain one shared attention outcome that allocates no episode by nomination.

    An opportunity may offer only retained parked work. Deferral or dismissal
    then touches no slot, but it still needs the durable opportunity identity, so
    a later tick cannot pay for the same selection call again. A parked
    re-admission and a displaced-question admission record their decision here
    too, because the slot change itself is committed through its own boundary.
    A nomination selection uses
    :func:`stage_attention`, which allocates the episode in the same commit.
    """
    if decision.nomination_id is not None:
        raise ValueError("A nomination selection allocates its episode through stage_attention.")
    if len(nomination_ids) > 3 or len(set(nomination_ids)) != len(nomination_ids):
        raise ValueError("An attention outcome retains at most three unique offered nomination IDs.")
    if decision.parked_receipt_id is not None and decision.disposition != "select":
        raise ValueError("Only a selection can name parked work.")
    if decision.displaced_receipt_id is not None and decision.disposition != "select":
        raise ValueError("Only a selection can name a displaced question.")
    view = batch.view()
    if any(record.metadata.get(ATTENTION_BATCH_KEY) == batch_id for record in view.all_knoxels.values()):
        raise ValueError("One attention opportunity must have exactly one retained outcome.")
    expect_inquiry_grant(batch, grant)
    _guard_family(batch, ATTENTION_BATCH_KEY, batch_id)
    if decision.displaced_receipt_id is not None and (
            displaced_selection is None or displaced_selection.offered_receipt_id != decision.displaced_receipt_id):
        raise ValueError("An admitted displaced question must be the one its offer selection names.")
    return _stage_attention_record(batch, nomination_ids, decision, batch_id=batch_id, grant=grant,
                                   displaced_selection=displaced_selection)


def stage_inquiry_resume(batch: WriteBatch, *, parent_receipt_id: int, decision: InquiryResumeDecision,
                         attention: ThoughtAttentionDecision, grant: InquiryGrant, now_utc: datetime,
                         limits: ControlConfig) -> InquiryRecord:
    """Stage explicit parked admission without executing a thought or resetting bounds.

    Shared attention owns the choice: ``attention`` must be the decision that
    admits this exact parked receipt, and it is retained beside the admission so
    a later reader can see why the question won the slot again. The caller owns
    Critical and wait validation, and commit. An elapsed parked deadline permits
    reconsideration, not automatic execution. Earlier admission requires newly
    supplied original outside evidence. A resume preserves graph identity,
    original source fingerprints, operation count, and active duration. Only its
    stall counter and parking fields reset.
    """
    if now_utc.tzinfo is not None:
        raise ValueError("Internal inquiry resume timestamps require naive UTC.")
    if (attention.disposition != "select" or attention.parked_receipt_id != parent_receipt_id
            or attention.nomination_id is not None):
        raise ValueError("A parked inquiry resume requires the attention decision admitting this exact receipt.")
    if attention.admitted_operation != decision.operation or attention.reason != decision.reason:
        raise ValueError("A parked inquiry resume must carry the admitted operation and reason unchanged.")
    view = batch.view()
    parent = InquiryMetadata.model_validate(view.get(parent_receipt_id).metadata).inquiry_receipt
    prior = parent.state
    history = inquiry_history(view, prior.episode_id)
    if history[-1].record.id != parent_receipt_id:
        raise GhostConflict("A parked inquiry resume requires its exact current receipt parent.")
    if prior.status != "paused":
        raise ValueError("Only a paused inquiry can receive explicit resume admission.")
    if prior.sequence >= limits.episode_steps or prior.active_elapsed_seconds >= limits.episode_seconds:
        raise GhostConflict("A paused inquiry cannot resume beyond its original step or active time bound.")
    if grant.critical_epoch != prior.grant.critical_epoch:
        raise GhostConflict("A parked inquiry cannot resume under a different Critical epoch.")
    if grant.observer_entity_id != prior.grant.observer_entity_id or grant.scope != prior.grant.scope:
        raise ValueError("Resume admission cannot replace an inquiry's observer or reality.")
    _retains_original_grant(prior.grant, grant)
    validate_operation(decision.operation)
    if len(set(decision.trigger_source_ids)) != len(decision.trigger_source_ids):
        raise ValueError("Resume admission requires unique original trigger identities.")
    permitted = frozenset(source.record_id for source in grant.sources)
    already_supplied = {source.record_id for source in prior.grant.sources}
    for source_id in decision.trigger_source_ids:
        if source_id in already_supplied or source_id not in permitted:
            raise ValueError("An early resume trigger must be newly supplied in the captured grant.")
        if not outside_source(view, source_id, grant=grant):
            raise ValueError("A resume trigger must be newly supplied original outside evidence, never generated or derived content.")
    deadline_elapsed = prior.resume_after_utc is not None and now_utc >= prior.resume_after_utc
    if not deadline_elapsed and not decision.trigger_source_ids:
        raise GhostConflict("Parked work needs its original deadline or newly supplied outside evidence before admission.")
    expect_inquiry_grant(batch, grant)
    _guard_family(batch, INQUIRY_EPISODE_KEY, prior.episode_id)
    _guard_family(batch, PRIVATE_SLOT_KEY, _slot(grant))
    episode_ids = {InquiryMetadata.model_validate(record.metadata).inquiry_episode
                   for record in view.all_knoxels.values()
                   if record.metadata.get(PRIVATE_SLOT_KEY) == _slot(grant)}
    for episode_id in episode_ids:
        if episode_id != prior.episode_id and inquiry_history(view, episode_id)[-1].receipt.state.status == "active":
            raise GhostConflict("Another inquiry already holds this observer's private slot.")
    batch.expect_revision(parent_receipt_id, view.revisions[parent_receipt_id])
    batch.depend_on([parent_receipt_id])
    state = prior.model_copy(update={"grant": grant, "status": "active", "stop_reason": "", "stalled_steps": 0,
        "next_operation": decision.operation, "resume_after_utc": None, "resume_source_ids": ()})
    return _stage_receipt(batch, InquiryReceipt(parent_receipt_id=parent_receipt_id, state=state,
        attention=attention, step=None, operation_record_id=None, alternative_record_ids=(), resume=decision))


def stage_inquiry_displacement(batch: WriteBatch, *, parent_receipt_id: int,
                               superseding_critical_epoch: int) -> InquiryRecord:
    """Close active or parked work whose Critical epoch a newer accepted epoch replaced.

    The receipt is terminal for this episode and frees the private slot at
    once. It executes no thought, creates no GWT turn, and keeps the original
    grant, graph, operation count, and active duration as readable history.
    The original source fingerprints are deliberately not staged as
    dependencies: the inquiry ends whether or not its sources changed. The
    caller owns the control commit under the current authority.
    """
    view = batch.view()
    parent = InquiryMetadata.model_validate(view.get(parent_receipt_id).metadata).inquiry_receipt
    prior = parent.state
    history = inquiry_history(view, prior.episode_id)
    if history[-1].record.id != parent_receipt_id:
        raise GhostConflict("An inquiry displacement requires its exact current receipt parent.")
    if prior.status not in {"active", "paused"}:
        raise ValueError("Only active or parked work can be displaced.")
    if superseding_critical_epoch <= prior.grant.critical_epoch:
        raise ValueError("Only a newer accepted Critical epoch can displace an inquiry.")
    _guard_family(batch, INQUIRY_EPISODE_KEY, prior.episode_id)
    _guard_family(batch, PRIVATE_SLOT_KEY, _slot(prior.grant))
    batch.expect_revision(parent_receipt_id, view.revisions[parent_receipt_id])
    batch.depend_on([parent_receipt_id])
    state = prior.model_copy(update={"status": "displaced", "stop_reason": DISPLACED_REASON,
        "next_operation": "", "resume_after_utc": None, "resume_source_ids": ()})
    return _stage_receipt(batch, InquiryReceipt(parent_receipt_id=parent_receipt_id, state=state, attention=None,
        step=None, operation_record_id=None, alternative_record_ids=(),
        displacement=InquiryDisplacement(superseding_critical_epoch=superseding_critical_epoch)))


def stage_displaced_inquiry(batch: WriteBatch, *, displaced_receipt_id: int,
                            attention: ThoughtAttentionDecision, grant: InquiryGrant, episode_id: str,
                            now_utc: datetime) -> InquiryRecord:
    """Open a new episode for a displaced question under the current grant.

    Shared attention owns the choice: ``attention`` must be the decision that
    admits this exact displaced receipt. ``grant`` is captured from the current
    view under the current Critical epoch, and it must contain the displaced
    receipt so the earlier episode stays citable as private history. The new
    episode starts with no nodes, operation count zero, and no active duration.
    The earlier grant and budgets are not inherited. The source roots resolve
    through the earlier receipt chain to the same original records, so the
    earlier thoughts add no corroboration. The caller owns Critical and wait
    validation, and commit.
    """
    if now_utc.tzinfo is not None:
        raise ValueError("Internal inquiry timestamps require naive UTC.")
    if (attention.disposition != "select" or attention.displaced_receipt_id != displaced_receipt_id
            or attention.nomination_id is not None or attention.parked_receipt_id is not None):
        raise ValueError("A displaced question requires the attention decision admitting this exact receipt.")
    validate_operation(attention.admitted_operation)
    view = batch.view()
    displaced = read_displaced_inquiry(view, displaced_receipt_id)
    earlier = displaced.receipt.state
    displacement = displaced.receipt.displacement
    assert displacement is not None  # read_displaced_inquiry requires it.
    if displaced_successor(view, displaced_receipt_id) is not None:
        raise GhostConflict("This displaced question already opened a new episode.")
    if grant.observer_entity_id != earlier.grant.observer_entity_id or grant.scope != earlier.grant.scope:
        raise ValueError("A displaced question cannot change its observer or reality.")
    if grant.critical_epoch < displacement.superseding_critical_epoch:
        raise ValueError("A displaced question requires a grant from the superseding Critical epoch or a later one.")
    permitted = frozenset(source.record_id for source in grant.sources)
    if displaced_receipt_id not in permitted:
        raise ValueError("The new grant must retain the displaced receipt its question came from.")
    if not episode_id.strip():
        raise ValueError("A displaced question requires a new episode identity.")
    if inquiry_history(view, episode_id):
        raise ValueError("A new inquiry cannot reuse an existing episode identity.")
    expect_inquiry_grant(batch, grant)
    roots = source_roots(view, [displaced_receipt_id], scope=grant.scope, permitted_ids=permitted)
    _guard_family(batch, INQUIRY_EPISODE_KEY, episode_id)
    _guard_family(batch, PRIVATE_SLOT_KEY, _slot(grant))
    _guard_family(batch, DISPLACED_ORIGIN_KEY, str(displaced_receipt_id))
    episode_ids = {InquiryMetadata.model_validate(record.metadata).inquiry_episode
                   for record in view.all_knoxels.values()
                   if record.metadata.get(PRIVATE_SLOT_KEY) == _slot(grant)}
    for existing_id in episode_ids:
        if inquiry_history(view, existing_id)[-1].receipt.state.status == "active":
            raise GhostConflict("The private inquiry slot already holds an active episode.")
    batch.expect_revision(displaced_receipt_id, view.revisions[displaced_receipt_id])
    batch.depend_on([displaced_receipt_id])
    state = InquiryState(episode_id=episode_id, origin_record_id=displaced_receipt_id, grant=grant, node_ids=(),
        frontier_ids=(), pending_generation_ids=(), root_source_ids=roots, inspected_root_ids=roots,
        sequence=0, status="active", stop_reason="", started_at_utc=now_utc, active_elapsed_seconds=0,
        stalled_steps=0, next_operation=attention.admitted_operation, resume_after_utc=None, resume_source_ids=())
    return _stage_receipt(batch,
        InquiryReceipt(parent_receipt_id=None, state=state, attention=attention, step=None,
            operation_record_id=None, alternative_record_ids=()),
        extra_metadata={DISPLACED_ORIGIN_KEY: str(displaced_receipt_id)})


def stage_deliberate_inquiry(batch: WriteBatch, *, admission: DeliberateAdmission,
                            grant: InquiryGrant, now_utc: datetime) -> InquiryRecord:
    """Stage a private episode admitted by main cognition's accepted deliberate action.

    The slower deliberate path already chose this work, so no nomination batch
    and no attention model call happen here. The same observer, reality,
    Critical epoch, private-slot, and source-fingerprint guards apply as for a
    spontaneous selection, because one slot cannot hold two episodes. No
    nomination judgments are invented for an instruction that the private
    proposal stage never scored.
    """
    if now_utc.tzinfo is not None:
        raise ValueError("Internal inquiry timestamps require naive UTC.")
    view = batch.view()
    action = view.get(admission.action_record_id)
    if not isinstance(action, ActionSchemaRecord) or action.scope != grant.scope:
        # An action record carries no entity link: main cognition creates the
        # companion's own actions inside one reality. Its scope, plus the fact
        # that only this observer's own owner captured a grant containing it,
        # is what ties the accepted work to this private slot.
        raise ValueError("A deliberate inquiry requires an accepted action record in its inquiry reality.")
    metadata = ActionMetadata.model_validate(action.metadata)
    if metadata.dispatch_status not in {"pending", "admitted"} or action.metadata.get("dispatch_deferred") is True:
        raise ValueError("Only a deliberate action released by main cognition can enter the private slot.")
    if action.metadata.get("mental_result_id") is not None:
        raise ValueError("This deliberate action already owns a mental result.")
    if any(record.metadata.get(DELIBERATE_ACTION_KEY) == str(action.id) for record in view.all_knoxels.values()):
        raise GhostConflict("This accepted deliberate action already owns a private inquiry.")
    validate_operation(admission.operation)
    expect_inquiry_grant(batch, grant)
    permitted = frozenset(source.record_id for source in grant.sources)
    if action.id not in permitted:
        raise ValueError("A deliberate inquiry grant must retain the accepted action it serves.")
    roots = source_roots(view, [action.id, *admission.evidence_ids], scope=grant.scope, permitted_ids=permitted)
    episode_id = f"deliberate:{action.id}"
    _guard_family(batch, INQUIRY_EPISODE_KEY, episode_id)
    _guard_family(batch, PRIVATE_SLOT_KEY, _slot(grant))
    episode_ids = {InquiryMetadata.model_validate(record.metadata).inquiry_episode
                   for record in view.all_knoxels.values()
                   if record.metadata.get(PRIVATE_SLOT_KEY) == _slot(grant)}
    for existing_id in episode_ids:
        if inquiry_history(view, existing_id)[-1].receipt.state.status == "active":
            raise GhostConflict("The private inquiry slot already holds an active episode.")
    batch.expect_revision(action.id, view.revisions[action.id])
    batch.depend_on([action.id])
    state = InquiryState(episode_id=episode_id, origin_record_id=action.id, grant=grant, node_ids=(),
        frontier_ids=(), pending_generation_ids=(), root_source_ids=roots, inspected_root_ids=roots,
        sequence=0, status="active", stop_reason="", started_at_utc=now_utc, active_elapsed_seconds=0,
        stalled_steps=0, next_operation=admission.operation, resume_after_utc=None, resume_source_ids=())
    return _stage_receipt(batch,
        InquiryReceipt(parent_receipt_id=None, state=state, attention=None, step=None,
            operation_record_id=None, alternative_record_ids=(), deliberate=admission),
        extra_metadata={DELIBERATE_ACTION_KEY: str(action.id)})


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
