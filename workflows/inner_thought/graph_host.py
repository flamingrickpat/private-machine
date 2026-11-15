"""Bind inquiry staging to the private host's durable acceptance boundary.

The canonical overview is :mod:`workflows.inner_thought.host`. This adapter
owns no timer, model, or second slot. The existing host supplies its control
commit callback and its accepted mental state. Nomination and attention commits
create no GWT turn. An executed step creates an inner turn; only an explicit
outcome boundary becomes a finding for main appraisal. One exception is
intentional: a metacognition step (``rumination_notice`` or
``attention_drift_notice``) is delivered as an intermediate result even when
the episode continues, so main cognition hears what the companion noticed
about its own thinking; the step's recorded activity travels with it.

Every graph append is a recovery root; the derived GWT is a separate commit.
Accepted graph nodes survive interrupted
delivery, and replay uses actual main claimed IDs rather than rerunning a
completed model call. Scheduling and Critical/wait validation remain with the
injected host commit callback.

The slot readers separate three kinds of retained work. Active work holds the
slot. Parked work can resume its own graph. Displaced work is closed by
accepted Critical reconsideration; only its question returns, as a new episode.

Accepted deliberate mental work enters the same slot through
:meth:`PrivateInquiryOwner.admit_deliberate_action`. It parks spontaneous work
that holds the slot and leaves a receipt naming the action. A finding of a
deliberate episode carries its action identity, and the deliberate owner
(:mod:`workflows.focus_management.host`) delivers it instead of this loop.

Each executed operation also retains one felt-state sample. The sample is
computed once, when the operation is accepted, from the frozen mental state and
the previous accepted sample. The derived inner turn stores it and emits a
``QualeRecord`` for each notably moved channel. Repair reuses the retained
sample, so a later accepted state never rewrites an earlier felt state.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from threading import Event

from pydantic import BaseModel, ConfigDict, Field

from pm import vclock
from pm.memory.dependencies import changed_dependencies
from pm.memory.snapshot import Snapshot
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.base import Provenance
from pm.records.enums import EntityClass, FeatureType, InterlocusType, SourceKind
from pm.records.lineage import RecordLineage
from pm.records.memory_records import Entity, Feature
from pm.records.mental_state import FullMentalState
from pm.records.qualia_records import QualeChannelState, QualeRecord
from workflows.cognitive_control.schemas import ControlConfig
from workflows.cognitive_control.attention import CapturedPrivateFocus, validate_private_focus
from workflows.inner_thought.inquiry import (
    AttentionRecord, DELIBERATE_ACTION_KEY, InquiryMetadata, InquiryRecord, PRIVATE_SLOT_KEY,
    deliberate_admission, displaced_successor, inquiry_history, stage_attention, stage_attention_outcome,
    stage_deliberate_inquiry, stage_displaced_inquiry, stage_inquiry_displacement, stage_inquiry_preemption,
    stage_inquiry_outdated, stage_inquiry_resume, stage_inquiry_step,
)
from workflows.inner_thought.metacognition import METACOGNITION_OPERATIONS
from workflows.inner_thought.nominations import extend_inquiry_grant
from workflows.inner_thought.nominations import NominationBatchRecord, stage_nomination_batch
from workflows.inner_thought.request_views import ThoughtActivityView
from workflows.inner_thought.schemas import (
    DeliberateAdmission, InquiryGrant, InquiryResumeDecision, InquiryStep, NominationBatch, RetainedOfferSelection,
    ThoughtAttentionDecision,
)
from workflows.inner_thought.graph_outcomes import (
    GraphOutcomePayload, attach_graph_outcome, read_graph_outcome, stage_graph_outcome,
)
from workflows.inner_thought.felt_state import (
    is_reportable, notable_channels, previous_cycle_sample, project_channels,
)
from workflows.main_cognition.cycle_commits import active_intentions
from workflows.main_cognition.schemas import ActionMetadata

#: Metadata key that names the accepted mental action a result belongs to. The
#: deliberate owner delivers every record that carries it.
MENTAL_ACTION_KEY = "mental_action_id"
#: Metadata key of the correlation the deliberate owner delivers that result under.
MENTAL_CORRELATION_KEY = "mental_correlation"


class FeltStateSample(BaseModel):
    """The felt-state channels sampled when one private operation was accepted."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    sampled_at_utc: datetime = Field(description="Naive UTC time of the sample. Channel derivatives use the interval from the previous accepted sample to this time.")
    channels: dict[str, QualeChannelState] = Field(description="Position, velocity, and acceleration of every delivered channel, keyed by channel id.")


class FrozenProjectionPayload(BaseModel):
    """Retain the actual accepted state supplied during this private operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    mental_state: FullMentalState = Field(description="Original frozen mental state for this operation's inner GWT projection; repair never substitutes later state.")
    felt_state: FeltStateSample | None = Field(default=None, description="Felt-state sample taken at acceptance. None only for an operation accepted before the graph sampled felt state; no sample is invented for it.")


class InnerProjectionMetadata(BaseModel):
    """Read the required original projection from heterogeneous operation metadata."""

    model_config = ConfigDict(extra="ignore")
    inquiry_projection: FrozenProjectionPayload


def attach_inner_projection(operation: Feature, payload: FrozenProjectionPayload) -> None:
    """Attach the exact projection before the operation's graph-root commit."""
    operation.metadata["inquiry_projection"] = payload.model_dump(mode="json")


def capture_felt_state(store: MemoryStore, mental_state: FullMentalState, *,
                       driver_ids: Sequence[int], now_utc: datetime) -> FeltStateSample:
    """Sample every felt-state channel for one operation that is being accepted.

    Call while holding the store lock. ``mental_state`` is the frozen accepted
    state the operation ran under, and ``driver_ids`` are the sources the
    operation cited. The previous sample is the newest accepted cycle that
    carries one, from either loop. The sample is pure math over accepted state;
    it changes no mental state.
    """
    # ``store.states`` is read directly. ``read_cycles`` would record every
    # cycle as a causal read of the surrounding invocation.
    previous_channels, previous_time = previous_cycle_sample(list(store.states))
    active_goals = active_intentions(store.snapshot(capture=False))
    channels = project_channels(mental_state, active_goals=active_goals, previous=previous_channels,
        previous_time=previous_time, now=now_utc, driver_ids=list(driver_ids))
    return FeltStateSample(sampled_at_utc=now_utc, channels=channels)


def stage_felt_state(batch: WriteBatch, operation: Feature, evidence_ids: Sequence[int],
                     sample: FeltStateSample) -> list[int]:
    """Stage one ``QualeRecord`` for each notably moved channel of a retained sample.

    Returns the new record IDs in channel order. A channel below the movement
    threshold stays in the cycle's channel map and gets no record. A record is
    reportable only when a driver was cited by the operation, which is the same
    causal gate the linear path applies.
    """
    gate_ids = {*evidence_ids, operation.id}
    quale_ids: list[int] = []
    for channel_id in notable_channels(sample.channels):
        channel = sample.channels[channel_id]
        reportable = is_reportable(channel.last_driver_ids, attention_ids=gate_ids, causal_ids=gate_ids)
        quale = batch.create(QualeRecord(
            content=f"{channel_id} moved to {channel.position:.3f} "
                    f"(velocity {channel.velocity:.4f} per second).",
            channel_id=channel_id, position=channel.position, velocity=channel.velocity,
            acceleration=channel.acceleration, driver_ids=list(channel.last_driver_ids),
            reportable=reportable, scope=operation.scope))
        quale_ids.append(quale.id)
    return quale_ids


#: Metadata key of the receipt that records a graph outcome effect left unapplied.
STALE_OUTCOME_KEY = "stale_graph_outcome"


class StaleOutcomeNote(BaseModel):
    """Which retained outcome effect a late inner-turn projection did not apply, and why."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    operation_id: int = Field(description="The accepted graph operation whose outcome effect was left out.")
    expectation_action: str = Field(description="The effect that was not applied: the outcome's expectation action.")
    changed_source_ids: tuple[int, ...] = Field(description="Outcome sources that changed after the graph root was accepted.")


def stage_stale_outcome_note(batch: WriteBatch, operation: Feature, outcome: GraphOutcomePayload,
                             changed_source_ids: tuple[int, ...]) -> Feature:
    """Stage the receipt for an outcome whose expectation effect can no longer be applied.

    The finding itself stays valid and reaches main cognition; only the
    expectation effect, which depended on the changed sources, is dropped.
    """
    if operation.lineage is None:
        raise ValueError("An accepted graph operation requires its episode lineage.")
    note = StaleOutcomeNote(operation_id=operation.id, expectation_action=outcome.integration.expectation.action,
                            changed_source_ids=changed_source_ids)
    batch.depend_on([operation.id])
    return batch.create(Feature(
        content=f"The expectation effect of private step {operation.id} was not applied: "
                f"records {list(changed_source_ids)} changed after the step was accepted.",
        feature_type=FeatureType.StateUpdate, interlocus=InterlocusType.PrivateInternal,
        entity_id=operation.entity_id, scope=operation.scope, provenance=Provenance(source_kind=SourceKind.Self),
        metadata={STALE_OUTCOME_KEY: note.model_dump(mode="json"), "memory_kind": "runtime_receipt"},
        lineage=RecordLineage(episode_id=operation.lineage.episode_id, operation="stale_graph_outcome",
            parent_record_ids=[operation.id])))


def private_observer(view: Snapshot, companion_name: str) -> Entity:
    """Read the one companion identity anchor that holds the private slot.

    Zero or several matching anchors raise: a private inquiry cannot guess
    whose slot it uses.
    """
    identities = [record for record in view.all_knoxels.values()
                  if isinstance(record, Entity) and record.entity_class == EntityClass.Self
                  and record.content == companion_name]
    if len(identities) != 1:
        raise ValueError("A private inquiry requires one exact companion identity anchor.")
    return identities[0]


def accepted_source_ids(store: MemoryStore) -> frozenset[int]:
    """Return every record whose producing commit is accepted.

    A private grant may only name these records. Every graph commit is an
    append-only recovery root, and the store refuses a recovery root that
    depends on an unaccepted commit. The derived inner turn is such a commit
    until a main cycle reads it, so an expectation that a private outcome just
    opened is not citable by the next private opportunity. It becomes citable
    once main cognition has accepted a cycle that read it.
    """
    return frozenset(store.recovery_snapshot(capture=False).revisions)


def _slot_heads(view: Snapshot, *, observer_entity_id: int) -> list[InquiryRecord]:
    """Read the current receipt head of every episode in one observer's private slot."""
    episodes = {InquiryMetadata.model_validate(record.metadata).inquiry_episode
                for record in view.all_knoxels.values()
                if record.metadata.get(PRIVATE_SLOT_KEY) == str(observer_entity_id)}
    return [inquiry_history(view, episode)[-1] for episode in sorted(episodes)]


def active_inquiry(view: Snapshot, *, observer_entity_id: int) -> InquiryRecord | None:
    """Read the sole active slot without treating parked work as active."""
    active = [head for head in _slot_heads(view, observer_entity_id=observer_entity_id)
              if head.receipt.state.status == "active"]
    if len(active) > 1:
        raise ValueError("One private observer cannot hold multiple active inquiries.")
    return active[0] if active else None


def parked_inquiries(view: Snapshot, *, observer_entity_id: int) -> tuple[InquiryRecord, ...]:
    """Read paused heads in descending receipt allocation order.

    Parked work never holds the slot, so several parked episodes can exist.
    The caller offers the newest eligible head to shared attention. Episode
    names can contain hashes and carry no chronology. A newer ineligible head
    must not hide older eligible work.
    """
    parked = [head for head in _slot_heads(view, observer_entity_id=observer_entity_id)
              if head.receipt.state.status == "paused"]
    parked.sort(key=lambda head: head.record.id, reverse=True)
    return tuple(parked)


def displaced_inquiries(view: Snapshot, *, observer_entity_id: int) -> tuple[InquiryRecord, ...]:
    """Read displaced heads whose question no episode took over, newest first.

    A displaced episode never holds the slot and never resumes. Its question
    stays here until shared attention admits it as a new episode.
    """
    waiting: list[InquiryRecord] = []
    for head in _slot_heads(view, observer_entity_id=observer_entity_id):
        if head.receipt.state.status != "displaced":
            continue
        if displaced_successor(view, head.record.id) is None:
            waiting.append(head)
    waiting.sort(key=lambda head: head.record.id, reverse=True)
    return tuple(waiting)


def deliberate_inquiry(view: Snapshot, action_record_id: int) -> InquiryRecord | None:
    """Read the episode one accepted deliberate action already owns, if any.

    A restart or replayed dispatch must reuse an accepted inquiry instead of
    paying for the same deliberate work twice.
    """
    episodes = {InquiryMetadata.model_validate(record.metadata).inquiry_episode
                for record in view.all_knoxels.values()
                if record.metadata.get(DELIBERATE_ACTION_KEY) == str(action_record_id)}
    if not episodes:
        return None
    if len(episodes) != 1:
        raise ValueError("One deliberate action cannot own multiple private inquiries.")
    return inquiry_history(view, next(iter(episodes)))[-1]


def deliberate_action_ids(view: Snapshot) -> tuple[int, ...]:
    """Return every accepted action that owns a private inquiry, in ascending order."""
    action_ids = {int(str(record.metadata[DELIBERATE_ACTION_KEY])) for record in view.all_knoxels.values()
                  if DELIBERATE_ACTION_KEY in record.metadata}
    return tuple(sorted(action_ids))


def pending_findings(store: MemoryStore) -> tuple[Feature, ...]:
    """Return accepted outcome nodes of spontaneous work not yet claimed by actual main appraisal.

    A finding of a deliberate episode carries its action identity. The
    deliberate owner delivers it with the action's correlation, so it is left
    out here and never delivered twice. A noticing is a finding even without
    an outcome, because main cognition must hear it once; it keeps the step's
    recorded activity, and the episode can continue.
    """
    with store.lock:
        appraised = {record_id for cycle in store.read_cycles() for record_id in cycle.claimed_ids()}
        return tuple(record for record in store.all_features
                     if record.metadata.get("private_inquiry_finding") is True
                     and record.metadata.get(MENTAL_ACTION_KEY) is None and record.id not in appraised)


class PrivateInquiryOwner:
    """Use the host's control fence for graph allocation and executed inner turns."""

    def __init__(self, store: MemoryStore, *, commit: Callable[[WriteBatch], None],
                 cancel: Event, limits: ControlConfig) -> None:
        self.store: MemoryStore = store
        self.commit: Callable[[WriteBatch], None] = commit
        self.cancel: Event = cancel
        self.limits: ControlConfig = limits

    def accept_nominations(self, result: NominationBatch, *, grant: InquiryGrant,
                           batch_id: str) -> NominationBatchRecord:
        """Retain a generated batch, including a genuine empty result."""
        with self.store.lock:
            batch = self.store.begin_batch("nomination:" + batch_id, embed=False, cancel=self.cancel)
            accepted = stage_nomination_batch(batch, result, grant=grant, batch_id=batch_id)
            batch.preserve_on_recovery()
            self.commit(batch)
            return accepted

    def accept_attention(self, nomination_ids: Sequence[int], decision: ThoughtAttentionDecision, *,
                         commit_id: str, episode_id: str, now_utc: datetime,
                         validate_context: Callable[[], None] | None = None,
                         displaced_selection: RetainedOfferSelection | None = None) -> AttentionRecord:
        """Commit selection separately; it cannot fabricate an executed thought.

        ``displaced_selection`` records why the displaced question offered
        beside these nominations was the one offered, when one was.
        """
        with self.store.lock:
            if validate_context is not None:
                validate_context()
            batch = self.store.begin_batch(commit_id, embed=False, cancel=self.cancel)
            selected = stage_attention(batch, nomination_ids, decision, episode_id=episode_id, now_utc=now_utc,
                                       displaced_selection=displaced_selection)
            batch.preserve_on_recovery()
            self.commit(batch)
            return selected

    def admit_parked_inquiry(self, parked: InquiryRecord, decision: InquiryResumeDecision,
                            *, attention: ThoughtAttentionDecision, nomination_ids: Sequence[int],
                            offered_grant: InquiryGrant, grant: InquiryGrant, batch_id: str,
                            now_utc: datetime, focus: CapturedPrivateFocus,
                            displaced_selection: RetainedOfferSelection | None = None) -> InquiryRecord:
        """Commit the attention decision that re-admits parked work in one batch.

        The opportunity identity and the slot change land together, so an
        interrupted admission can neither spend the opportunity without resuming
        the question nor resume it twice. ``offered_grant`` is the grant the
        offered nominations were captured under; ``grant`` is the extended grant
        that retains the parked inquiry's original fingerprints plus its fresh
        trigger evidence. The captured focus head and its original expiry are
        checked under the store lock before either record is staged.
        """
        with self.store.lock:
            validate_private_focus(self.store.snapshot(capture=False), focus, now_utc=vclock.utcnow())
            batch = self.store.begin_batch("attention:" + batch_id, embed=False, cancel=self.cancel)
            stage_attention_outcome(batch, batch_id=batch_id, nomination_ids=tuple(nomination_ids),
                decision=attention, grant=offered_grant, displaced_selection=displaced_selection)
            accepted = stage_inquiry_resume(batch, parent_receipt_id=parked.record.id, decision=decision,
                attention=attention, grant=grant, now_utc=now_utc, limits=self.limits)
            batch.preserve_on_recovery()
            self.commit(batch)
            return accepted

    def displace_superseded(self, *, observer_entity_id: int, critical_epoch: int) -> tuple[InquiryRecord, ...]:
        """Close every active or parked inquiry admitted under an older Critical epoch.

        ``critical_epoch`` is the accepted epoch of the current opportunity.
        Each closed episode gets its own commit through the host's control
        fence, so the slot is free before this opportunity nominates or admits
        anything. No model call happens here. The returned receipts are the
        terminal displaced heads; their questions are offered to shared
        attention through :func:`displaced_inquiries`.
        """
        displaced: list[InquiryRecord] = []
        with self.store.lock:
            view = self.store.snapshot(capture=False)
            for head in _slot_heads(view, observer_entity_id=observer_entity_id):
                state = head.receipt.state
                if state.status not in {"active", "paused"}:
                    continue
                if state.grant.critical_epoch >= critical_epoch:
                    continue
                batch = self.store.begin_batch(f"inquiry-displace:{head.record.id}", embed=False, cancel=self.cancel)
                closed = stage_inquiry_displacement(batch, parent_receipt_id=head.record.id,
                    superseding_critical_epoch=critical_epoch)
                batch.preserve_on_recovery()
                self.commit(batch)
                displaced.append(closed)
        return tuple(displaced)

    def close_outdated(self, *, observer_entity_id: int) -> tuple[InquiryRecord, ...]:
        """Close every active or parked inquiry whose retained grant names a changed source.

        A continuation keeps its admission grant and may not refresh it. When
        main cognition replaces a goal or an expectation that the grant holds
        with current-head authority, or a source changes meaning, every later
        step would pay for a model call and then be refused at commit. Each
        such episode gets its own terminal receipt through the host's control
        fence instead (:func:`~workflows.inner_thought.inquiry.stage_inquiry_outdated`),
        so the slot is free before this opportunity nominates or admits
        anything. No model call happens here. The question returns through
        :func:`displaced_inquiries` as a new episode under a fresh grant.
        """
        closed: list[InquiryRecord] = []
        with self.store.lock:
            view = self.store.snapshot(capture=False)
            for head in _slot_heads(view, observer_entity_id=observer_entity_id):
                state = head.receipt.state
                if state.status not in {"active", "paused"}:
                    continue
                if not changed_dependencies(view, state.grant.sources):
                    continue
                batch = self.store.begin_batch(f"inquiry-outdated:{head.record.id}", embed=False, cancel=self.cancel)
                outdated = stage_inquiry_outdated(batch, parent_receipt_id=head.record.id)
                batch.preserve_on_recovery()
                self.commit(batch)
                closed.append(outdated)
        return tuple(closed)

    def displaced_grant(self, displaced: InquiryRecord, *, offered_grant: InquiryGrant) -> InquiryGrant:
        """Capture the grant a displaced question would start its new episode under.

        The current opportunity's grant supplies the observer, the reality, the
        accepted Critical epoch, and the present source meanings. The displaced
        receipt and the earlier nodes are added from the current view, so the
        new episode can cite its earlier thinking as private history. No
        fingerprint of the superseded grant is reused.
        """
        earlier = displaced.receipt.state
        with self.store.lock:
            return extend_inquiry_grant(self.store.snapshot(capture=False), offered_grant,
                                        [displaced.record.id, *earlier.node_ids])

    def admit_displaced_inquiry(self, displaced: InquiryRecord, *, attention: ThoughtAttentionDecision,
                                nomination_ids: Sequence[int], offered_grant: InquiryGrant,
                                grant: InquiryGrant, batch_id: str, episode_id: str, now_utc: datetime,
                                focus: CapturedPrivateFocus,
                                displaced_selection: RetainedOfferSelection) -> InquiryRecord:
        """Commit the attention decision that admits a displaced question in one batch.

        ``displaced_selection`` is the receipt of why this question was the one
        offered; it must name the admitted receipt.

        The opportunity identity and the new episode land together, exactly as
        for a parked re-admission, so an interrupted admission can neither
        spend the opportunity without opening the episode nor open it twice.
        ``offered_grant`` is the grant the offered nominations were captured
        under; ``grant`` comes from :meth:`displaced_grant`. The captured focus
        head and its original expiry are checked under the store lock first.
        """
        with self.store.lock:
            validate_private_focus(self.store.snapshot(capture=False), focus, now_utc=vclock.utcnow())
            batch = self.store.begin_batch("attention:" + batch_id, embed=False, cancel=self.cancel)
            stage_attention_outcome(batch, batch_id=batch_id, nomination_ids=tuple(nomination_ids),
                decision=attention, grant=offered_grant, displaced_selection=displaced_selection)
            accepted = stage_displaced_inquiry(batch, displaced_receipt_id=displaced.record.id,
                attention=attention, grant=grant, episode_id=episode_id, now_utc=now_utc)
            batch.preserve_on_recovery()
            self.commit(batch)
            return accepted

    def record_attention_outcome(self, *, batch_id: str, nomination_ids: Sequence[int],
                                 decision: ThoughtAttentionDecision, grant: InquiryGrant,
                                 focus: CapturedPrivateFocus,
                                 displaced_selection: RetainedOfferSelection | None = None) -> Feature:
        """Commit an attention decision that leaves the private slot untouched.

        This retains the opportunity identity for an opportunity that offered
        only parked work and produced a deferral or dismissal. Replaced or
        expired focus cannot consume that opportunity.
        """
        with self.store.lock:
            validate_private_focus(self.store.snapshot(capture=False), focus, now_utc=vclock.utcnow())
            batch = self.store.begin_batch("attention:" + batch_id, embed=False, cancel=self.cancel)
            record = stage_attention_outcome(batch, batch_id=batch_id, nomination_ids=tuple(nomination_ids),
                decision=decision, grant=grant, displaced_selection=displaced_selection)
            batch.preserve_on_recovery()
            self.commit(batch)
            return record

    def execution_grant(self, parent: InquiryRecord, *, source_ids: Sequence[int] = ()) -> InquiryGrant:
        """Capture inspected graph IDs without refreshing earlier source meaning."""
        state = parent.receipt.state
        with self.store.lock:
            return extend_inquiry_grant(self.store.snapshot(capture=False), state.grant,
                                        [*state.node_ids, *source_ids])

    def accept_resume(self, parent: InquiryRecord, decision: InquiryResumeDecision,
                      *, attention: ThoughtAttentionDecision, grant: InquiryGrant, commit_id: str,
                      now_utc: datetime) -> InquiryRecord:
        """Retain explicit parked admission through the same control-owned commit.

        This boundary creates no operation or GWT turn. Shared attention must
        have made the supplied decision for this exact parked receipt; the timer
        never auto-resumes. Original source, epoch, private-slot, and
        finite-budget checks remain mandatory through staging and the host's
        commit callback.
        """
        with self.store.lock:
            batch = self.store.begin_batch(commit_id, embed=False, cancel=self.cancel)
            accepted = stage_inquiry_resume(batch, parent_receipt_id=parent.record.id, decision=decision,
                attention=attention, grant=grant, now_utc=now_utc, limits=self.limits)
            batch.preserve_on_recovery()
            self.commit(batch)
            return accepted

    def accept_deliberate_entry(self, admission: DeliberateAdmission, *, grant: InquiryGrant,
                                commit_id: str, now_utc: datetime) -> InquiryRecord:
        """Commit one accepted deliberate action into the same private slot.

        Allocation creates no operation and no GWT turn, exactly like a
        spontaneous selection. The deliberate action's own identity guards keep
        a replayed dispatch from allocating the work twice.
        """
        with self.store.lock:
            batch = self.store.begin_batch(commit_id, embed=False, cancel=self.cancel)
            accepted = stage_deliberate_inquiry(batch, admission=admission, grant=grant, now_utc=now_utc)
            batch.preserve_on_recovery()
            self.commit(batch)
            return accepted

    def admit_deliberate_action(self, admission: DeliberateAdmission, *, grant: InquiryGrant,
                                now_utc: datetime) -> InquiryRecord:
        """Give one accepted deliberate action the private slot, parking spontaneous work.

        ``grant`` is captured by the deliberate owner at the current accepted
        Critical epoch and must contain the action. Three things happen under
        the store lock, each as its own commit through the control fence:

        1. Work from an older Critical epoch, and work whose retained grant
           names a changed source, is closed as displaced, exactly as on a
           private timer opportunity, so stale work cannot hold the slot.
        2. An active spontaneous inquiry is parked with a receipt that names
           this action. It keeps its graph and returns through shared attention.
        3. The deliberate episode opens. It runs no operation here; the private
           loop executes it on its timer.

        Deliberate work never parks other deliberate work: that case raises
        ``GhostConflict`` and the caller retries when the slot is free. An
        interruption after step 2 leaves the spontaneous question parked and
        the action unentered; a replayed request then enters the free slot. An
        action that already owns an episode returns that episode's current
        receipt and commits nothing.
        """
        with self.store.lock:
            existing = deliberate_inquiry(self.store.snapshot(capture=False), admission.action_record_id)
            if existing is not None:
                return existing
            self.displace_superseded(observer_entity_id=grant.observer_entity_id,
                                     critical_epoch=grant.critical_epoch)
            self.close_outdated(observer_entity_id=grant.observer_entity_id)
            holder = active_inquiry(self.store.snapshot(capture=False), observer_entity_id=grant.observer_entity_id)
            if holder is not None:
                batch = self.store.begin_batch(f"inquiry-preempt:{holder.record.id}", embed=False, cancel=self.cancel)
                stage_inquiry_preemption(batch, parent_receipt_id=holder.record.id,
                    action_record_id=admission.action_record_id, now_utc=now_utc)
                batch.preserve_on_recovery()
                self.commit(batch)
            return self.accept_deliberate_entry(admission, grant=grant,
                commit_id=f"deliberate-entry:{admission.action_record_id}", now_utc=now_utc)

    def accept_step(self, parent: InquiryRecord, step: InquiryStep, *, grant: InquiryGrant,
                    commit_id: str, active_seconds: float, now_utc: datetime,
                    mental_state: FullMentalState, control_generation: int,
                    outcome: GraphOutcomePayload | None = None,
                    activity: ThoughtActivityView | None = None) -> InquiryRecord:
        """Commit the graph atomically, then derive its inner GWT projection.

        Generation and comparison nodes remain internal graph events. A finding
        marker is attached only at an explicit outcome boundary, or when this
        step is a metacognition operation: a noticing is delivered to main
        cognition as an intermediate result while the episode continues, so
        main can appraise what the companion noticed about its own thinking.
        ``activity`` is the recorded activity facts of a noticing step; they
        travel with its delivery for review. The host sends the accepted
        identity to main after this method returns. A finding of a deliberate
        episode also carries its action identity and correlation, and the
        deliberate owner sends it instead.
        """
        with self.store.lock:
            deliberate = deliberate_admission(self.store.snapshot(capture=False), parent.receipt.state)
            batch = self.store.begin_batch(commit_id, new_tick=True, embed=False, cancel=self.cancel)
            accepted = stage_inquiry_step(batch, parent_receipt_id=parent.record.id, step=step, grant=grant,
                active_seconds=active_seconds, now_utc=now_utc, limits=self.limits)
            operation_id = accepted.receipt.operation_record_id
            assert operation_id is not None
            operation = batch.view().get(operation_id)
            if not isinstance(operation, Feature):
                raise TypeError("An executed inquiry operation must be a Feature.")
            operation.metadata.update({"loop": "inquiry", "control_generation": control_generation,
                "private_inquiry_finding": step.outcome != "continue"
                or step.operation in METACOGNITION_OPERATIONS,
                "inquiry_receipt_id": accepted.record.id})
            if activity is not None:
                operation.metadata["thought_activity"] = activity.model_dump(mode="json")
            felt_state = capture_felt_state(self.store, mental_state, driver_ids=step.evidence_ids, now_utc=now_utc)
            attach_inner_projection(operation, FrozenProjectionPayload(mental_state=mental_state, felt_state=felt_state))
            if outcome is not None:
                attach_graph_outcome(operation, step, outcome,
                    perspective_subject_id=deliberate.subject_entity_id if deliberate is not None else None)
            if deliberate is not None and step.outcome != "continue":
                action = batch.view().get(deliberate.action_record_id)
                correlation = ActionMetadata.model_validate(action.metadata).dispatch_correlation
                operation.metadata.update({MENTAL_ACTION_KEY: action.id, MENTAL_CORRELATION_KEY: correlation})
            batch.revise(operation)
            batch.preserve_on_recovery()
            self.commit(batch)
            self.ensure_inner_turn(accepted)
            return accepted

    def ensure_inner_turn(self, accepted: InquiryRecord) -> None:
        """Restore a missing derived inner turn from accepted graph IDs, without inference.

        Recovery roots deliberately exclude workspace and cycle mutation. The
        graph commit is authoritative; this idempotent second commit projects
        its actual operation and original mental state into the inner workspace.
        A missing original payload fails rather than inventing historical state.
        Failure leaves the
        accepted graph intact for repair and never repeats its model call.

        The retained felt-state sample becomes the cycle's channel map, and each
        notably moved channel gets a ``QualeRecord`` in the same turn. The cycle
        then carries the sample's own time, so a late repair does not shorten
        the interval that the next sample measures. An operation accepted
        before the graph sampled felt state gets an empty channel map.
        """
        with self.store.lock:
            operation_id = accepted.receipt.operation_record_id
            step = accepted.receipt.step
            if operation_id is None or step is None:
                raise ValueError("Selection alone cannot create an executed inner GWT turn.")
            if any(turn.slot_id == "inner" and operation_id in turn.knoxel_ids
                   for turn in self.store.gwt_turns.values()):
                return
            batch = self.store.begin_batch("inquiry-gwt:" + str(accepted.record.id),
                                            new_tick=True, embed=False, cancel=self.cancel)
            batch.depend_on([accepted.record.id, operation_id])
            operation = batch.view().get(operation_id)
            if not isinstance(operation, Feature):
                raise TypeError("An executed inquiry operation must be a Feature.")
            projection = InnerProjectionMetadata.model_validate(operation.metadata).inquiry_projection
            outcome = read_graph_outcome(operation)
            stale_note_ids: list[int] = []
            if outcome is not None:
                stale_ids = changed_dependencies(batch.view(), outcome.sources)
                if stale_ids:
                    # The graph root was accepted, then a source of its outcome
                    # changed before this projection (a restart between the two
                    # commits). The projection would be refused forever, so the
                    # turn and the finding go ahead without the expectation
                    # effect, and a receipt says which effect was not applied.
                    stale_note_ids.append(stage_stale_outcome_note(batch, operation, outcome, stale_ids).id)
                else:
                    stage_graph_outcome(batch, batch.view(), operation, step, outcome)
            sample = projection.felt_state
            if sample is None:
                quale_ids: list[int] = []
                cycle = CycleState(tick_id=batch.tick_id, mental_state=projection.mental_state)
            else:
                quale_ids = stage_felt_state(batch, operation, step.evidence_ids, sample)
                cycle = CycleState(tick_id=batch.tick_id, timestamp=sample.sampled_at_utc,
                    mental_state=projection.mental_state, qualia_channels=dict(sample.channels))
            batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, slot_id="inner",
                knoxel_ids=[operation_id, *quale_ids, *stale_note_ids], summary=step.thought), cycle)
            self.commit(batch)
