"""Bind inquiry staging to the private host's durable acceptance boundary.

The canonical overview is :mod:`workflows.inner_thought.host`. This adapter
owns no timer, model, or second slot. The existing host supplies its control
commit callback and its accepted mental state. Nomination and attention commits
create no GWT turn. An executed step creates an inner turn; only an explicit
outcome boundary becomes a finding for main appraisal.

Every graph append is a recovery root; the derived GWT is a separate commit.
Accepted graph nodes survive interrupted
delivery, and replay uses actual main claimed IDs rather than rerunning a
completed model call. Scheduling and Critical/wait validation remain with the
injected host commit callback.

The slot readers separate three kinds of retained work. Active work holds the
slot. Parked work can resume its own graph. Displaced work is closed by
accepted Critical reconsideration; only its question returns, as a new episode.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from threading import Event

from pydantic import BaseModel, ConfigDict, Field

from pm import vclock
from pm.memory.snapshot import Snapshot
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Feature
from pm.records.mental_state import FullMentalState
from workflows.cognitive_control.schemas import ControlConfig
from workflows.cognitive_control.attention import CapturedPrivateFocus, validate_private_focus
from workflows.inner_thought.inquiry import (
    AttentionRecord, DELIBERATE_ACTION_KEY, InquiryMetadata, InquiryRecord, PRIVATE_SLOT_KEY,
    displaced_successor, inquiry_history, stage_attention, stage_attention_outcome, stage_deliberate_inquiry,
    stage_displaced_inquiry, stage_inquiry_displacement, stage_inquiry_resume, stage_inquiry_step,
)
from workflows.inner_thought.nominations import extend_inquiry_grant
from workflows.inner_thought.nominations import NominationBatchRecord, stage_nomination_batch
from workflows.inner_thought.schemas import (
    DeliberateAdmission, InquiryGrant, InquiryResumeDecision, InquiryStep, NominationBatch, RetainedOfferSelection,
    ThoughtAttentionDecision,
)
from workflows.inner_thought.graph_outcomes import (
    GraphOutcomePayload, attach_graph_outcome, read_graph_outcome, stage_graph_outcome,
)


class FrozenProjectionPayload(BaseModel):
    """Retain the actual accepted state supplied during this private operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    mental_state: FullMentalState = Field(description="Original frozen mental state for this operation's inner GWT projection; repair never substitutes later state.")


class InnerProjectionMetadata(BaseModel):
    """Read the required original projection from heterogeneous operation metadata."""

    model_config = ConfigDict(extra="ignore")
    inquiry_projection: FrozenProjectionPayload


def attach_inner_projection(operation: Feature, payload: FrozenProjectionPayload) -> None:
    """Attach the exact projection before the operation's graph-root commit."""
    operation.metadata["inquiry_projection"] = payload.model_dump(mode="json")


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


def pending_findings(store: MemoryStore) -> tuple[Feature, ...]:
    """Return accepted outcome nodes not yet claimed by actual main appraisal."""
    with store.lock:
        appraised = {record_id for cycle in store.read_cycles() for record_id in cycle.claimed_ids()}
        return tuple(record for record in store.all_features
                     if record.metadata.get("private_inquiry_finding") is True and record.id not in appraised)


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

    def accept_step(self, parent: InquiryRecord, step: InquiryStep, *, grant: InquiryGrant,
                    commit_id: str, active_seconds: float, now_utc: datetime,
                    mental_state: FullMentalState, control_generation: int,
                    outcome: GraphOutcomePayload | None = None) -> InquiryRecord:
        """Commit the graph atomically, then derive its inner GWT projection.

        Generation and comparison nodes remain internal graph events. A finding
        marker is attached only at an explicit outcome boundary. The host sends
        that accepted identity to main after this method returns.
        """
        with self.store.lock:
            batch = self.store.begin_batch(commit_id, new_tick=True, embed=False, cancel=self.cancel)
            accepted = stage_inquiry_step(batch, parent_receipt_id=parent.record.id, step=step, grant=grant,
                active_seconds=active_seconds, now_utc=now_utc, limits=self.limits)
            operation_id = accepted.receipt.operation_record_id
            assert operation_id is not None
            operation = batch.view().get(operation_id)
            if not isinstance(operation, Feature):
                raise TypeError("An executed inquiry operation must be a Feature.")
            operation.metadata.update({"loop": "inquiry", "control_generation": control_generation,
                "private_inquiry_finding": step.outcome != "continue",
                "inquiry_receipt_id": accepted.record.id})
            attach_inner_projection(operation, FrozenProjectionPayload(mental_state=mental_state))
            if outcome is not None:
                attach_graph_outcome(operation, step, outcome)
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
            if outcome is not None:
                stage_graph_outcome(batch, batch.view(), operation, step, outcome)
            batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, slot_id="inner",
                knoxel_ids=[operation_id], summary=step.thought),
                CycleState(tick_id=batch.tick_id, mental_state=projection.mental_state))
            self.commit(batch)
