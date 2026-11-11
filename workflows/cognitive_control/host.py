"""Durable inference pauses and replayable timeout stimuli.

The memory store owns every transition. No transition calls a model. A timeout
commits a pending wake record before bus publication. Main cognition consumes
that record in its own commit. Restart replays unconsumed records, not elapsed
private work. Bus publication alone is not an acknowledgment.
"""
from __future__ import annotations

from pm import vclock

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from threading import Event
from typing import TYPE_CHECKING, Any

from pm.bus.envelopes import Envelope, new_envelope, new_id
from pm.host.application import BusModule
from pm.memory.snapshot import GhostCancelled, GhostConflict
from pm.llm.concurrency import InferenceAdmission, InferenceRole
from pm.memory.dependencies import source_fingerprint
from pm.memory.write_batch import WriteBatch
from pm.memory.read_scope import ReadScope
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.base import Provenance
from pm.records.memory_records import Feature, new_feature
from pm.records.lineage import RecordLineage
from workflows.cognitive_control.schemas import ControlConfig, WaitRequest
from workflows.public_speech.publication_control import PublicationControl, ReplySuperseded, validate_unmanaged_selection
from workflows.input_triage.schemas import EarlyTriageAssessment
from workflows.main_cognition.schemas import TriageResult

if TYPE_CHECKING:
    from pm.memory.store import MemoryStore


class CognitiveControl(BusModule):
    """Own waits, external-input fences, and the wake outbox.

    ``now`` supplies UTC wall time. Persisted deadlines survive process stops.
    A backward clock change delays a deadline. A forward change can expire it.
    Elapsed-time receipts never claim cognition during the gap.
    """

    def __init__(self, *args: Any, ghost: MemoryStore, config: ControlConfig | None = None,
                 now: Callable[[], datetime] = vclock.utcnow, **kwargs: Any) -> None:
        """Forward the module configuration and retain the store and clock."""
        super().__init__(*args, **kwargs)
        self.ghost = ghost
        self.config = config or ControlConfig()
        self.now = now
        # Accepted cycles retain the generation across process stops. Without
        # this watermark, restart can discard a valid continuing episode.
        self.generation = max([0,
            *(int(item.metadata.get("control_generation", 0)) for item in ghost.all_features),
            *(item.control_generation for item in ghost.states),
            *(int(item.metadata.get("generation", 0)) + 1 for item in ghost.all_features
              if item.metadata.get("control_kind") == "inference_wait")])
        self.inference_admission: InferenceAdmission | None = None
        self.cancel_private: Callable[[], None] = lambda: None
        self._pending_inputs: set[str] = set()
        self._seen_inputs: set[str] = set()
        self._enqueued_wakes: set[int] = set()
        # Evidence that arrived during a wait and was held for the next cycle.
        # This list is in memory only, like the poller that produced it.
        self._deferred: list[Envelope] = []
        self.input_triage_enabled: bool = False
        self._assessments: dict[int, EarlyTriageAssessment] = {}
        self._pending_triage: set[int] = set()
        self._critical_sources: set[int] = set()
        self._reconsideration_epoch: int = 0
        self._critical_accepted_tick: int = 0
        self.cancel_critical_private: Callable[[], None] = lambda: None
        self.prioritize_critical: Callable[[set[int]], None] = lambda source_ids: None
        self.cancel_public_drafts: Callable[[int], None] = lambda epoch: None
        self.validate_selection: Callable[[str], None] = validate_unmanaged_selection

    def enable_input_triage(self) -> None:
        """Reconstruct unresolved classification and reconsideration from memory.

        Historical input already claimed by accepted cognition needs no new
        classification. Durable Critical assessments remain required until an
        accepted cycle actually claims their sources.
        """
        with self.ghost.lock:
            self.input_triage_enabled = True
            claimed = {source_id: state.tick_id for state in self.ghost.states
                       for source_id in state.claimed_ids()}
            for record in self.ghost.all_features:
                if record.metadata.get("control_kind") != "input_triage":
                    continue
                assessment = EarlyTriageAssessment.model_validate(record.metadata["assessment"])
                self._reconsideration_epoch = max(self._reconsideration_epoch, assessment.reconsideration_epoch)
                for source_id in assessment.source_ids:
                    self._assessments[source_id] = assessment
                for item in assessment.result.items:
                    if item.category == "Critical":
                        if item.knoxel_id in claimed:
                            self._critical_accepted_tick = max(self._critical_accepted_tick, claimed[item.knoxel_id])
                        else:
                            self._critical_sources.add(item.knoxel_id)
            for record in self.ghost.all_features:
                if (record.metadata.get("exchange_correlation") and record.metadata.get("exchange_status") == "pending"
                        and record.id not in claimed and record.id not in self._assessments):
                    self._pending_triage.add(record.id)
            if self.inference_admission is not None:
                for source_id in self._critical_sources:
                    self.inference_admission.reserve(f"critical:{source_id}", role=InferenceRole.Critical)

    def register_input(self, source_id: int) -> None:
        """Fence a committed input under the same store lock as publication."""
        if not self.input_triage_enabled:
            return
        with self.ghost.lock:
            self.ghost.snapshot({source_id}).get(source_id)
            if source_id not in self._assessments:
                self._pending_triage.add(source_id)

    def has_assessment(self, source_id: int) -> bool:
        """Return whether this source already has an accepted early assessment."""
        with self.ghost.lock:
            return source_id in self._assessments

    def accept_triage(self, assessment: EarlyTriageAssessment) -> EarlyTriageAssessment:
        """Commit classification before releasing its unknown-input barrier."""
        critical = {item.knoxel_id for item in assessment.result.items if item.category == "Critical"}
        with self.ghost.lock, ReadScope() as acceptance_reads:
            # A control receipt preserves its exact supplied evidence, not
            # speculative branches captured by the surrounding workflow view.
            acceptance_reads.collecting = False
            source_ids = set(assessment.source_ids)
            if (len(assessment.source_ids) != len(source_ids)
                    or not source_ids.issubset(assessment.evidence.record_fingerprints)
                    or set(assessment.source_fingerprints) != source_ids
                    or {item.knoxel_id for item in assessment.result.items} != source_ids
                    or len(assessment.result.items) != len(source_ids)):
                raise ValueError("Early triage must classify exactly its supplied source IDs.")
            for source_id in source_ids:
                if source_fingerprint(self.ghost.snapshot({source_id}).get(source_id)) != assessment.source_fingerprints[source_id]:
                    raise GhostConflict(f"Triage source {source_id} changed during classification.")
                if source_id in self._assessments:
                    raise ValueError(f"Source {source_id} already has an accepted assessment.")
            evidence_view = self.ghost.recovery_snapshot(capture=False)
            required_heads = set(assessment.evidence.current_head_ids) | source_ids
            for source_id, fingerprint in assessment.evidence.record_fingerprints.items():
                if source_id in required_heads and source_id not in evidence_view.all_knoxels:
                    raise GhostConflict(f"Triage evidence {source_id} acquired an accepted successor.")
                if source_fingerprint(evidence_view.get(source_id)) != fingerprint:
                    raise GhostConflict(f"Triage evidence {source_id} changed during classification.")
            epoch = self._reconsideration_epoch + int(bool(critical))
            accepted = assessment.model_copy(update={"reconsideration_epoch": epoch})
            batch = self.ghost.begin_batch("input-triage-" + new_id(), embed=False)
            batch.depend_on(assessment.evidence.record_fingerprints,
                            manifest={"kind": "input_triage", "evidence": assessment.evidence.model_dump(mode="json")})
            batch.depend_on_commits(assessment.evidence.commit_ids)
            batch.preserve_on_recovery()
            batch.create(Feature(
                content="; ".join(f"{item.category}: {item.reasoning}" for item in assessment.result.items),
                feature_type=FeatureType.StateUpdate, interlocus=InterlocusType.PrivateInternal,
                provenance=Provenance(source_kind=SourceKind.System, source_ref="input_triage"),
                metadata={"control_kind": "input_triage", "assessment": accepted.model_dump(mode="json")},
                lineage=RecordLineage(episode_id=batch.commit_id, operation="input_triage",
                                     source_event_ids=assessment.source_ids)))
            self.ghost.commit(batch)
            self._reconsideration_epoch = epoch
            self._critical_sources.update(critical)
            if self.inference_admission is not None:
                for source_id in critical:
                    self.inference_admission.reserve(f"critical:{source_id}", role=InferenceRole.Critical)
            self._pending_triage.difference_update(source_ids)
            for source_id in source_ids:
                self._assessments[source_id] = accepted
            if critical:
                # Mark the old main invocation interrupted before another
                # writer can observe the required sources at its commit fence.
                self.prioritize_critical(critical)
                self.cancel_public_drafts(epoch)
        if critical:
            self.cancel_critical_private()
        return accepted

    def publication_control(self) -> PublicationControl:
        """Expose typed scheduling callbacks without a speech-to-behavior import."""
        return PublicationControl(
            wait_for_reconsideration=self.wait_for_reconsideration,
            wait_for_classification=self.wait_for_classification,
            publication_epoch=self.publication_epoch,
            validate_publication=self.validate_publication,
            validate_selection=lambda selection_id: self.validate_selection(selection_id),
        )

    def validate_cognition(self, source_ids: Sequence[int]) -> None:
        """Reject an older calculation before any state or action commits."""
        with self.ghost.lock:
            if self._critical_sources.difference(source_ids):
                raise GhostCancelled("New Critical input requires a fresh cycle including its evidence.")

    def wait_for_triage(self, source_ids: Sequence[int], cancel: Event) -> bool:
        """Wait for early classification without holding a store or model slot."""
        while not cancel.is_set():
            with self.ghost.lock:
                if not self.input_triage_enabled or all(key in self._assessments for key in source_ids):
                    return True
            cancel.wait(0.05)
        return False

    def triage_result(self, source_ids: Sequence[int]) -> TriageResult | None:
        """Return accepted classifications for the given user sources."""
        with self.ghost.lock:
            if not self.input_triage_enabled or not source_ids:
                return None
            assessments = [self._assessments[key] for key in source_ids]
            items = [next(item for item in assessment.result.items if item.knoxel_id == key)
                     for key, assessment in zip(source_ids, assessments, strict=True)]
            return TriageResult(items=items, codelet_budget=max(item.result.codelet_budget for item in assessments))

    def reconsideration_required(self) -> bool:
        """Return whether unknown or Critical input currently holds ordinary work."""
        with self.ghost.lock:
            return bool(self._pending_triage or self._critical_sources)

    def wait_for_reconsideration(self, cancel: Event) -> bool:
        """Defer ordinary work until required cognition is accepted or stopped."""
        while not cancel.is_set():
            if not self.reconsideration_required():
                return True
            cancel.wait(0.05)
        return False

    def wait_for_classification(self, cancel: Event) -> bool:
        """Resolve unknown input before deciding whether to supersede a draft."""
        while not cancel.is_set():
            with self.ghost.lock:
                if not self._pending_triage:
                    return True
            cancel.wait(0.05)
        return False

    def publication_epoch(self) -> int:
        """Capture the Critical assessment epoch while freezing public context."""
        with self.ghost.lock:
            if self.reconsideration_required():
                raise ReplySuperseded("Input requires classification or accepted reconsideration.")
            return self._reconsideration_epoch

    def validate_publication(self, epoch: int | None, guidance_tick: int | None) -> None:
        """Atomically reject drafts composed before required reconsideration.

        Call while holding ``ghost.lock`` through the durable utterance commit.
        Delivered utterances are already events and never pass this guard again.
        """
        if not self.input_triage_enabled:
            return
        if epoch is None:
            raise ReplySuperseded("Public draft has no early-triage publication stamp.")
        if self.reconsideration_required():
            raise ReplySuperseded("New input requires accepted reconsideration before speech.")
        if epoch != self._reconsideration_epoch:
            raise ReplySuperseded("A Critical input superseded this draft's composition context.")
        if guidance_tick is not None and guidance_tick < self._critical_accepted_tick:
            raise ReplySuperseded("Selected action predates the latest accepted Critical reconsideration.")

    #: Events that end a deliberate wait at once. Outside input and work that
    #: needs a decision cannot wait for a quiet window to expire.
    WAKING_EVENTS = frozenset({"failed", "cancelled", "interrupted", "waiting_for_input"})
    #: Events that a wait holds for the next cycle instead of answering now.
    DEFERRED_EVENTS = frozenset({"completed", "gameplay_progress"})

    @staticmethod
    def is_external(envelope: Envelope) -> bool:
        """Identify evidence that ends a deliberate wait at once."""
        if envelope.topic in {"user.input", "workflow.question"}:
            return True
        return envelope.topic == "workflow.event" and envelope.payload.get("event") in CognitiveControl.WAKING_EVENTS

    @staticmethod
    def is_deferred(envelope: Envelope) -> bool:
        """Identify evidence that a wait holds instead of answering inside it.

        A Minecraft poller observation and a finished managed run are new
        facts, but none of them needs an answer during the quiet window after
        a reply. Left in the waking tier, the 120 second world poll cut a
        600 second window down to two minutes. :meth:`admit` parks these
        envelopes and the wake path replays them, so the record still reaches
        a cycle and the body poller can clear its pending observation.
        """
        if envelope.topic == "minecraft.world":
            return True
        return envelope.topic == "workflow.event" and envelope.payload.get("event") in CognitiveControl.DEFERRED_EVENTS

    def _record(self) -> Feature | None:
        return next((record for record in reversed(list(self.ghost.all_knoxels.values()))
                     if isinstance(record, Feature)
                     and record.metadata.get("control_kind") == "inference_wait"), None)

    def waiting(self) -> bool:
        """Return whether a committed wait blocks routine inference."""
        with self.ghost.lock:
            record = self._record()
            return record is not None and record.metadata["status"] == "waiting"

    def paused_source_ids(self) -> set[int]:
        """Return sources handled by accepted waits, without completing exchanges.

        Restart must not recast a deliberate pause as fresh outside input.
        Original user records keep their pending public-exchange status.
        """
        with self.ghost.lock:
            return {key for record in self.ghost.all_features
                    if record.metadata.get("control_kind") == "inference_wait"
                    for key in record.metadata.get("source_event_ids", [])}

    def observe(self, envelope: Envelope) -> None:
        """Wake before queueing outside input, and fence older private work.

        Pending input also prevents a wait between mailbox drain and run start.
        Main cognition clears it only after its result commits successfully.
        """
        if not self.is_external(envelope):
            return
        woke = False
        with self.ghost.lock:
            if envelope.id in self._seen_inputs:
                return
            self._seen_inputs.add(envelope.id)
            self._pending_inputs.add(envelope.id)
            self.generation += 1
            record = self._record()
            if record is not None and record.metadata["status"] == "waiting":
                self._wake(record, envelope.topic, timeout=False, now=self.now())
                woke = True
        self.cancel_private()
        if woke:
            self._flush_deferred()

    def admit(self, envelopes: list[Envelope]) -> tuple[list[Envelope], int]:
        """Discard consumed wakes and private output from an older generation.

        Waiting never consumes fresh outside input: ``observe`` ends the wait
        before delivery. Self-generated frames and old goals cannot end it.
        """
        with self.ghost.lock:
            if self.waiting():
                # Hold routine evidence for the next cycle. The parked record
                # stays unclaimed, so the body poller keeps its pending
                # observation and clears it when a later cycle reads it.
                for item in envelopes:
                    if self.is_deferred(item) and item.id not in {held.id for held in self._deferred}:
                        self._deferred.append(item)
                return [], self.generation
            accepted = []
            for envelope in envelopes:
                record = self.ghost.get_knoxel_by_id(envelope.payload.get("knoxel_id", -1))
                if envelope.topic == "cognition.wake":
                    if record is None or record.metadata.get("wake_status") != "pending":
                        continue
                if envelope.topic == "thought.created" and record is not None:
                    generation = record.metadata.get("control_generation")
                    if generation is not None and generation != self.generation:
                        continue
                accepted.append(envelope)
            return accepted, self.generation

    def commit(self, batch: WriteBatch, wait: WaitRequest | None, generation: int,
               *, envelope_ids: set[str] | None = None, wake_ids: Sequence[int] = (),
               private_generation: int | None = None, source_ids: Sequence[int] = ()) -> Any:
        """Commit cognition, wake acknowledgment, and a possible wait together.

        Newer input prevents a late Sleep from hiding that input. A private-only
        result from an older generation raises GhostCancelled without a commit.
        Failed commits retain pending input and the wake outbox for recovery.
        """
        armed = False
        handled = envelope_ids or set()
        with self.ghost.lock:
            if batch.commit_id in self.ghost.applied_commits:
                return self.ghost.commit(batch)
            self.validate_cognition(source_ids)
            if private_generation is not None and (self.waiting() or private_generation != self.generation):
                raise GhostCancelled("New outside input superseded private cognition.")
            for key in wake_ids:
                record = batch.view().get(key).copy_for_edit()
                if record.metadata.get("wake_status") == "pending":
                    batch.set_lifecycle(record.id, wake_status="consumed", consumed_by=batch.commit_id)
            if wait is not None:
                # Revalidate objects supplied by trusted host callers too.
                wait = WaitRequest.model_validate(wait.model_dump())
                pending_wakes = {record.id for record in self.ghost.all_features
                                 if record.metadata.get("control_kind") == "inference_wake"
                                 and record.metadata.get("wake_status") == "pending"}
                armed = (generation == self.generation and not (self._pending_inputs - handled)
                         and not (pending_wakes - set(wake_ids)))
                paused_sources = set(source_ids)
                if armed:
                    # A wait chosen after private thought also parks earlier
                    # appraised exchanges. Unprocessed input cannot enter this
                    # set and still blocks acceptance through pending input.
                    view = batch.view()
                    appraised = {key for cycle in self.ghost.states for key in cycle.claimed_ids()}
                    for key in appraised & set(view.revisions):
                        source = view.get(key)
                        if source.metadata.get("exchange_status") == "pending":
                            paused_sources.add(key)
                now = self.now()
                seconds = min(wait.seconds, self.config.maximum_wait_seconds)
                new_feature(self.ghost, wait.reason, FeatureType.StateUpdate,
                    source=self.ghost.get_companion_name(), source_kind=SourceKind.Self,
                    interlocus=InterlocusType.PrivateInternal, changes=batch,
                    metadata={"control_kind": "inference_wait" if armed else "inference_wait_rejected",
                              "status": "waiting" if armed else "superseded",
                              "wake_at": (now + timedelta(seconds=seconds)).isoformat(),
                              "started_at": now.isoformat(), "seconds": seconds,
                              "reason": wait.reason, "generation": self.generation,
                              "source_event_ids": sorted(paused_sources)},
                    lineage=RecordLineage(episode_id=batch.commit_id, operation="inference_wait",
                                          parent_record_ids=list(source_ids), source_event_ids=sorted(paused_sources)))
            receipt = self.ghost.commit(batch)
            accepted_critical = self._critical_sources.intersection(source_ids)
            if accepted_critical:
                self._critical_sources.difference_update(accepted_critical)
                if self.inference_admission is not None:
                    for source_id in accepted_critical:
                        self.inference_admission.release(f"critical:{source_id}")
                self._critical_accepted_tick = max(self._critical_accepted_tick, batch.tick_id)
            self._pending_inputs.difference_update(handled)
            if armed:
                # A wait also invalidates output from a private run that has
                # not yet reached its commit boundary.
                self.generation += 1
        if armed:
            self.cancel_private()
        return receipt

    def commit_private(self, batch: WriteBatch, generation: int) -> Any:
        """Reject private work after outside input or an accepted wait.

        The store lock covers the generation check and durable commit. Native
        cancellation is cooperative, so this fence also rejects a late result.
        """
        with self.ghost.lock:
            if self.waiting() or self.reconsideration_required() or generation != self.generation:
                raise GhostCancelled("Private work was superseded by a wait or new stimulus.")
            return self.ghost.commit(batch)

    def _wake(self, record: Feature, reason: str, *, timeout: bool, now: datetime) -> None:
        batch = self.ghost.begin_batch("wake-" + new_id(), cancel=self.stop_event)
        accepted = batch.view().get(record.id).copy_for_edit()
        batch.set_lifecycle(accepted.id, status="awake", wake_reason=reason, woke_at=now.isoformat())
        if timeout:
            seconds = max(0, (now - datetime.fromisoformat(record.metadata["started_at"])).total_seconds())
            new_feature(self.ghost,
                f"{seconds:.0f} seconds passed since I chose to wait. I can reconsider what to do.",
                FeatureType.WorldEvent, source="cognitive_control", source_kind=SourceKind.System,
                interlocus=InterlocusType.PrivateInternal, changes=batch,
                metadata={"origin": "cognition.wake", "control_kind": "inference_wake",
                          "wake_status": "pending", "source_event_ids": [record.id],
                          "elapsed_seconds": seconds, "memory_kind": "executed_event"},
                lineage=RecordLineage(episode_id=record.lineage.episode_id if record.lineage else batch.commit_id,
                    operation="cognition_wake", sequence=record.lineage.sequence + 1 if record.lineage else 0,
                    parent_record_ids=[record.id], source_event_ids=[record.id]))
        self.ghost.commit(batch)

    def _flush_deferred(self) -> None:
        """Re-publish evidence that a wait held, so the next cycle reads it.

        Each parked envelope goes back to the bus in arrival order with a
        fresh envelope id and its original causation, so a reader can still
        trace it to the module that produced it. The store already holds its
        record, so a stop before republish loses only the delivery: the record
        still reaches a later cycle through the shard tail of its group.
        """
        held, self._deferred = self._deferred, []
        for envelope in held:
            self.bus.put(new_envelope(envelope.source, envelope.kind, envelope.topic,
                                      envelope.correlation_id, dict(envelope.payload),
                                      target=envelope.target, causation_id=envelope.causation_id or envelope.id))

    def release_wakes(self, envelopes: list[Envelope]) -> None:
        """Permit replay after a cancelled or rejected main invocation."""
        with self.ghost.lock:
            self._enqueued_wakes.difference_update(
                int(item.payload["knoxel_id"]) for item in envelopes if item.topic == "cognition.wake")

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Expire a wait, then publish pending wakes once per active invocation.

        A crash before publication leaves a pending record. A crash after main
        commit leaves a consumed record. Restart only replays the former.
        """
        try:
            self._expire_and_publish()
        except GhostCancelled:
            # Shutdown can arrive after timer admission but before checkpoint
            # replacement. It must not turn normal cancellation into a crash.
            if not self.stop_event.is_set():
                raise

    def _expire_and_publish(self) -> None:
        """Commit an expired deadline before publishing its durable outbox."""
        with self.ghost.lock:
            record = self._record()
            now = self.now()
            if (record is not None and record.metadata["status"] == "waiting"
                    and now >= datetime.fromisoformat(record.metadata["wake_at"])):
                self._wake(record, "timeout", timeout=True, now=now)
                self.generation += 1
            pending = [record.id for record in self.ghost.all_features
                       if record.metadata.get("control_kind") == "inference_wake"
                       and record.metadata.get("wake_status") == "pending"
                       and record.id not in self._enqueued_wakes]
            self._enqueued_wakes.update(pending)
        for key in pending:
            try:
                self.bus.put(new_envelope(self.module_id, "data", "cognition.wake", "",
                                          {"knoxel_id": key}, target="main_cog"))
            except BaseException:
                with self.ghost.lock:
                    self._enqueued_wakes.discard(key)
                raise
        if not self.waiting():
            self._flush_deferred()
