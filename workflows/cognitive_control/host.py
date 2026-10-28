"""Durable inference pauses and replayable timeout stimuli.

The memory store owns every transition. No transition calls a model. A timeout
commits a pending wake record before bus publication. Main cognition consumes
that record in its own commit. Restart replays unconsumed records, not elapsed
private work. Bus publication alone is not an acknowledgment.
"""
from __future__ import annotations

from pm import vclock

from datetime import datetime, timedelta
from typing import Callable, Sequence

from pm.bus.envelopes import Envelope, new_envelope, new_id
from pm.host.application import BusModule
from pm.memory.snapshot import GhostCancelled
from pm.memory.write_batch import WriteBatch
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import Feature, new_feature
from pm.records.lineage import RecordLineage
from workflows.cognitive_control.schemas import ControlConfig, WaitRequest


class CognitiveControl(BusModule):
    """Own waits, external-input fences, and the wake outbox.

    ``now`` supplies UTC wall time. Persisted deadlines survive process stops.
    A backward clock change delays a deadline. A forward change can expire it.
    Elapsed-time receipts never claim cognition during the gap.
    """

    def __init__(self, *args, ghost, config: ControlConfig | None = None,
                 now: Callable[[], datetime] = vclock.utcnow, **kwargs):
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
        self.cancel_private: Callable[[], None] = lambda: None
        self._pending_inputs: set[str] = set()
        self._seen_inputs: set[str] = set()
        self._enqueued_wakes: set[int] = set()
        # Evidence that arrived during a wait and was held for the next cycle.
        # This list is in memory only, like the poller that produced it.
        self._deferred: list[Envelope] = []

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
               private_generation: int | None = None, source_ids: Sequence[int] = ()):
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
            self._pending_inputs.difference_update(handled)
            if armed:
                # A wait also invalidates output from a private run that has
                # not yet reached its commit boundary.
                self.generation += 1
        if armed:
            self.cancel_private()
        return receipt

    def commit_private(self, batch: WriteBatch, generation: int):
        """Reject private work after outside input or an accepted wait.

        The store lock covers the generation check and durable commit. Native
        cancellation is cooperative, so this fence also rejects a late result.
        """
        with self.ghost.lock:
            if self.waiting() or generation != self.generation:
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
