"""Coordinate a managed memory worker with the Ghost's publication boundaries.

Preparation runs on the supervisor's worker thread. Only the owner of a
completed cognition, wake, or clean shutdown calls publish_ready. The same
store lock protects the ready queue, cancellation, and acceptance. A waking
message can therefore close the queue without waiting for inference to exit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from threading import Event
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from pm.llm.concurrency import InferenceRole, inference_role
from pm.memory.snapshot import GhostCancelled
from pm.vclock import SystemClock

if TYPE_CHECKING:
    from pm.llm.stage_model import LazyStageModel
    from pm.memory.embeddings import EmbeddingService
    from pm.memory.reconsolidation.pipeline import PreparedMemoryUnit, ReconsolidationPass
    from pm.memory.store import MemoryStore


class ReconsolidationMode(StrEnum):
    Background = "background"
    Sleep = "sleep"


class MemoryWorkRequest(BaseModel):
    """The memory worker's projection of the managed request envelope."""

    run_id: str
    reconsolidation_mode: ReconsolidationMode = ReconsolidationMode.Background
    action_record_id: int | None = Field(default=None, alias="_public_action_schema_id", gt=0)


class MemoryWorkProgress(BaseModel):
    """Units prepared by one worker and their publication state."""

    run_id: str
    prepared: int = 0
    published: int = 0
    deferred: int = 0
    awaiting_prior_publication: int = 0

    @property
    def awaiting_publication(self) -> int:
        return self.prepared - self.published - self.deferred


@dataclass(slots=True)
class MemoryWorker:
    request: MemoryWorkRequest
    cancel: Event
    progress: MemoryWorkProgress


@dataclass(slots=True)
class MemorySleep:
    """An approved sleep stays fenced until the awake checkpoint succeeds."""

    action_record_id: int
    ending: bool = False


@dataclass(frozen=True, slots=True)
class ReadyMemory:
    preparation: ReconsolidationPass
    unit: PreparedMemoryUnit
    progress: MemoryWorkProgress


class ReconsolidationCoordinator:
    """One preparation worker and an ordered publication queue for one Ghost."""

    def __init__(self, store: MemoryStore, clock: SystemClock) -> None:
        self.store = store
        self.clock = clock
        self.worker: MemoryWorker | None = None
        self.ready: deque[ReadyMemory] = deque()
        self.sleep: MemorySleep | None = None
        self.closing = Event()

    def arm_sleep(self, action_record_id: int) -> None:
        """Bind a committed inference wait to its approved action record."""
        with self.store.lock:
            if self.sleep is not None:
                raise RuntimeError(f"Memory sleep already belongs to action {self.sleep.action_record_id}.")
            self.sleep = MemorySleep(action_record_id)

    def prepare(self, request: MemoryWorkRequest, model: LazyStageModel, embeddings: EmbeddingService,
                cancel: Event) -> MemoryWorkProgress:
        """Offer only fully prepared units. Publication outlives the worker."""
        from pm.memory.reconsolidation.pipeline import ReconsolidationPass, ReconsolidationProgress

        worker = MemoryWorker(request, cancel, MemoryWorkProgress(run_id=request.run_id))
        with self.store.lock:
            if self.closing.is_set() or cancel.is_set():
                cancel.set()
                raise GhostCancelled(f"Memory preparation {request.run_id} stopped before admission.")
            if request.reconsolidation_mode is ReconsolidationMode.Sleep:
                if request.action_record_id is None:
                    raise ValueError("Sleep reconsolidation needs the main cycle's approved action record.")
                if self.sleep is None or self.sleep.ending or self.sleep.action_record_id != request.action_record_id:
                    cancel.set()
                    raise GhostCancelled(f"Memory sleep action {request.action_record_id} was superseded before start.")
            if self.worker is not None:
                raise RuntimeError(f"Memory worker {self.worker.request.run_id} still owns the Ghost.")
            if self.ready:
                # A direct supervisor start can arrive between two main cycles.
                # The previous worker already prepared this accepted frontier.
                worker.progress.awaiting_prior_publication = len(self.ready)
                return worker.progress
            self.worker = worker
        try:
            with inference_role(InferenceRole.Reconsolidation):
                now = self.clock.utc().replace(tzinfo=None)
                preparation = ReconsolidationPass(self.store, complete=model, embeddings=embeddings,
                    force=True, cancel=cancel, now=now, progress=ReconsolidationProgress(enabled=True))
                for unit in preparation.prepare_units():
                    with self.store.lock:
                        if cancel.is_set():
                            raise GhostCancelled(f"Memory preparation {request.run_id} stopped before handoff.")
                        self.ready.append(ReadyMemory(preparation, unit, worker.progress))
                        worker.progress.prepared += 1
                    if self.closing.is_set():
                        break
            with self.store.lock:
                return worker.progress.model_copy()
        finally:
            with self.store.lock:
                self.worker = None

    def stop_sleep_preparation(self, action_record_id: int) -> None:
        """Cut off further proposals, including a worker still awaiting launch."""
        with self.store.lock:
            if self.sleep is None or self.sleep.action_record_id != action_record_id:
                raise RuntimeError(f"Wake for action {action_record_id} conflicts with memory sleep {self.sleep}.")
            self.sleep.ending = True
            if self.worker is not None and self.worker.request.action_record_id == action_record_id:
                self.worker.cancel.set()

    def publish_ready(self) -> int:
        """Apply ready units in child-before-parent order at the caller's boundary."""
        published = 0
        with self.store.lock:
            while self.ready:
                ready = self.ready[0]
                if ready.preparation.publish(ready.unit):
                    ready.progress.published += 1
                    published += 1
                else:
                    ready.progress.deferred += 1
                self.ready.popleft()
        return published

    def close_preparation(self) -> None:
        """Fence preparation without interrupting the current worker."""
        self.closing.set()
