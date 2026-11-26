"""One module step for direct debugging and threaded execution.

Direct errors reach the caller. Thread errors reach the application handler.
Termination wakes the mailbox and runs cleanup once.
"""
from __future__ import annotations

from pm.host.cognition_profiler import CognitionProfiler

from pm.introspection.capabilities import CapabilityOverlay

import threading
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

from pm.bus.bus import BusSystem, Mailbox
from pm.bus.envelopes import Envelope, new_id
from pm.host.diagnostics import join_thread
from pm import vclock

if TYPE_CHECKING:
    from pm.memory.store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AdmittedCycle:
    """The current batch and the dispatches it publishes before returning."""

    envelopes: list[Envelope]
    dispatch_ids: set[str] = field(default_factory=set[str])


class BusModule(ABC):
    """Common inbox, thread, and termination behavior for every module."""

    DATA_TOPICS: frozenset[str] = frozenset()

    def __init__(
        self,
        module_id: str,
        application: Application,
        tick_seconds: float | None = None,
    ) -> None:
        self.module_id = module_id
        self.application = application
        self.bus = application.bus
        self.tick_seconds = tick_seconds
        self.mailbox = Mailbox()
        self.stop_event = threading.Event()
        self.invocation_id: str | None = None
        self.next_tick_seconds: float | None = None
        if tick_seconds is not None:
            self.next_tick_seconds = self.application.clock.monotonic() + tick_seconds
        self._thread: threading.Thread | None = None
        self._termination_done = False
        self.draining = False
        self.active_cycle: AdmittedCycle | None = None

    @property
    def working(self) -> bool:
        return self.active_cycle is not None

    # ------------------------------------------------------------ interest

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.kind == "control":
            return envelope.topic == "terminate" and envelope.target == self.module_id
        return envelope.topic in self.DATA_TOPICS and envelope.target in {None, self.module_id}

    def deliver(self, envelope: Envelope) -> None:
        """Called outside the bus lock; never runs a workflow here."""
        if envelope.kind == "control":
            self.stop_event.set()
            self.mailbox.wake()
        else:
            self.mailbox.put(envelope)

    # ------------------------------------------------------------ lifecycle

    @abstractmethod
    def handle_batch(self, batch: list[Envelope]) -> None:
        """Process one input batch or one timer tick."""

    def on_terminate(self) -> None:
        """Hook for subclasses to emit final state once before shutdown."""

    def run(self) -> None:
        self._thread = threading.Thread(
            target=self._run_loop, name=self.module_id, daemon=False,
        )
        self._thread.start()

    def join(self) -> None:
        if self._thread is not None:
            join_thread(self._thread)

    def stop(self) -> None:
        if not self.stop_event.is_set():
            logger.info("Module stop requested module=%s invocation=%s", self.module_id, self.invocation_id)
        self.stop_event.set()
        self.mailbox.wake()

    def _run_loop(self) -> None:
        logger.info("Module thread started module=%s native_id=%s", self.module_id, threading.get_native_id())
        try:
            try:
                while not self.stop_event.is_set():
                    self.step(block=True)
            finally:
                self._finish_termination()
        except BaseException as exc:  # a crash must reach the application
            logger.exception("Module thread crashed module=%s invocation=%s", self.module_id, self.invocation_id)
            self.application.on_error(exc)
        finally:
            logger.info("Module thread stopped module=%s invocation=%s", self.module_id, self.invocation_id)

    def publish_cycle_action(self, envelope: Envelope) -> None:
        """Identify a synchronous dispatch before its recipient can claim it."""
        with self.application.admission_lock:
            if self.active_cycle is not None:
                self.active_cycle.dispatch_ids.add(envelope.id)
        self.bus.put(envelope)

    def _logged_batch(self, batch: list[Envelope]) -> None:
        """Record work boundaries without copying private input into diagnostics.

        Empty batches and scheduler batches run silently to limit log noise.
        Other batches record their start and exit without private payloads.
        """
        started = time.monotonic()
        log_work = bool(batch) and str(self.module_id) != "scheduler"
        if log_work:
            logger.debug("Module work start module=%s topics=%s correlations=%s", self.module_id,
                         [item.topic for item in batch], [item.correlation_id for item in batch])
        try:
            with vclock.busy():
                self.handle_batch(batch)
        finally:
            if log_work:
                logger.debug("Module work exit module=%s invocation=%s seconds=%.3f", self.module_id,
                             self.invocation_id, time.monotonic() - started)

    def _finish_termination(self) -> None:
        if self._termination_done:
            return
        self._termination_done = True
        self.on_terminate()
        self.bus.put(Envelope(
            id=new_id(),
            source=self.module_id,
            kind="telemetry",
            topic="module.terminated",
            correlation_id="",
            payload={"module_id": self.module_id, "invocation_id": self.invocation_id},
        ))

    def step(self, *, block: bool = False) -> bool:
        """Run at most one batch or one timer tick. Returns True only on work."""
        app = self.application
        cycle: AdmittedCycle | None = None
        while not self.stop_event.is_set():
            wait: float | None = None
            with app.admission_lock:
                if self.stop_event.is_set():
                    break
                batch = self.mailbox.take_batch(self.stop_event, 0)
                if app.draining.is_set():
                    batch = app.shutdown_batch(self, batch)
                if batch:
                    cycle = AdmittedCycle(batch)
                elif not self.draining and self.next_tick_seconds is not None:
                    now = app.clock.monotonic()
                    wait = self.next_tick_seconds - now
                    if wait <= 0:
                        if self.tick_seconds is None:
                            raise RuntimeError(f"Module {self.module_id} has a timer without an interval.")
                        self.next_tick_seconds = now + self.tick_seconds
                        cycle = AdmittedCycle([])
                if cycle is not None:
                    self.active_cycle = cycle
                    break
            if not block:
                return False
            self.mailbox.wait_for_items(self.stop_event, app.draining, vclock.wait_cap(wait))

        if cycle is None:
            self._finish_termination()
            return False
        try:
            self._logged_batch(cycle.envelopes)
        finally:
            with app.admission_lock:
                self.active_cycle = None
        return True


class Application:
    """Own modules, the first error, and optional worker lifecycle hooks.

    A graceful shutdown closes admission, waits for active work, and joins
    the module and worker threads. The final checkpoint follows cleanup.
    An immediate shutdown signals cancellation. External services must remain
    available until the work that uses them finishes.
    """

    def __init__(self, bus: BusSystem, modules: list[BusModule], *,
                 clock: vclock.SystemClock | None = None) -> None:
        self.bus = bus
        self.ghost: MemoryStore | None = None
        self.profiler: CognitionProfiler | None = None
        self.capability_overlay: CapabilityOverlay | None = None
        self.modules = modules
        self.clock: vclock.SystemClock = clock if clock is not None else vclock.current_clock()
        self.shutdown_event = threading.Event()
        self.error: BaseException | None = None
        self._error_lock = threading.Lock()
        self.stop_workers: Callable[[], None] | None = None
        self.join_workers: Callable[[], None] | None = None
        self.draining = threading.Event()
        self.admission_lock = threading.Lock()
        self.shutdown_request_lock = threading.Lock()
        self._join_started = False
        self.drained = threading.Event()
        self.frozen = threading.Event()
        self._drain_thread: threading.Thread | None = None

    def idle(self) -> bool:
        """Return whether runnable module and managed work is quiet.

        Parked workflows do not count as runnable work. A due timer prevents
        virtual time advancement unless graceful shutdown already fenced it.
        """
        for module in self.modules:
            if module.working or module.mailbox.pending():
                return False
            if (not module.draining and module.next_tick_seconds is not None
                    and self.clock.monotonic() >= module.next_tick_seconds):
                return False
        return self.background_work_idle()

    def background_work_idle(self) -> bool:
        """Whether owned work outside the bus can make progress."""
        return True

    def request_shutdown(self) -> None:
        """Close intake and finish admitted work in a separate thread.

        A normal stop has no elapsed-time limit. A runtime failure still uses
        immediate cancellation. Parked workflows retain their durable journals.
        """
        with self.shutdown_request_lock:
            if self.draining.is_set() or self.shutdown_event.is_set():
                return
            self.close_shutdown_intake()
            with self.admission_lock:
                self.draining.set()
                self.capture_shutdown_admissions()
                for module in self.modules:
                    module.draining = True
                    module.mailbox.wake()
            self._drain_thread = threading.Thread(target=self._drain, name="shutdown-drain", daemon=False)
            self._drain_thread.start()

    def close_shutdown_intake(self) -> None:
        """Close external stimulus admission before fencing module work."""

    def capture_shutdown_admissions(self) -> None:
        """Capture active work under the admission lock at the cutoff."""

    def shutdown_can_finish_cognition(self, pending_sources: set[int]) -> bool:
        """Whether an admitted cycle can clear a publication hold during shutdown."""
        return False

    def shutdown_batch(self, module: BusModule, batch: list[Envelope]) -> list[Envelope]:
        """Select continuations of admitted work under the admission lock."""
        return []

    def active_worker_ids(self) -> list[str]:
        """List worker threads still completing their admitted work."""
        return []

    def finish_shutdown_memory(self) -> None:
        """Finalize domain memory after all work has joined."""

    def _drain(self) -> None:
        """Wait for admitted work, join it, and write the final checkpoint."""
        try:
            started = time.monotonic()
            next_report = started
            ghost = self.ghost
            if ghost is not None and ghost.checkpoint is not None and not self.idle():
                print(f"PM shutdown: saving recovery checkpoint to {ghost.checkpoint}", flush=True)
                ghost.save()
            while not self.shutdown_event.is_set():
                workers = self.active_worker_ids()
                with self.admission_lock:
                    modules = [module.module_id for module in self.modules if module.working]
                    queued = any(module.mailbox.pending() for module in self.modules)
                    if not (modules or workers or queued):
                        for module in self.modules:
                            module.stop()
                        break
                now = time.monotonic()
                if now >= next_report:
                    print(f"PM shutdown: waiting {now - started:.1f}s, modules={modules}, runs={workers}", flush=True)
                    next_report = now + 5.0
                self.shutdown_event.wait(.02)
            if self.shutdown_event.is_set():
                return
            print("PM shutdown: joining work and finalizing memory", flush=True)
            self._join_owned_work(graceful=True)
            if self.error is not None or self.shutdown_event.is_set():
                return
            self.finish_shutdown_memory()
            if ghost is not None and ghost.checkpoint is not None:
                ghost.save()
                print(f"PM shutdown complete: checkpoint {ghost.checkpoint} ({time.monotonic() - started:.1f}s)", flush=True)
            else:
                print("PM shutdown complete", flush=True)
            self.drained.set()
            self.shutdown_event.set()
        except BaseException as error:
            self.on_error(error)

    def on_error(self, exc: BaseException) -> None:
        logger.error("Application failure", exc_info=(type(exc), exc, exc.__traceback__))
        with self._error_lock:
            if self.error is None:
                self.error = exc
        self.shutdown()

    def shutdown(self) -> None:
        """Signal every module and owned worker without joining their threads."""
        if not self.shutdown_event.is_set():
            logger.info("Application shutdown requested")
        self.shutdown_event.set()
        for module in self.modules:
            module.stop()
        self.cancel_workers()

    def cancel_workers(self) -> None:
        """Signal immediate cancellation to owned workers."""
        if self.stop_workers is not None:
            self.stop_workers()

    def run(self) -> None:
        logger.info("Application starting modules=%s", [module.module_id for module in self.modules])
        for module in self.modules:
            module.run()

    def finish_workers(self, *, graceful: bool) -> None:
        """Join owned workers after module threads stop."""
        if self.join_workers is not None:
            self.join_workers()

    def _join_owned_work(self, *, graceful: bool) -> None:
        if self._join_started:
            return
        self._join_started = True
        for module in self.modules:
            module.join()
        self.finish_workers(graceful=graceful and self.error is None)

    def join(self) -> None:
        """Wait for the drain or join work after immediate cancellation."""
        with self.shutdown_request_lock:
            drain_thread = self._drain_thread
        if drain_thread is not None and drain_thread is not threading.current_thread():
            drain_thread.join()
        self._join_owned_work(graceful=False)
        logger.info("Application threads joined")

    def raise_if_error(self) -> None:
        if self.error is not None:
            raise self.error
