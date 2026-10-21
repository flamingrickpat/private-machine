"""One module step for direct debugging and threaded execution.

Direct errors reach the caller. Thread errors reach the application handler.
Termination wakes the mailbox and runs cleanup once.
"""
from __future__ import annotations

import threading
import logging
import time
from abc import ABC, abstractmethod
from typing import Callable

from pm.bus.bus import BusSystem, Mailbox
from pm.bus.envelopes import Envelope, new_id
from pm.host.diagnostics import join_thread

logger = logging.getLogger(__name__)


class BusModule(ABC):
    """Common inbox, thread, and termination behavior for every module."""

    DATA_TOPICS: frozenset[str] = frozenset()

    def __init__(
        self,
        module_id: str,
        bus: BusSystem,
        on_error: Callable[[BaseException], None],
        clock: Callable[[], float],
        tick_seconds: float | None = None,
    ) -> None:
        self.module_id = module_id
        self.bus = bus
        self.on_error = on_error
        self.clock = clock
        self.tick_seconds = tick_seconds
        self.mailbox = Mailbox()
        self.stop_event = threading.Event()
        self.invocation_id: str | None = None
        self._next_tick: float | None = None
        if tick_seconds is not None:
            self._next_tick = self.clock() + tick_seconds
        self._thread: threading.Thread | None = None
        self._termination_done = False

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
            self.on_error(exc)
        finally:
            logger.info("Module thread stopped module=%s invocation=%s", self.module_id, self.invocation_id)

    def _logged_batch(self, batch: list[Envelope]) -> None:
        """Record work boundaries without copying private input into diagnostics."""
        if not batch and not hasattr(self, "workflow"):
            self.handle_batch(batch)
            return
        started = time.monotonic()
        logger.debug("Module work start module=%s topics=%s correlations=%s", self.module_id,
                     [item.topic for item in batch], [item.correlation_id for item in batch])
        try:
            self.handle_batch(batch)
        finally:
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
        start = self.clock()
        if self.stop_event.is_set():
            self._finish_termination()
            return False

        if block:
            wait: float | None
            if self._next_tick is not None:
                wait = max(0.0, self._next_tick - start)
            else:
                wait = None
        else:
            wait = 0.0

        batch = self.mailbox.take_batch(self.stop_event, wait)

        if self.stop_event.is_set():
            self._finish_termination()
            return False

        if batch:
            self._logged_batch(batch)
            return True

        now = self.clock()
        if self._next_tick is not None and now >= self._next_tick:
            self._next_tick = now + self.tick_seconds
            self._logged_batch([])
            return True

        return False


class Application:
    """Own modules, the first error, and optional worker lifecycle hooks.

    The assembly supplies idempotent worker hooks. Shutdown signals workers;
    join waits for them after the module threads finish. Hook failures raise
    to the caller. External services must remain available until join returns.
    """

    def __init__(self, bus: BusSystem, modules: list[BusModule]) -> None:
        self.bus = bus
        self.modules = modules
        self.shutdown_event = threading.Event()
        self.error: BaseException | None = None
        self._error_lock = threading.Lock()
        self.stop_workers: Callable[[], None] | None = None
        self.join_workers: Callable[[], None] | None = None

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
        if self.stop_workers is not None:
            self.stop_workers()

    def run(self) -> None:
        logger.info("Application starting modules=%s", [module.module_id for module in self.modules])
        for module in self.modules:
            module.run()

    def join(self) -> None:
        """Wait for module and worker cleanup. Propagate worker-hook failures."""
        for module in self.modules:
            module.join()
        if self.join_workers is not None:
            self.join_workers()
        logger.info("Application shutdown complete")

    def raise_if_error(self) -> None:
        if self.error is not None:
            raise self.error
