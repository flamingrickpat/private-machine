"""One module step for direct debugging and threaded execution.

Direct errors reach the caller. Thread errors reach the application handler.
Termination wakes the mailbox and runs cleanup once.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Callable

from bus.bus import BusSystem, Mailbox
from bus.envelopes import Envelope, _new_id


class ShellModule(ABC):
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
            self._thread.join()

    def stop(self) -> None:
        self.stop_event.set()
        self.mailbox.wake()

    def _run_loop(self) -> None:
        try:
            try:
                while not self.stop_event.is_set():
                    self.step(block=True)
            finally:
                self._finish_termination()
        except BaseException as exc:  # a crash must reach the application
            self.on_error(exc)

    def _finish_termination(self) -> None:
        if self._termination_done:
            return
        self._termination_done = True
        self.on_terminate()
        self.bus.put(Envelope(
            id=_new_id(),
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
            self.handle_batch(batch)
            return True

        now = self.clock()
        if self._next_tick is not None and now >= self._next_tick:
            self._next_tick = now + self.tick_seconds
            self.handle_batch([])
            return True

        return False


class Application:
    """Owns modules, an error queue, and a shared shutdown event."""

    def __init__(self, bus: BusSystem, modules: list[ShellModule]) -> None:
        self.bus = bus
        self.modules = modules
        self.shutdown_event = threading.Event()
        self.error: BaseException | None = None
        self._error_lock = threading.Lock()

    def on_error(self, exc: BaseException) -> None:
        with self._error_lock:
            if self.error is None:
                self.error = exc
        self.shutdown()

    def shutdown(self) -> None:
        self.shutdown_event.set()
        for module in self.modules:
            module.stop()

    def run(self) -> None:
        for module in self.modules:
            module.run()

    def join(self) -> None:
        for module in self.modules:
            module.join()

    def raise_if_error(self) -> None:
        if self.error is not None:
            raise self.error
