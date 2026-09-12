"""Bus, envelope, and the shared module lifecycle.

Every cognitive module runs as one :class:`ShellModule`. The bus fans one
published envelope out to every module whose predicate accepts it; each
module owns its own mailbox and thread. Data delivery enqueues; control
delivery sets the stop event so termination is never queued behind data.
"""
from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Literal


def _new_id() -> str:
    return uuid.uuid4().hex


class Envelope:
    """One immutable bus message."""

    __slots__ = ("id", "source", "kind", "topic", "correlation_id", "payload", "target")

    def __init__(
        self,
        id: str,
        source: str,
        kind: Literal["data", "telemetry", "control"],
        topic: str,
        correlation_id: str,
        payload: dict,
        target: str | None = None,
    ):
        self.id = id
        self.source = source
        self.kind = kind
        self.topic = topic
        self.correlation_id = correlation_id
        self.payload = payload
        self.target = target

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Envelope) and other.id == self.id

    def __repr__(self) -> str:
        return f"Envelope({self.topic!r}, {self.id[:8]})"


class Mailbox:
    """A private queue with a condition, owned by one module."""

    def __init__(self) -> None:
        self._items: deque[Envelope] = deque()
        self._condition = threading.Condition()

    def put(self, envelope: Envelope) -> None:
        with self._condition:
            self._items.append(envelope)
            self._condition.notify()

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def take_batch(self, stop_event: threading.Event, timeout: float | None) -> list[Envelope]:
        with self._condition:
            if not self._items and not stop_event.is_set():
                self._condition.wait(timeout)
            items = list(self._items)
            self._items = deque()
            return items


class BusSystem:
    """An in-memory fan-out bus. No worker thread, no destructive get."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscriptions: list[tuple[str, Callable, Callable]] = []

    def subscribe(self, name: str, accepts: Callable, deliver: Callable) -> None:
        with self._lock:
            self._subscriptions.append((name, accepts, deliver))

    def put(self, envelope: Envelope) -> None:
        with self._lock:
            for _name, accepts, deliver in self._subscriptions:
                if accepts(envelope):
                    deliver(envelope)

    def subscribers(self) -> list[str]:
        with self._lock:
            return [name for name, _a, _d in self._subscriptions]


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
        return envelope.topic in self.DATA_TOPICS

    def deliver(self, envelope: Envelope) -> None:
        """Called by the bus under its lock; never runs a workflow here."""
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
            while not self.stop_event.is_set():
                self.step(block=True)
        except BaseException as exc:  # a crash must reach the application
            self.on_error(exc)
        finally:
            self._finish_termination()

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


__all__ = [
    "Envelope",
    "Mailbox",
    "BusSystem",
    "ShellModule",
    "Application",
    "_new_id",
]