"""Private mailboxes and isolated fan-out.

Invalid payloads and callback errors propagate. Callbacks select or enqueue
messages outside the subscription lock. Delivery is not durable replay.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Callable

from pm.bus.envelopes import Envelope


class Mailbox:
    """A private queue with a condition, owned by one module."""

    def __init__(self) -> None:
        self._items: deque[Envelope] = deque()
        self._condition = threading.Condition()
        self._claimed = False

    def put(self, envelope: Envelope) -> None:
        with self._condition:
            self._items.append(envelope)
            self._condition.notify()

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def pending(self) -> int:
        """Return queued envelopes without removing them."""
        with self._condition:
            return len(self._items) + int(self._claimed)

    def take_batch(self, stop_event: threading.Event, timeout: float | None, *, claim: bool = False) -> list[Envelope]:
        with self._condition:
            if not self._items and not stop_event.is_set():
                self._condition.wait(timeout)
            items = list(self._items)
            self._items = deque()
            if claim and items:
                self._claimed = True
            return items

    def release(self) -> None:
        """End a module's claimed delivery after all synchronous publication."""
        with self._condition:
            self._claimed = False


class BusSystem:
    """An in-memory fan-out bus with private deliveries.

    An optional observer receives a detached envelope and recipient before
    delivery. Observer errors propagate. Observation must not publish messages
    recursively or block on work that requires this delivery to finish.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subscriptions: list[tuple[str, Callable, Callable]] = []
        self.observer: Callable[[Envelope, str], None] | None = None

    def subscribe(self, name: str, accepts: Callable, deliver: Callable) -> None:
        with self._lock:
            self._subscriptions.append((name, accepts, deliver))

    def put(self, envelope: Envelope) -> None:
        encoded = envelope.to_json()
        with self._lock:
            subscriptions = list(self._subscriptions)
        for _name, accepts, deliver in subscriptions:
            delivered = Envelope.from_json(encoded)
            if accepts(delivered):
                if self.observer is not None:
                    self.observer(Envelope.from_json(encoded), _name)
                deliver(delivered)

    def subscribers(self) -> list[str]:
        with self._lock:
            return [name for name, _a, _d in self._subscriptions]
