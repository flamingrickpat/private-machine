"""Private mailboxes and isolated fan-out.

Owns: one in-memory queue per module (``Mailbox``) and the fan-out of a
published envelope to every subscriber (``BusSystem``). Inputs are
``Envelope`` values from modules and workflows; the only output is a call to a
subscriber. Nothing here decides what a message means or when a module runs.

Invariants: an envelope reaches each subscriber as a detached copy created
from its JSON text, so one subscriber cannot change another subscriber's
payload. Callbacks run outside the subscription lock, so a callback may
publish again without deadlocking.

Failure modes: a payload that is not JSON serializable raises ``TypeError`` on
publication, and a callback error propagates to the publisher. Delivery is not
durable replay; a process restart loses queued envelopes.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Callable

from pm.bus.envelopes import Envelope

#: Predicate that decides whether a module wants a delivered envelope.
AcceptsEnvelope = Callable[[Envelope], bool]
#: Sink that a module or an observer receives one envelope through.
DeliverEnvelope = Callable[[Envelope], None]


class Mailbox:
    """A private queue with a condition, owned by one module."""

    def __init__(self) -> None:
        self._items: deque[Envelope] = deque()
        self._condition = threading.Condition()
        self._claimed = False

    def put(self, envelope: Envelope) -> None:
        """Queue one envelope and wake a waiting reader."""
        with self._condition:
            self._items.append(envelope)
            self._condition.notify()

    def wake(self) -> None:
        """Release every waiting reader, for example after a stop request."""
        with self._condition:
            self._condition.notify_all()

    def pending(self) -> int:
        """Return queued envelopes without removing them."""
        with self._condition:
            return len(self._items) + int(self._claimed)

    def take_batch(self, stop_event: threading.Event, timeout: float | None, *, claim: bool = False) -> list[Envelope]:
        """Return and clear the queued envelopes after at most ``timeout`` seconds.

        ``claim`` marks the module as mid-delivery, so ``pending`` keeps
        reporting the work until ``release`` runs after synchronous publication.
        """
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
        self._subscriptions: list[tuple[str, AcceptsEnvelope, DeliverEnvelope]] = []
        self.observer: Callable[[Envelope, str], None] | None = None

    def subscribe(self, name: str, accepts: AcceptsEnvelope, deliver: DeliverEnvelope) -> None:
        """Register the predicate and the sink of one module under its name."""
        with self._lock:
            self._subscriptions.append((name, accepts, deliver))

    def put(self, envelope: Envelope) -> None:
        """Deliver a detached copy of ``envelope`` to every accepting subscriber.

        The envelope is serialized once before fan-out, so a subscriber cannot
        change the payload that a later subscriber or the observer receives.
        """
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
        """Return the registered module names in subscription order."""
        with self._lock:
            return [name for name, _a, _d in self._subscriptions]
