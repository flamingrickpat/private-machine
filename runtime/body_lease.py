"""Keep one actor in control of each configured Minecraft endpoint."""
from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Iterator


class BodyBusy(RuntimeError):
    """Another session owns this avatar. No new action was sent."""


class BodyLeases:
    """Own process-local leases across application instances and worker threads.

    The endpoint identifies the avatar. A caller cannot bypass ownership with
    another body label. A12 owns leases across processes and worker restarts.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._owners: dict[str, str] = {}
        self._commands: dict[str, threading.Lock] = {}

    @contextmanager
    def command(self, endpoint: str, cancel=None) -> Iterator[None]:
        """Let a read wait for the current command without taking body ownership.

        Actor and Assistant calls share this boundary. A queued actor checks
        cancellation before it can send another command.
        """
        key = endpoint.rstrip("/")
        with self._lock:
            lock = self._commands.setdefault(key, threading.Lock())
        from pm_workflows.runtime import WorkflowTerminated
        while not lock.acquire(timeout=0.1):
            if cancel is not None and cancel.is_set():
                raise WorkflowTerminated()
        try:
            if cancel is not None and cancel.is_set():
                raise WorkflowTerminated()
            yield
        finally:
            lock.release()

    @contextmanager
    def acquire(self, endpoint: str, owner: str) -> Iterator[None]:
        """Hold ownership until cleanup finishes. Raise BodyBusy on contention."""
        key = endpoint.rstrip("/")
        with self._lock:
            if key in self._owners:
                raise BodyBusy(f"Minecraft body is owned by {self._owners[key]}.")
            self._owners[key] = owner
        try:
            yield
        finally:
            with self._lock:
                del self._owners[key]

    def owner(self, endpoint: str) -> str | None:
        """Read ownership without acquiring a mutation lease."""
        with self._lock:
            return self._owners.get(endpoint.rstrip("/"))


BODY_LEASES = BodyLeases()
