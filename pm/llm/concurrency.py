"""Run independent calls with bounded workers and stable result order.

Callbacks must use the supplied cancellation signal for external I/O. The
first failure cancels siblings. All owned workers join before it propagates.
"""
from __future__ import annotations

from asyncio import CancelledError
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Event, Lock
from typing import TypeVar
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
import asyncio

_BACKGROUND = ContextVar("background_inference", default=False)
_ADMISSIONS = {}
_ADMISSIONS_LOCK = Lock()


@contextmanager
def background_inference():
    """Mark private-loop inference on this execution context as background work."""
    token = _BACKGROUND.set(True)
    try:
        yield
    finally:
        _BACKGROUND.reset(token)


class InferenceAdmission:
    """Admit one HTTP completion at a time across module threads.

    Queued main and dialogue events hold foreground reservations across calls.
    Background work yields between completions. No active call is interrupted.
    """

    def __init__(self):
        self._lock = Lock()
        self._waiting = []
        self._active = False
        self._active_background = False
        self._foreground = set()

    @property
    def background_active(self) -> bool:
        """Return whether a private completion currently owns the HTTP slot."""
        with self._lock:
            return self._active and self._active_background

    def reserve(self, event_id: str) -> None:
        """Hold foreground priority until the event's workflow finishes."""
        with self._lock:
            self._foreground.add(event_id)

    def release(self, event_id: str) -> None:
        """Release a queued or completed event's foreground reservation."""
        with self._lock:
            self._foreground.discard(event_id)

    @asynccontextmanager
    async def slot(self, cancel: Event | None = None):
        """Wait in priority order. Cancellation removes the waiting request."""
        ticket = (object(), _BACKGROUND.get())
        with self._lock:
            self._waiting.append(ticket)
        acquired = False
        try:
            while not acquired:
                if cancel is not None and cancel.is_set():
                    raise asyncio.CancelledError
                with self._lock:
                    eligible = [row for row in self._waiting if not row[1]]
                    if not eligible and not self._foreground:
                        eligible = self._waiting
                    if not self._active and eligible and eligible[0] == ticket:
                        self._waiting.remove(ticket)
                        self._active = acquired = True
                        self._active_background = ticket[1]
                if not acquired:
                    await asyncio.sleep(0.01)
            yield
        finally:
            with self._lock:
                if acquired:
                    self._active = False
                    self._active_background = False
                else:
                    self._waiting.remove(ticket)


def inference_admission(endpoint: str) -> InferenceAdmission:
    """Share one admission gate for each endpoint in this process."""
    with _ADMISSIONS_LOCK:
        return _ADMISSIONS.setdefault(endpoint, InferenceAdmission())

T = TypeVar("T")
R = TypeVar("R")


def ordered_parallel_map(operation: Callable[[T], R], items: Iterable[T], *, workers: int, cancel: Event) -> list[R]:
    """Return results in input order. Failure signals cancellation and joins workers."""
    errors: list[BaseException] = []
    lock = Lock()

    def call(item: T) -> R:
        try:
            if cancel.is_set():
                raise CancelledError
            return operation(item)
        except BaseException as error:
            with lock:
                if not errors:
                    errors.append(error)
            cancel.set()
            raise

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="inference") as pool:
        futures = [pool.submit(call, item) for item in items]
        wait(futures)
    if errors:
        raise errors[0]
    return [future.result() for future in futures]
