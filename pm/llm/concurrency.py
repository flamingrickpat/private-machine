"""Run independent calls with bounded workers and stable result order.

Callbacks must use the supplied cancellation signal for external I/O. The
first failure cancels siblings. All owned workers join before it propagates.
The endpoint admission gate shares role-aware scheduling across their event
loops. See :mod:`pm.llm` for the inference subsystem overview.
"""
from __future__ import annotations

from asyncio import CancelledError
from collections.abc import AsyncGenerator, Callable, Generator, Iterable
from concurrent.futures import ThreadPoolExecutor, wait
from contextvars import ContextVar, copy_context
from dataclasses import dataclass
from enum import StrEnum
import logging
from time import monotonic
from threading import Event, Lock
from typing import TypeVar
from contextlib import asynccontextmanager, contextmanager
import asyncio

class InferenceRole(StrEnum):
    """Identify a call's scheduling purpose without granting memory authority."""

    Triage = "triage"
    Critical = "critical"
    Interaction = "interaction"
    Cognition = "cognition"
    Private = "private"


_ROLE: ContextVar[InferenceRole] = ContextVar("inference_role", default=InferenceRole.Cognition)
_ADMISSIONS: dict[str, "InferenceAdmission"] = {}
_ADMISSIONS_LOCK = Lock()
_LOGGER = logging.getLogger(__name__)


@contextmanager
def inference_role(role: InferenceRole) -> Generator[None, None, None]:
    """Scope a role to this context; copied worker contexts retain its value."""
    token = _ROLE.set(role)
    try:
        yield
    finally:
        _ROLE.reset(token)


@contextmanager
def background_inference() -> Generator[None, None, None]:
    """Mark private-loop inference through the shared role context."""
    with inference_role(InferenceRole.Private):
        yield


@dataclass(eq=False, frozen=True, slots=True)
class AdmissionTicket:
    """One request's queue identity and immutable scheduling role."""

    role: InferenceRole


class InferenceAdmission:
    """Serialize endpoint calls with urgent priority and bounded normal fairness.

    Triage precedes interaction. After four urgent calls, the oldest waiting
    cognition or private call receives a turn. A critical reservation instead
    admits only triage and critical calls until its workflow releases it.
    Active calls finish cooperatively; reservations never interrupt HTTP.
    State is process-local and protected across independent workflow threads.
    Admission logs record role, queued duration, active duration, and critical
    reservation count without prompt content or credentials.
    See :mod:`pm.llm` for the inference subsystem overview.
    """

    def __init__(self) -> None:
        self._lock: Lock = Lock()
        self._waiting: list[AdmissionTicket] = []
        self._active: AdmissionTicket | None = None
        self._reservations: dict[str, InferenceRole] = {}
        self._consecutive_urgent: int = 0

    @property
    def background_active(self) -> bool:
        """Return whether a private call currently owns the HTTP slot."""
        with self._lock:
            return self._active is not None and self._active.role == InferenceRole.Private

    def reserve(self, event_id: str, *, role: InferenceRole = InferenceRole.Critical) -> None:
        """Retain a workflow's role between calls; only Critical blocks others.

        The default preserves older foreground reservation callers while they
        migrate to explicit roles. Reusing an event ID updates its role.
        """
        with self._lock:
            self._reservations[event_id] = role

    def release(self, event_id: str) -> None:
        """Release a queued or completed event's reservation, if still present."""
        with self._lock:
            self._reservations.pop(event_id, None)

    def _next_ticket(self) -> AdmissionTicket | None:
        """Select under the lock; critical barriers suspend normal fairness."""
        triage: AdmissionTicket | None = None
        critical: AdmissionTicket | None = None
        interaction: AdmissionTicket | None = None
        normal: AdmissionTicket | None = None
        for ticket in self._waiting:
            match ticket.role:
                case InferenceRole.Triage:
                    if triage is None:
                        triage = ticket
                case InferenceRole.Critical:
                    if critical is None:
                        critical = ticket
                case InferenceRole.Interaction:
                    if interaction is None:
                        interaction = ticket
                case InferenceRole.Cognition | InferenceRole.Private:
                    if normal is None:
                        normal = ticket

        # A marked critical call also has priority if its owner has not yet
        # installed the cross-phase reservation. It cannot grant normal work.
        if InferenceRole.Critical in self._reservations.values() or critical is not None:
            return triage if triage is not None else critical
        urgent = triage if triage is not None else interaction
        if normal is not None and (urgent is None or self._consecutive_urgent >= 4):
            return normal
        return urgent

    @asynccontextmanager
    async def slot(self, cancel: Event | None = None) -> AsyncGenerator[None, None]:
        """Wait for one call slot; cancellation removes this request's ticket.

        The short async wait supports independent event loops in bus threads.
        Cancellation while queued or inside the slot releases all owned state.
        """
        queued_at = monotonic()
        admitted_at: float | None = None
        critical_reservations: int = 0
        ticket = AdmissionTicket(role=_ROLE.get())
        with self._lock:
            self._waiting.append(ticket)
        acquired = False
        try:
            while not acquired:
                if cancel is not None and cancel.is_set():
                    raise asyncio.CancelledError
                with self._lock:
                    if self._active is None and self._next_ticket() is ticket:
                        self._waiting.remove(ticket)
                        self._active = ticket
                        acquired = True
                        admitted_at = monotonic()
                        critical_reservations = sum(
                            role == InferenceRole.Critical for role in self._reservations.values()
                        )
                        if ticket.role in (InferenceRole.Triage, InferenceRole.Interaction):
                            self._consecutive_urgent = min(self._consecutive_urgent + 1, 4)
                        elif ticket.role in (InferenceRole.Cognition, InferenceRole.Private):
                            self._consecutive_urgent = 0
                if not acquired:
                    await asyncio.sleep(0.01)
            assert admitted_at is not None
            _LOGGER.info(
                "Inference admitted role=%s queue_seconds=%.3f critical_reservations=%d",
                ticket.role.value, admitted_at - queued_at, critical_reservations,
            )
            yield
        finally:
            with self._lock:
                if acquired:
                    self._active = None
                else:
                    self._waiting.remove(ticket)
            if admitted_at is not None:
                _LOGGER.info("Inference released role=%s active_seconds=%.3f",
                             ticket.role.value, monotonic() - admitted_at)
            else:
                _LOGGER.info("Inference cancelled while queued role=%s queue_seconds=%.3f",
                             ticket.role.value, monotonic() - queued_at)


def inference_admission(endpoint: str) -> InferenceAdmission:
    """Share one admission gate for each endpoint in this process."""
    with _ADMISSIONS_LOCK:
        return _ADMISSIONS.setdefault(endpoint, InferenceAdmission())

T = TypeVar("T")
R = TypeVar("R")


def ordered_parallel_map(operation: Callable[[T], R], items: Iterable[T], *, workers: int, cancel: Event) -> list[R]:
    """Return ordered results with the caller's memory and logging scopes.

    Each worker enters its own context copy. Failure signals cancellation and
    joins every worker before returning control to the publishing thread.
    """
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
        futures = [pool.submit(copy_context().run, call, item) for item in items]
        wait(futures)
    if errors:
        raise errors[0]
    return [future.result() for future in futures]
