"""Run independent calls with bounded workers and stable result order.

Callbacks must use the supplied cancellation signal for external I/O. The
first failure cancels siblings. All owned workers join before it propagates.
The endpoint admission gate shares role-aware scheduling across their event
loops. See :mod:`pm.llm` for the inference subsystem overview.
"""
from __future__ import annotations

from asyncio import CancelledError
from collections.abc import AsyncGenerator, Generator
from concurrent.futures import Future, FIRST_EXCEPTION, wait
from contextvars import ContextVar
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
    Reconsolidation = "reconsolidation"


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
    """Fill configured endpoint slots in queued role priority order.

    Triage, critical, interaction, then ordinary work receive the next free
    slot before memory reconsolidation. In-flight requests keep their slots
    until they finish or cancel. Capacity is process-local.
    """

    def __init__(self, max_concurrent_requests: int = 8) -> None:
        self._lock: Lock = Lock()
        self._waiting: list[AdmissionTicket] = []
        self._active: set[AdmissionTicket] = set()
        self.max_concurrent_requests: int = max_concurrent_requests

    @property
    def background_active(self) -> bool:
        """Return whether any private call is currently in flight."""
        with self._lock:
            return any(ticket.role == InferenceRole.Private for ticket in self._active)

    def _next_ticket(self) -> AdmissionTicket | None:
        """Choose FIFO within each role, with triage first."""
        for role in (InferenceRole.Triage, InferenceRole.Critical, InferenceRole.Interaction):
            for ticket in self._waiting:
                if ticket.role == role:
                    return ticket
        for ticket in self._waiting:
            if ticket.role != InferenceRole.Reconsolidation:
                return ticket
        return self._waiting[0] if self._waiting else None

    @asynccontextmanager
    async def slot(self, cancel: Event | None = None) -> AsyncGenerator[None, None]:
        """Own one slot until HTTP finishes; cancellation removes its ticket."""
        queued_at = monotonic()
        admitted_at: float | None = None
        ticket = AdmissionTicket(role=_ROLE.get())
        with self._lock:
            self._waiting.append(ticket)
        try:
            while admitted_at is None:
                if cancel is not None and cancel.is_set():
                    raise asyncio.CancelledError
                with self._lock:
                    limit = self.max_concurrent_requests
                    unlimited = limit <= 0 or limit >= 9999
                    if unlimited or (len(self._active) < limit and self._next_ticket() is ticket):
                        self._waiting.remove(ticket)
                        self._active.add(ticket)
                        admitted_at = monotonic()
                if admitted_at is None:
                    await asyncio.sleep(0.01)
            _LOGGER.info("Inference admitted role=%s queue_seconds=%.3f limit=%s",
                         ticket.role.value, admitted_at - queued_at, self.max_concurrent_requests)
            yield
        finally:
            with self._lock:
                if admitted_at is None:
                    self._waiting.remove(ticket)
                else:
                    self._active.remove(ticket)
            if admitted_at is not None:
                _LOGGER.info("Inference released role=%s active_seconds=%.3f",
                             ticket.role.value, monotonic() - admitted_at)


def inference_admission(endpoint: str, *, max_concurrent_requests: int = 8) -> InferenceAdmission:
    """Share configured capacity across clients using the same endpoint."""
    with _ADMISSIONS_LOCK:
        if endpoint not in _ADMISSIONS:
            _ADMISSIONS[endpoint] = InferenceAdmission(max_concurrent_requests)
        admission = _ADMISSIONS[endpoint]
        admission.max_concurrent_requests = max_concurrent_requests
        return admission


T = TypeVar("T")
R = TypeVar("R")

def wait_model_futures(futures: list[Future[T]], cancel: Event | None = None) -> list[T]:
    pending = set(futures)
    while pending:
        done, pending = wait(pending, timeout=0.1, return_when=FIRST_EXCEPTION)
        for future in done:
            if future.cancelled():
                if cancel is not None:
                    cancel.set()
                raise CancelledError
            error = future.exception()
            if error is not None:
                if cancel is not None:
                    cancel.set()
                for sibling in pending:
                    sibling.cancel()
                raise error
        if cancel is not None and cancel.is_set():
            for future in pending:
                future.cancel()
            raise CancelledError
    return [future.result() for future in futures]
