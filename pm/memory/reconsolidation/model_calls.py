"""Validated model calls of memory reconsolidation.

Every call returns one JSON object that a pydantic schema validates. An
invalid answer gets a repair turn with the validation error. A provider
failure (``ModelError``: a filtered, cut-off, or empty completion) repeats
the same call. Transport errors and HTTP 429/5xx already retry inside
``pm.llm.http_client.JsonClient``. Other errors and cancellation propagate.

Calls of one stage can run in parallel with :meth:`ModelCalls.map`. The
number of workers is bounded. A failed call stops the calls that did not
start yet, and the first error propagates after all workers end.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from threading import Event, Lock
from typing import TypeVar

from pydantic import BaseModel

from pm.constants import (RECONSOLIDATION_PROVIDER_RETRIES, RECONSOLIDATION_TEMPERATURE,
                          RECONSOLIDATION_WORKERS)
from pm.llm.completion import TextCompletion
from pm.llm.concurrency import ordered_parallel_map
from pm.llm.http_client import ModelError

T = TypeVar("T", bound=BaseModel)
I = TypeVar("I")
R = TypeVar("R")

from pm.llm.structured import (
    complete_json, context_measure, extract_json as extract_object,
    ModelFormatError as ReconsolidationOutputError,
)


class ModelCalls:
    """Make validated model calls and count them.

    ``complete`` is a ``TextCompletion``, for example
    ``LazyStageModel.complete``. ``cancel`` reaches every call, so a stopped
    run also stops the calls in flight. ``calls`` counts every request that
    this object sent, repairs and repeats included.
    """

    def __init__(self, complete: TextCompletion, *, cancel: Event | None = None,
                 temperature: float = RECONSOLIDATION_TEMPERATURE, workers: int = RECONSOLIDATION_WORKERS):
        self.complete = complete
        self.cancel = cancel
        self.temperature = temperature
        self.workers = workers
        self.calls = 0
        self._lock = Lock()

    def _send(self, turns: list[dict]) -> str:
        """Send one request. Repeat it after a provider failure."""
        for attempt in range(RECONSOLIDATION_PROVIDER_RETRIES + 1):
            with self._lock:
                self.calls += 1
            try:
                return self.complete(turns, cancel=self.cancel, temperature=self.temperature, enable_thinking=False)
            except ModelError:
                if attempt == RECONSOLIDATION_PROVIDER_RETRIES:
                    raise
        raise AssertionError("unreachable")

    def json(self, system: str, user: str, schema: type[T], *, context: dict | None = None) -> T:
        """Return validated JSON. Repair turns accumulate until the context is full."""
        original = [{"role": "system", "content": system.strip()}, {"role": "user", "content": user.strip()}]
        return complete_json(lambda turns, **settings: self._send(turns), original, schema,
                             measure=context_measure(self.complete), validation_context=context,
                             cancel=self.cancel, temperature=self.temperature, enable_thinking=False)

    def map(self, operation: Callable[[I], R], items: Iterable[I]) -> list[R]:
        """Run ``operation`` for each item with bounded workers. Keep the input order."""
        items = list(items)
        if not items:
            return []
        if self.workers <= 1 or len(items) == 1:
            return [operation(item) for item in items]
        return ordered_parallel_map(operation, items, workers=self.workers, cancel=Event())


__all__ = ["ModelCalls", "ReconsolidationOutputError", "extract_object"]
