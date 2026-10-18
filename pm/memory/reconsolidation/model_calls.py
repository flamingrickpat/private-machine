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

import json
import re
from collections.abc import Callable, Iterable
from threading import Event, Lock
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from pm.constants import (RECONSOLIDATION_PROVIDER_RETRIES, RECONSOLIDATION_REPAIRS, RECONSOLIDATION_TEMPERATURE,
                          RECONSOLIDATION_WORKERS)
from pm.llm.completion import TextCompletion
from pm.llm.concurrency import ordered_parallel_map
from pm.llm.http_client import ModelError

T = TypeVar("T", bound=BaseModel)
I = TypeVar("I")
R = TypeVar("R")

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ReconsolidationOutputError(RuntimeError):
    """The model gave no valid answer after every repair turn."""


def extract_object(text: str) -> dict:
    """Return the outermost JSON object in ``text``. Raise ValueError when there is none."""
    match = _JSON_OBJECT.search(text or "")
    if match is None:
        raise ValueError("The answer holds no JSON object.")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("The answer is JSON, but not an object.")
    return payload


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
        """Return the validated answer to one system and one user turn.

        ``context`` is the pydantic validation context, for example the
        allowed line numbers of a fact list. Each repair keeps the task and
        only the latest rejected answer.
        """
        # Plain role/content turns: pm.memory does not import pm.context_render.
        original = [{"role": "system", "content": system.strip()}, {"role": "user", "content": user.strip()}]
        turns = original
        error_text = ""
        for _ in range(RECONSOLIDATION_REPAIRS + 1):
            raw = self._send(turns)
            try:
                return schema.model_validate(extract_object(raw), context=context)
            except (ValueError, ValidationError) as error:
                error_text = str(error)
                turns = [*original, {"role": "assistant", "content": raw}, {"role": "user", "content":
                    f"The answer failed validation: {error_text}\n"
                    "Write one corrected JSON object only. Keep the task facts and the line numbers. "
                    "Escape quotation marks inside strings. Do not use Markdown fences."}]
        raise ReconsolidationOutputError(f"{schema.__name__} stayed invalid after {RECONSOLIDATION_REPAIRS} repairs: "
                                         f"{error_text}")

    def map(self, operation: Callable[[I], R], items: Iterable[I]) -> list[R]:
        """Run ``operation`` for each item with bounded workers. Keep the input order."""
        items = list(items)
        if not items:
            return []
        if self.workers <= 1 or len(items) == 1:
            return [operation(item) for item in items]
        return ordered_parallel_map(operation, items, workers=self.workers, cancel=Event())


__all__ = ["ModelCalls", "ReconsolidationOutputError", "extract_object"]
