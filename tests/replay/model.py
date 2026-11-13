"""The one model seam every workflow already shares, made recordable.

``build_app`` hands ``stage_model.complete`` to every cognitive module and to
every child-stage tools factory, so overriding that single method covers main
cognition, early triage, the private loop, and its child kernels. Context
measurement is overridden for the same reason: it reaches the endpoint too.

:class:`ScriptedModel` *is* a :class:`pm.llm.stage_model.LazyStageModel`, so it
can be passed anywhere the real model is accepted. In ``record`` mode it calls
the real endpoint and keeps the answer. In ``replay`` mode it never touches the
network; it returns the recorded answer and fails when the architecture asks
something the recording does not hold.

Reading map: :class:`ScriptedModel` is the wrapper. :func:`open_scripted_model`
builds it in the requested mode.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Event
from typing import Literal, cast

from pm.llm.activation import ModelActivation
from pm.llm.config import Connection
from pm.llm.contracts import ContextMeasurement
from pm.llm.prompt_encoding import TurnInput
from pm.llm.stage_model import LazyStageModel

from tests.replay.script import (
    CallKind, RecordingReader, RecordingWriter, ReplayDivergence, open_recording, request_text,
)


class ScriptedModel(LazyStageModel):
    """Serve one recorded response per request, or record the live one.

    Recording and replay differ only in where the answer comes from. Retries,
    repairs, and cancellation behave exactly as they do in production because
    those live in the stages, not here.
    """

    def __init__(self, connection: Connection, owner: ModelActivation,
                 recording: RecordingWriter | RecordingReader) -> None:
        super().__init__(connection, owner, live_test=True)
        self.recording = recording
        self._live_requests = 0
        #: Called before every request is served. A scenario uses it to deliver
        #: outside input at an exact point inside a private invocation.
        self.call_hook: Callable[[], None] | None = None

    @property
    def mode(self) -> Literal["record", "replay"]:
        """Return which side of the recording this model is on."""
        return "record" if isinstance(self.recording, RecordingWriter) else "replay"

    @property
    def request_count(self) -> int:
        """Return real provider requests made through this model."""
        return self._live_requests

    @property
    def recorded_calls(self) -> int:
        """Return how many calls the recording holds or has received."""
        if isinstance(self.recording, RecordingWriter):
            return self.recording.count
        return self.recording.position

    def complete(self, messages: Sequence[TurnInput], **generation: object) -> str:
        """Return the recorded completion, or call the endpoint and retain it.

        Cancellation is honoured the same way the real client honours it: a set
        cancel event ends the call instead of returning text. Otherwise a
        replayed run would ignore a real cooperative cancellation signal.
        """
        cancel = generation.get("cancel")
        if isinstance(cancel, Event) and cancel.is_set():
            raise asyncio.CancelledError
        if self.call_hook is not None:
            self.call_hook()
        request = request_text("completion", messages, generation)
        if isinstance(self.recording, RecordingReader):
            return self.recording.next_call("completion", request).response
        started = time.monotonic()
        response = super().complete(messages, **generation)
        self._live_requests = super().request_count
        self.recording.append("completion", request, response, time.monotonic() - started)
        return response

    def measure_context(self, messages: Sequence[TurnInput], *, cancel: Event | None = None,
                        **generation: object) -> ContextMeasurement:
        """Return the recorded measurement, or measure against the real endpoint.

        A measurement is not inference, but it does reach the endpoint for
        capacity discovery. Recording it keeps replay fully offline and keeps
        the replayed prompt accounting identical to the recording.
        """
        request = request_text("measurement", messages, {**generation, "cancel": cancel})
        if isinstance(self.recording, RecordingReader):
            row = self.recording.next_call("measurement", request)
            return read_measurement(row.response)
        started = time.monotonic()
        measurement = super().measure_context(messages, cancel=cancel, **generation)
        self.recording.append("measurement", request, json.dumps(measurement, sort_keys=True),
                              time.monotonic() - started)
        return measurement

    def unasked_calls(self) -> int:
        """Return recorded calls this replay never asked for.

        A scenario that no longer reaches part of the recorded work has changed
        behaviour, so the caller reports this rather than ignoring it.
        """
        if not isinstance(self.recording, RecordingReader):
            return 0
        return len(self.recording.unasked())


def read_measurement(text: str) -> ContextMeasurement:
    """Parse a recorded measurement and fail loudly if a field is absent.

    ``ContextMeasurement`` is a ``TypedDict``, so the shape is checked here at
    the recording boundary rather than trusted from JSON.
    """
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("A recorded context measurement must be a JSON object.")
    required = {"context_window", "context_source", "input_tokens", "measurement",
                "input_fraction", "input_budget_tokens", "measurement_complete"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"A recorded context measurement is missing {sorted(missing)}.")
    return cast(ContextMeasurement, raw)


def open_scripted_model(connection: Connection, directory: Path,
                        mode: Literal["record", "replay"]) -> ScriptedModel:
    """Build the scripted model in the requested mode."""
    return ScriptedModel(connection, ModelActivation(), open_recording(directory, mode))


__all__ = ["ScriptedModel", "open_scripted_model", "ReplayDivergence", "CallKind", "read_measurement"]
