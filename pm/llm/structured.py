"""Repair model JSON until its schema accepts it or its context is full.

Only syntax and declared validation errors receive repair turns. Completion,
measurement, recording, and programming errors keep their original boundary.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import inspect
import json
from functools import partial
from threading import Event
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from pm.llm.prompt_encoding import TurnInput

T = TypeVar("T", bound=BaseModel)
_MISSING = object()


class JsonOutputError(ValueError):
    """The model answer does not contain one valid JSON object."""


class ModelFormatError(JsonOutputError):
    """Repair history exhausted the input budget before a valid answer."""


def extract_json(text: str) -> dict:
    """Read an object, including an object inside a Markdown fence.

    Raise JsonOutputError for missing or malformed JSON. A non-text result
    raises TypeError because it violates the completion interface.
    """
    if not isinstance(text, str):
        raise TypeError("The completion must return text.")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise JsonOutputError("The model returned no JSON object.")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as error:
        raise JsonOutputError(f"The model returned invalid JSON: {error}") from None


def context_measure(complete: Callable) -> Callable | None:
    """Find context measurement on a completion or its bound model."""
    measure = getattr(complete, "measure_context", None)
    if measure is not None:
        return measure
    owner = getattr(complete, "__self__", None)
    if owner is not None:
        return getattr(owner, "measure_context", None)
    wrapped = getattr(complete, "func", None)
    return context_measure(wrapped) if wrapped is not None else None


class _Repair:
    """Own one uncommitted conversation and its validation failures."""

    def __init__(self, messages, schema, validation_context, on_rejection, validate, cancel):
        self.turns = list(messages)
        self.schema = schema
        self.validation_context = validation_context
        self.on_rejection = on_rejection
        self.validate = validate
        self.cancel = cancel
        self.attempt = 0
        self.last_error = ""

    def check_cancelled(self) -> None:
        """Stop before validation, measurement, or another completion."""
        if self.cancel is not None and self.cancel.is_set():
            raise asyncio.CancelledError

    def accept(self, raw: str) -> T | None:
        """Validate an answer or append its correction turn."""
        self.check_cancelled()
        self.attempt += 1
        try:
            payload = extract_json(raw)
            return (self.validate(payload, raw) if self.validate is not None else
                    self.schema.model_validate(payload, context=self.validation_context))
        except (JsonOutputError, ValidationError) as error:
            problem = str(error)
        except ValueError as error:
            if self.validate is None:
                raise
            problem = str(error)
        self.last_error = problem
        if self.on_rejection is not None:
            self.on_rejection(self.attempt, raw, problem)
        self.turns = [*self.turns, {"role": "assistant", "content": raw},
                      {"role": "user", "content":
                       f"The answer failed JSON or schema validation: {problem}\n"
                       "Return one corrected JSON object only. Keep the original task facts and source IDs. "
                       "Do not use Markdown fences. Escape quotation marks inside strings."}]
        return None

    def check_budget(self, measurement: dict) -> None:
        """Reject a repair that exceeds the reserved input budget."""
        self.check_cancelled()
        if measurement.get("measurement_complete") is False:
            raise ModelFormatError("Cannot establish the repair budget for unmeasured media.")
        if measurement["input_tokens"] > measurement["input_budget_tokens"]:
            raise ModelFormatError(
                f"JSON repair exhausted the input context after {self.attempt} invalid answers "
                f"({measurement['input_tokens']} input tokens; budget {measurement['input_budget_tokens']}). "
                f"Last validation error: {self.last_error}")


def complete_json(complete: Callable, messages: Sequence[TurnInput], schema: type[T], *,
                  measure: Callable | None = None, validation_context: dict | None = None,
                  on_rejection: Callable[[int, str, str], None] | None = None,
                  validate: Callable[[dict, str], T] | None = None,
                  initial: object = _MISSING, cancel: Event | None = None, **generation) -> T:
    """Return validated model output and append errors as repair turns.

    ``initial`` validates an existing agent result before any new completion.
    ``measure`` returns input_tokens and input_budget_tokens with output space
    reserved. It defaults to the completion model's measure_context method.
    Missing measurement raises ValueError when repair is necessary. Cancellation
    propagates. No tool actions or accepted state changes occur here.
    """
    if cancel is None and isinstance(complete, partial):
        cancel = complete.keywords.get("cancel")
    state = _Repair(messages, schema, validation_context, on_rejection, validate, cancel)
    measure = measure or context_measure(complete)
    while True:
        state.check_cancelled()
        if state.attempt:
            if measure is None:
                raise ValueError("JSON repair requires context measurement from the selected model.")
            state.check_budget(measure(state.turns, cancel=cancel, **generation))
        raw = (initial if isinstance(initial, str) else json.dumps(initial, ensure_ascii=False)) if initial is not _MISSING else complete(
            state.turns, cancel=cancel, **generation)
        initial = _MISSING
        result = state.accept(raw)
        if result is not None:
            return result


async def complete_json_async(complete: Callable, messages: Sequence[TurnInput], schema: type[T], *,
                              measure: Callable | None = None, validation_context: dict | None = None,
                              on_rejection: Callable[[int, str, str], None] | None = None,
                              validate: Callable[[dict, str], T] | None = None,
                              initial: object = _MISSING, cancel: Event | None = None, **generation) -> T:
    """Apply complete_json's repair policy to an asynchronous completion.

    Synchronous context measurement runs in a worker thread. Cancellation
    reaches measurement and completion through the supplied event.
    """
    if cancel is None and isinstance(complete, partial):
        cancel = complete.keywords.get("cancel")
    state = _Repair(messages, schema, validation_context, on_rejection, validate, cancel)
    measure = measure or context_measure(complete)
    while True:
        state.check_cancelled()
        if state.attempt:
            if measure is None:
                raise ValueError("JSON repair requires context measurement from the selected model.")
            if inspect.iscoroutinefunction(measure):
                measurement = await measure(state.turns, cancel=cancel, **generation)
            else:
                measurement = await asyncio.to_thread(measure, state.turns, cancel=cancel, **generation)
            state.check_budget(measurement)
        raw = (initial if isinstance(initial, str) else json.dumps(initial, ensure_ascii=False)) if initial is not _MISSING else await complete(
            state.turns, cancel=cancel, **generation)
        initial = _MISSING
        result = state.accept(raw)
        if result is not None:
            return result


def repair_agent_result(result, schema: type[T], model, *, instruction: str,
                        cancel: Event | None = None, validate=None) -> T:
    """Validate an agent's final answer without replay of its tool actions.

    A correction uses the same endpoint and the agent's final text. It has no
    tools. Valid answers require no model activation or extra completion.
    """
    from pm.llm.activation import ModelActivation
    from pm.llm.config import Connection
    from pm.llm.stage_model import LazyStageModel
    connection = Connection(base_url=model.base_url, model=model.model, api_key=model.api_key,
                            context_window=model.context_window)
    stage = LazyStageModel(connection, ModelActivation(), live_test=model.live_test)
    messages = [{"role": "system", "content":
                 "Correct the final report of a completed agent session. No tools are available. "
                 "Keep recorded facts and source IDs. Do not invent tool results or repeat actions. "
                 "The required JSON schema is: " + json.dumps(schema.model_json_schema(), ensure_ascii=False)},
                {"role": "user", "content": instruction}]
    initial = result.result_json if result.result_json is not None else result.stdout
    return complete_json(stage.complete, messages, schema, initial=initial, cancel=cancel, validate=validate,
                         measure=stage.measure_context, enable_thinking=False)
