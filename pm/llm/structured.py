"""Repair model JSON until its schema accepts it or its context is full.

The repair loop owns one uncommitted conversation. A malformed or schema-
rejected answer adds a correction turn that keeps the original task facts and
source ids; a rejected answer never becomes accepted state. Only syntax and
declared validation errors receive repair turns. Completion, measurement,
recording, and programming errors keep their original boundary and propagate.

Reading map: :func:`extract_json` reads the object, :class:`_Repair` owns the
conversation and its rejections, :func:`complete_json` and
:func:`complete_json_async` run the loop for synchronous and asynchronous
completions, and :func:`repair_agent_result` applies the same policy to the
final report of an agent session without replaying its tool actions.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
import inspect
import json
from functools import partial
from threading import Event
from typing import TypeVar, Protocol, runtime_checkable, cast, Awaitable
from types import MethodType

from json_repair import json_repair
from json_repair.schema_repair import SchemaDefinitionError, schema_from_input
from pm_workflows.protocol import AgentResult
from pm_workflows.runtime import AgentModel
from pydantic import BaseModel, SecretStr, ValidationError

from pm.llm.prompt_encoding import TurnInput
from pm.llm.contracts import ContextMeasurement, ContextMeasurer

T = TypeVar("T", bound=BaseModel)
_MISSING = object()


class JsonOutputError(ValueError):
    """The model answer does not contain one valid JSON object."""


class ModelFormatError(JsonOutputError):
    """Repair history exhausted the input budget before a valid answer."""


def extract_json(text: str) -> dict[str, object]:
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
        return cast(dict[str, object], json.loads(text[start:end + 1]))
    except json.JSONDecodeError as error:
        raise JsonOutputError(f"The model returned invalid JSON: {error}") from None


@runtime_checkable
class MeasuredModel(Protocol):
    """A model or callable that exposes an explicit context measurement API."""

    def measure_context(self, messages: Sequence[TurnInput], *, cancel: Event | None = None,
                        **generation: object) -> ContextMeasurement:
        """Return estimated input usage and the configured input ceiling."""
        ...


def context_measure(complete: Callable[..., object]) -> ContextMeasurer | None:
    """Resolve the supported completion forms through their named interfaces."""
    if isinstance(complete, MeasuredModel):
        return complete.measure_context
    if isinstance(complete, MethodType):
        owner = complete.__self__
        if isinstance(owner, MeasuredModel):
            return owner.measure_context
    if isinstance(complete, partial):
        return context_measure(complete.func)
    return None


class _Repair[T: BaseModel]:
    """Own one uncommitted conversation and its validation failures."""

    def __init__(self, messages: Sequence[TurnInput], schema: type[T],
                 validation_context: dict[str, object] | None,
                 on_rejection: Callable[[int, str, str], None] | None,
                 validate: Callable[[dict[str, object], str], T] | None, cancel: Event | None) -> None:
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
        # Resolve model metadata before the repair boundary so schema errors propagate.
        repair_schema = schema_from_input(self.schema)
        try:
            try:
                extracted = json.dumps(extract_json(raw))
            except JsonOutputError:
                extracted = raw
            try:
                payload = json_repair.loads(extracted, schema=repair_schema)
            except SchemaDefinitionError:
                raise
            except ValueError as error:
                # The library uses ValueError for malformed values and schema rejections.
                raise JsonOutputError(str(error)) from error
            return (self.validate(cast(dict[str, object], payload), raw) if self.validate is not None else
                    self.schema.model_validate(payload, context=self.validation_context))
        except (JsonOutputError, ValidationError) as error:
            problem = str(error)
        except SchemaDefinitionError:
            raise
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

# Reset the repair history after more than this many rejected answers.
RESET_STATE_LIMIT = 8


def complete_json(complete: Callable[..., str], messages: Sequence[TurnInput], schema: type[T], *,
                  measure: ContextMeasurer | None = None, validation_context: dict[str, object] | None = None,
                  on_rejection: Callable[[int, str, str], None] | None = None,
                  validate: Callable[[dict[str, object], str], T] | None = None,
                  initial: object = _MISSING, cancel: Event | None = None, **generation: object) -> T:
    """Repair local JSON syntax, then request corrections for rejected answers.

    After nine consecutive rejections, start again with the original messages.
    Each reset retains the validator, validation context, and cancellation event.

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
    rejections_since_reset = 0
    while True:
        state.check_cancelled()
        if initial is _MISSING:
            if state.attempt and measure is None:
                raise ValueError("JSON repair requires context measurement from the selected model.")
            if measure is not None:
                measurement = measure(state.turns, cancel=cancel, **generation)
                if measurement["input_tokens"] > measurement["input_budget_tokens"]:
                    if state.attempt:
                        # Discard model repair chatter, retaining the original task and evidence.
                        state = _Repair(messages, schema, validation_context, on_rejection, validate, cancel)
                        rejections_since_reset = 0
                        continue
                    from pm.context_render.budget import RequiredContextTooLarge
                    raise RequiredContextTooLarge("The initial rendered prompt exceeds its input ceiling.")
        raw = (initial if isinstance(initial, str) else json.dumps(initial, ensure_ascii=False)) if initial is not _MISSING else complete(
            state.turns, cancel=cancel, **generation)
        initial = _MISSING
        result = state.accept(raw)
        if result is not None:
            return result
        rejections_since_reset += 1
        if rejections_since_reset > RESET_STATE_LIMIT:
            state = _Repair(messages, schema, validation_context, on_rejection, validate, cancel)
            rejections_since_reset = 0


async def complete_json_async(complete: Callable[..., Awaitable[str]], messages: Sequence[TurnInput], schema: type[T], *,
                              measure: ContextMeasurer | None = None, validation_context: dict[str, object] | None = None,
                              on_rejection: Callable[[int, str, str], None] | None = None,
                              validate: Callable[[dict[str, object], str], T] | None = None,
                              initial: object = _MISSING, cancel: Event | None = None, **generation: object) -> T:
    """Apply complete_json's repair policy to an asynchronous completion.

    Synchronous context measurement runs in a worker thread. Cancellation
    reaches measurement and completion through the supplied event.
    """
    if cancel is None and isinstance(complete, partial):
        cancel = complete.keywords.get("cancel")
    state = _Repair(messages, schema, validation_context, on_rejection, validate, cancel)
    measure = measure or context_measure(complete)
    rejections_since_reset = 0
    while True:
        state.check_cancelled()
        if initial is _MISSING:
            if state.attempt and measure is None:
                raise ValueError("JSON repair requires context measurement from the selected model.")
            if measure is not None:
                if inspect.iscoroutinefunction(measure):
                    measurement = await measure(state.turns, cancel=cancel, **generation)
                else:
                    measurement = await asyncio.to_thread(measure, state.turns, cancel=cancel, **generation)
                if measurement["input_tokens"] > measurement["input_budget_tokens"]:
                    if state.attempt:
                        # Discard model repair chatter, retaining the original task and evidence.
                        state = _Repair(messages, schema, validation_context, on_rejection, validate, cancel)
                        rejections_since_reset = 0
                        continue
                    from pm.context_render.budget import RequiredContextTooLarge
                    raise RequiredContextTooLarge("The initial rendered prompt exceeds its input ceiling.")
        raw = (initial if isinstance(initial, str) else json.dumps(initial, ensure_ascii=False)) if initial is not _MISSING else await complete(
            state.turns, cancel=cancel, **generation)
        initial = _MISSING
        result = state.accept(raw)
        if result is not None:
            return result
        rejections_since_reset += 1
        if rejections_since_reset > RESET_STATE_LIMIT:
            state = _Repair(messages, schema, validation_context, on_rejection, validate, cancel)
            rejections_since_reset = 0



def repair_agent_result(result: AgentResult, schema: type[T], model: AgentModel, *, instruction: str,
                        cancel: Event | None = None,
                        validate: Callable[[dict[str, object], str], T] | None = None) -> T:
    """Validate an agent's final answer without replay of its tool actions.

    A correction uses the same endpoint and the agent's final text. It has no
    tools. Valid answers require no model activation or extra completion.

    ``result.result_json`` is preferred as the initial answer; a session that
    ended with prose falls back to the final text. ``validate`` receives the
    parsed mapping and the raw answer, so a workflow can apply checks that the
    schema alone cannot express.
    """
    from pm.llm.activation import ModelActivation
    from pm.llm.config import Connection
    from pm.llm.stage_model import LazyStageModel
    # The session hands over a plain secret string; the connection field owns
    # the masking, so wrap it here instead of keeping a bare string around.
    connection = Connection(base_url=model.base_url, model=model.model, api_key=SecretStr(model.api_key),
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
