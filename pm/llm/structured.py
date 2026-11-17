"""Repair model JSON until its schema accepts it or its context is full.

The repair loop owns one uncommitted conversation. A malformed or schema-
rejected answer adds a correction turn that keeps the original task facts and
source ids; a rejected answer never becomes accepted state. Only syntax and
declared validation errors receive repair turns. Completion, measurement,
recording, and programming errors keep their original boundary and propagate.

A fresh conversation is a new sample, so a stochastic mistake gets unlimited
conversations. One case is not stochastic: when the same set of problems rejects
two consecutive complete conversations, the correction text
itself does not lead the model to a valid answer. That raises
:class:`RepeatedRejectionError` instead of a third conversation, because the
prompt or the validator message needs a human look.

Reading map: :func:`extract_json` reads the object, :class:`RepairConversation` owns the
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


class RepeatedRejectionError(ModelFormatError):
    """One identical problem rejected every answer of two complete repair conversations.

    The second conversation started from the original messages, so the model
    made the same mistake again and the same correction text failed again. A
    further sample cannot plausibly help. The rejected answers stay where the
    caller's ``on_rejection`` callback wrote them.
    """

    def __init__(self, problem: str, rejected_answers: int) -> None:
        self.problem: str = problem
        self.rejected_answers: int = rejected_answers
        super().__init__(
            f"The same validation problem rejected {rejected_answers} consecutive answers across a fresh "
            f"repair conversation, so the correction text does not lead to a valid answer: {problem}")


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


#: JSON Schema keywords whose value is a list of alternative branches.
_UNION_KEYWORDS = ("anyOf", "oneOf")


def _null_branches_first(node: object) -> object:
    """Copy a JSON Schema with the ``null`` branch first in every union.

    The node is foreign JSON Schema from the repair library, so its shape is
    read here and nowhere else.
    """
    if isinstance(node, list):
        items = cast(list[object], node)
        return [_null_branches_first(item) for item in items]
    if not isinstance(node, dict):
        return node
    schema = cast(dict[str, object], node)
    reordered: dict[str, object] = {}
    for keyword, value in schema.items():
        copied = _null_branches_first(value)
        if keyword in _UNION_KEYWORDS and isinstance(copied, list):
            branches = cast(list[object], copied)
            null_branches = [branch for branch in branches if branch == {"type": "null"}]
            other_branches = [branch for branch in branches if branch != {"type": "null"}]
            copied = [*null_branches, *other_branches]
        reordered[keyword] = copied
    return reordered


def _unmasked_problem(extracted: str, repair_schema: dict[str, object] | bool, problem: str) -> str:
    """Return the rejection a model can act on when an optional field hid it.

    The repair library reports the last branch of a union that matched nothing.
    Pydantic writes an optional field as ``[value, null]``, so a wrong value in
    an optional field is reported as "Expected null at <path>". A model that
    reads this drops the field instead of correcting it. This function repeats
    the check with the null branch first, so the reported branch is the value
    branch, and returns that problem.

    Only the correction text changes. The accepted payload still comes from
    the original schema, and a rejection that no null branch hid keeps its
    original text.
    """
    null_first = _null_branches_first(repair_schema)
    if null_first == repair_schema:
        return problem
    try:
        json_repair.loads(extracted, schema=null_first)
    except SchemaDefinitionError:
        raise
    except ValueError as error:
        return str(error)
    # The reordered schema accepted the answer. Keep the original rejection.
    return problem


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


class RepairConversation[T: BaseModel]:
    """Keep original evidence, latest rejected answer, and accumulated constraints.

    Rejection callbacks retain complete failed answers outside the next prompt.
    Earlier rejected answers cannot crowd original task evidence out of context.
    """

    def __init__(self, messages: Sequence[TurnInput], schema: type[T],
                 validation_context: dict[str, object] | None,
                 on_rejection: Callable[[int, str, str], None] | None,
                 validate: Callable[[dict[str, object], str], T] | None, cancel: Event | None) -> None:
        self.original_turns: list[TurnInput] = list(messages)
        self.turns = list(messages)
        self.schema = schema
        self.validation_context = validation_context
        self.on_rejection = on_rejection
        self.validate = validate
        self.cancel = cancel
        self.attempt = 0
        self.last_error = ""
        # Every rejection problem of this conversation, in answer order.
        self.problems: list[str] = []

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
                raise JsonOutputError(_unmasked_problem(extracted, repair_schema, str(error))) from error
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
        self.problems.append(problem)
        if self.on_rejection is not None:
            self.on_rejection(self.attempt, raw, problem)
        constraints = "\n".join(dict.fromkeys(self.problems))
        self.turns = [*self.original_turns, {"role": "assistant", "content": raw},
                      {"role": "user", "content":
                       f"The answer failed JSON or schema validation: {problem}\n"
                       f"Constraints reported during this repair conversation:\n{constraints}\n"
                       "Return one corrected JSON object only. Keep the original task facts and source IDs. "
                       "Do not use Markdown fences. Escape quotation marks inside strings."}]
        return None

    def check_repeated_rejections(self, previous_problem: str | None) -> str:
        """Reject consecutive repair conversations with the same failure cycle."""
        problem = "\n".join(sorted(set(self.problems)))
        if problem == previous_problem:
            raise RepeatedRejectionError(problem, rejected_answers=2 * len(self.problems))
        return problem


# Reset the repair history after more than this many rejected answers.
RESET_STATE_LIMIT = 8





def _fit_overflow(turns: Sequence[TurnInput], capacity: ContextMeasurement) -> list[TurnInput]:
    """Select annotated context against a measured complete attempt."""
    from pm.llm.prompt_encoding import normalize_turns
    from pm.llm.prompt_fitting import fit_measured_prompt
    rendered = normalize_turns(turns)
    return list(fit_measured_prompt(rendered, capacity).turns)


def complete_json(complete: Callable[..., str], messages: Sequence[TurnInput], schema: type[T], *,
                  measure: ContextMeasurer | None = None, validation_context: dict[str, object] | None = None,
                  on_rejection: Callable[[int, str, str], None] | None = None,
                  validate: Callable[[dict[str, object], str], T] | None = None,
                  initial: object = _MISSING, cancel: Event | None = None, **generation: object) -> T:
    """Repair local JSON syntax, then request corrections for rejected answers.

    After nine consecutive rejections, start again with the original messages.
    Each reset retains the validator, validation context, and cancellation event.
    Two consecutive nine-rejection conversations repeating the same set of
    problems raise RepeatedRejectionError instead of a third conversation.

    ``initial`` validates an existing agent result before any new completion.
    ``measure`` returns input_tokens and input_budget_tokens with output space
    reserved. It defaults to the completion model's measure_context method.
    Missing measurement raises ValueError when repair is necessary. Cancellation
    propagates. No tool actions or accepted state changes occur here.
    """
    if cancel is None and isinstance(complete, partial):
        cancel = complete.keywords.get("cancel")
    state = RepairConversation(messages, schema, validation_context, on_rejection, validate, cancel)
    measure = measure or context_measure(complete)
    rejections_since_reset = 0
    previous_problem: str | None = None
    while True:
        state.check_cancelled()
        if initial is _MISSING:
            if state.attempt and measure is None:
                raise ValueError("JSON repair requires context measurement from the selected model.")
            if measure is not None:
                from pm.context_render.budget import RequiredContextTooLarge
                try:
                    measurement = measure(state.turns, cancel=cancel, **generation)
                    if measurement["input_tokens"] > measurement["input_budget_tokens"]:
                        state.turns = _fit_overflow(state.turns, measurement)
                except RequiredContextTooLarge:
                    if not state.attempt:
                        raise
                    # Required repair chatter can fill the window. Start a fresh
                    # sample with the original task, retaining all rejection receipts.
                    state = RepairConversation(messages, schema, validation_context, on_rejection, validate, cancel)
                    rejections_since_reset = 0
                    previous_problem = None
                    continue
        raw = (initial if isinstance(initial, str) else json.dumps(initial, ensure_ascii=False)) if initial is not _MISSING else complete(
            state.turns, cancel=cancel, **generation)
        initial = _MISSING
        result = state.accept(raw)
        if result is not None:
            return result
        rejections_since_reset += 1
        if rejections_since_reset > RESET_STATE_LIMIT:
            previous_problem = state.check_repeated_rejections(previous_problem)
            state = RepairConversation(messages, schema, validation_context, on_rejection, validate, cancel)
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
    state = RepairConversation(messages, schema, validation_context, on_rejection, validate, cancel)
    measure = measure or context_measure(complete)
    rejections_since_reset = 0
    previous_problem: str | None = None
    while True:
        state.check_cancelled()
        if initial is _MISSING:
            if state.attempt and measure is None:
                raise ValueError("JSON repair requires context measurement from the selected model.")
            if measure is not None:
                from pm.context_render.budget import RequiredContextTooLarge
                try:
                    if inspect.iscoroutinefunction(measure):
                        measurement = await measure(state.turns, cancel=cancel, **generation)
                    else:
                        measurement = await asyncio.to_thread(measure, state.turns, cancel=cancel, **generation)
                    if measurement["input_tokens"] > measurement["input_budget_tokens"]:
                        state.turns = await asyncio.to_thread(_fit_overflow, state.turns, measurement)
                except RequiredContextTooLarge:
                    if not state.attempt:
                        raise
                    state = RepairConversation(messages, schema, validation_context, on_rejection, validate, cancel)
                    rejections_since_reset = 0
                    previous_problem = None
                    continue
        raw = (initial if isinstance(initial, str) else json.dumps(initial, ensure_ascii=False)) if initial is not _MISSING else await complete(
            state.turns, cancel=cancel, **generation)
        initial = _MISSING
        result = state.accept(raw)
        if result is not None:
            return result
        rejections_since_reset += 1
        if rejections_since_reset > RESET_STATE_LIMIT:
            previous_problem = state.check_repeated_rejections(previous_problem)
            state = RepairConversation(messages, schema, validation_context, on_rejection, validate, cancel)
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
