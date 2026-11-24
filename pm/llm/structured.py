"""Repair model JSON until its schema accepts it or its context is full.

The repair loop owns one uncommitted conversation. A malformed or schema-
rejected answer adds a correction turn that keeps the original task facts and
source ids; a rejected answer never becomes accepted state. Only syntax and
declared validation errors receive repair turns. Completion, measurement,
recording, and programming errors keep their original boundary and propagate.

Each repair starts with the same fitted evidence and the latest bounded
diagnostics. A schema rejection can resample until cancellation. Repairing a
completed agent report never runs the agent's tools again.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
import json
import re
from threading import Event
from typing import TypeVar
from dataclasses import dataclass

from json_repair import json_repair
from json_repair.schema_repair import SchemaDefinitionError
from pm_workflows.protocol import AgentResult
from pm_workflows.runtime import AgentModel
from pydantic import BaseModel, SecretStr, ValidationError, JsonValue, TypeAdapter

from pm.llm.prompt_encoding import TurnInput
from pm.llm.session import ModelSession
from pm.llm.workflow_sessions import ConfiguredAgentModel
from pm.llm.generation import GenerationSettings
from pm.context_render.prompt import require_rendered_prompt, RendererCapture, RendererInput
from pm.context_render.correction import (
    AgentReportCorrectionInput, CorrectionHistory, JsonCorrectionInput, render_agent_report_correction,
    render_json_correction,
)
from pm.llm.prompt_encoding import normalize_turns

T = TypeVar("T", bound=BaseModel)
@dataclass(frozen=True, slots=True)
class InitialResponse:
    raw_text: str



class InternalResponseSchemaError(RuntimeError):
    pass


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


def extract_json(text: str) -> dict[str, JsonValue]:
    """Read an object, including an object inside a Markdown fence.

    Raise JsonOutputError for missing or malformed JSON. A non-text result
    raises TypeError because it violates the completion interface.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise JsonOutputError("The model returned no JSON object.")
    try:
        return TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_json(text[start:end + 1])
    except ValidationError as error:
        raise JsonOutputError(f"The model returned invalid JSON: {error}") from None


#: JSON Schema keywords whose value is a list of alternative branches.
_UNION_KEYWORDS = ("anyOf", "oneOf")


def _null_branches_first(node: JsonValue) -> JsonValue:
    if isinstance(node, list):
        return [_null_branches_first(item) for item in node]
    if isinstance(node, dict):
        return _reorder_schema(node)
    return node


def _reorder_schema(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    reordered: dict[str, JsonValue] = {}
    for keyword, value in schema.items():
        copied = _null_branches_first(value)
        if keyword in _UNION_KEYWORDS and isinstance(copied, list):
            null_branches = [branch for branch in copied if branch == {"type": "null"}]
            other_branches = [branch for branch in copied if branch != {"type": "null"}]
            copied = [*null_branches, *other_branches]
        reordered[keyword] = copied
    return reordered


def _unmasked_problem(extracted: str, repair_schema: dict[str, JsonValue] | bool, problem: str) -> str:
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
    null_first = repair_schema if isinstance(repair_schema, bool) else _reorder_schema(repair_schema)
    if null_first == repair_schema:
        return problem
    try:
        json_repair.loads(extracted, schema=null_first)
    except SchemaDefinitionError as error:
        raise InternalResponseSchemaError("The internal response schema is invalid.") from error
    except ValueError as error:
        return str(error)
    # The reordered schema accepted the answer. Keep the original rejection.
    return problem


#: The repair library's enum rejection, which names the path but nothing else.
_ENUM_PROBLEM = re.compile(r"Value at (?P<path>\S+) does not match enum")


def _path_parts(path: str) -> list[str | int]:
    """Split a reported path such as ``$.facts[21].record_type`` into its parts."""
    parts: list[str | int] = []
    for segment in path.split("."):
        if segment in ("", "$"):
            continue
        name = segment.split("[", 1)[0]
        if name:
            parts.append(name)
        for index in re.findall(r"\[(\d+)\]", segment):
            parts.append(int(index))
    return parts


def _follow_reference(schema: dict[str, JsonValue], node: dict[str, JsonValue]) -> dict[str, JsonValue] | None:
    """Return the definition a ``$ref`` such as ``#/$defs/FactFiling`` points at."""
    reference = node.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/"):
        return node
    resolved: dict[str, JsonValue] | None = schema
    for part in reference[2:].split("/"):
        if resolved is None:
            return None
        child = resolved.get(part)
        resolved = child if isinstance(child, dict) else None
    return resolved


def _schema_node(schema: dict[str, JsonValue], path: str) -> dict[str, JsonValue] | None:
    """Return the JSON Schema node a reported path names, following ``$ref`` links.

    Array indices are ignored while walking the schema because one ``items`` node
    describes every element. The schema is foreign wire format, which is why this
    module, and not the domain layer, walks it as a mapping.
    """
    node: dict[str, JsonValue] | None = schema
    for part in _path_parts(path):
        if node is None:
            return None
        node = _follow_reference(schema, node)
        if node is None:
            return None
        if isinstance(part, int):
            items = node.get("items")
            node = items if isinstance(items, dict) else None
            continue
        properties = node.get("properties")
        child = properties.get(part) if isinstance(properties, dict) else None
        node = child if isinstance(child, dict) else None
    if node is None:
        return None
    return _follow_reference(schema, node)


def _answer_value(answer: JsonValue, path: str) -> JsonValue:
    """Return the value a reported path names in the decoded model answer."""
    current: JsonValue = answer
    for part in _path_parts(path):
        if isinstance(part, int):
            if not isinstance(current, list) or part >= len(current):
                return None
            current = current[part]
            continue
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _enum_detail(extracted: str, schema: dict[str, JsonValue] | bool, problem: str) -> str:
    """Name the rejected value and the values its schema allows.

    The repair library reports an enum violation as ``Value at $.facts[21].record_type
    does not match enum.`` A model that cannot see what it wrote, or what the schema
    accepts, sends the same value back. Measured on 2026-11-22: one fact
    classification was repaired 164 times in 108 minutes with that text alone. This
    appends one line per enum violation the schema can describe. A path the schema
    does not describe keeps the original text, because a guessed allowed set would
    be a guess about our own schema.
    """
    matches = list(_ENUM_PROBLEM.finditer(problem))
    if not matches or isinstance(schema, bool):
        return problem
    try:
        answer = json.loads(extracted)
    except ValueError:
        return problem
    details: list[str] = []
    for match in matches:
        path = match.group("path")
        node = _schema_node(schema, path)
        allowed = node.get("enum") if node is not None else None
        if not isinstance(allowed, list):
            continue
        chosen = _answer_value(answer, path)
        allowed_names = ", ".join(str(option) for option in allowed)
        details.append(f"{path} is {chosen!r}, which the schema does not allow. Use one of {allowed_names}.")
    if not details:
        return problem
    return problem + "\n" + "\n".join(details)


def parse_response(raw: str, schema: type[T], *, validation_context: BaseModel | None = None) -> T:
    try:
        repair_schema = TypeAdapter[dict[str, JsonValue] | bool](dict[str, JsonValue] | bool).validate_python(schema.model_json_schema())
    except ValidationError as error:
        raise InternalResponseSchemaError(f"Invalid internal response schema: {schema.__name__}.") from error
    try:
        extracted = json.dumps(extract_json(raw))
    except JsonOutputError:
        extracted = raw
    try:
        payload = json_repair.loads(extracted, schema=repair_schema)
    except SchemaDefinitionError as error:
        raise InternalResponseSchemaError("The internal response schema is invalid.") from error
    except ValueError as error:
        problem = _unmasked_problem(extracted, repair_schema, str(error))
        raise JsonOutputError(_enum_detail(extracted, repair_schema, problem)) from error
    return schema.model_validate(payload, context=validation_context)


def complete_json(session: ModelSession, messages: Sequence[TurnInput], schema: type[T], *,
                  validation_context: BaseModel | None = None,
                  initial: InitialResponse | None = None, initial_problem: str = "", cancel: Event | None = None,
                  settings: GenerationSettings = GenerationSettings()) -> T:
    original = tuple(normalize_turns(messages))
    render_settings = session.renderer_settings_for(settings)
    prompt = require_rendered_prompt(schema.__name__, original, render_settings)
    corrections = CorrectionHistory()
    if initial_problem:
        correction = JsonCorrectionInput(schema.__name__, original, "", initial_problem)
        prompt = render_json_correction(correction, render_settings)
    while True:
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        raw = initial.raw_text if initial is not None else session.complete(prompt.turns, settings=settings)
        initial = None
        try:
            return parse_response(raw, schema, validation_context=validation_context)
        except (JsonOutputError, ValidationError) as error:
            diagnostic = corrections.record_rejection(schema.__name__, str(error))
            correction = JsonCorrectionInput(schema.__name__, original, raw, diagnostic)
            with RendererCapture(session.capture_settings("json correction"),
                                 RendererInput("json correction", correction, render_settings)) as capture:
                prompt = render_json_correction(correction, render_settings)
                capture.save_output(prompt)


async def complete_json_async(session: ModelSession, messages: Sequence[TurnInput], schema: type[T], *,
                              validation_context: BaseModel | None = None,
                              initial: InitialResponse | None = None, cancel: Event | None = None,
                  settings: GenerationSettings = GenerationSettings()) -> T:
    original = tuple(normalize_turns(messages))
    render_settings = session.resolved_renderer_settings_for(settings)
    prompt = require_rendered_prompt(schema.__name__, original, render_settings)
    corrections = CorrectionHistory()
    while True:
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        raw = initial.raw_text if initial is not None else await session.complete_async(prompt.turns, settings=settings)
        initial = None
        try:
            return parse_response(raw, schema, validation_context=validation_context)
        except (JsonOutputError, ValidationError) as error:
            diagnostic = corrections.record_rejection(schema.__name__, str(error))
            correction = JsonCorrectionInput(schema.__name__, original, raw, diagnostic)
            with RendererCapture(session.capture_settings("json correction"),
                                 RendererInput("json correction", correction, render_settings)) as capture:
                prompt = render_json_correction(correction, render_settings)
                capture.save_output(prompt)


def repair_agent_result(result: AgentResult, schema: type[T], model: AgentModel, *, instruction: str,
                        cancel: Event | None = None) -> T:
    """Validate an agent's final answer without replay of its tool actions.

    A correction uses the same endpoint and the agent's final text. It has no
    tools. Valid answers require no model activation or extra completion.

    ``result.result_json`` is preferred as the initial answer; a session that
    ended with prose falls back to the final text. ``validate`` receives the
    parsed mapping and the raw answer, so a workflow can apply checks that the
    schema alone cannot express.
    """
    if cancel is not None and cancel.is_set():
        raise asyncio.CancelledError
    raw = json.dumps(result.result_json, ensure_ascii=False) if result.result_json is not None else result.stdout
    try:
        return parse_response(raw, schema)
    except (JsonOutputError, ValidationError) as error:
        initial_problem = str(error)
    from pm.llm.activation import ModelActivation
    from pm.llm.config import Connection
    from pm.llm.stage_model import LazyStageModel
    # The session hands over a plain secret string; the connection field owns
    # the masking, so wrap it here instead of keeping a bare string around.
    if isinstance(model, ConfiguredAgentModel):
        connection = model.connection
        activation = model.activation
    else:
        # Standalone native callers supply their endpoint directly, without a PM preset.
        connection = Connection(base_url=model.base_url, model=model.model, api_key=SecretStr(model.api_key),
                                context_window=model.context_window)
        activation = ModelActivation()
    stage = LazyStageModel(connection, activation, live_test=model.live_test)
    session = ModelSession(stage, cancel)
    generation = GenerationSettings(enable_thinking=model.enable_thinking)
    render_settings = session.renderer_settings_for(generation)
    input = AgentReportCorrectionInput(schema.__name__, instruction,
        json.dumps(schema.model_json_schema(), ensure_ascii=False), raw, initial_problem)
    with RendererCapture(session.capture_settings("agent report correction"),
                         RendererInput("agent report correction", input, render_settings)) as capture:
        prompt = render_agent_report_correction(input, render_settings)
        capture.save_output(prompt)
    return complete_json(session, prompt.turns, schema, initial=InitialResponse(raw), cancel=cancel,
                         settings=generation)
