"""Shared helpers for the eight cognitive stage roles.

Each stage keeps its own prompts, schema, and finalizer. This module holds only
the mechanical plumbing that is identical across stages: read the request, call
the injected ``llm_complete`` tool, extract and validate the JSON object, and
write ``outbox/report.yaml``. It contains no stage prompts and no cognitive
policy.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError
from collections.abc import Sequence
from pm.llm.prompt_encoding import TurnInput, normalize_turns
from pm.context_render.turns import Role, Turn

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class StageOutputError(RuntimeError):
    """The model did not return a valid JSON object for this stage."""


class ModelFormatError(StageOutputError):
    """Repeated model output failed syntax or schema validation before any commit."""


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_request(environment) -> dict:
    return yaml.safe_load(environment.read_text("inbox/request.yaml"))


def extract_json(text: str) -> dict:
    match = _JSON_OBJECT.search(text or "")
    if match is None:
        raise StageOutputError("The stage model returned no JSON object.")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise StageOutputError(f"The stage model returned invalid JSON: {error}") from None
    if not isinstance(payload, dict):
        raise StageOutputError("The stage model returned JSON that is not an object.")
    return payload


def complete_json(context, messages: Sequence[TurnInput], schema: type[BaseModel], **generation: object):
    """Validate model JSON with at most ten syntax or schema attempts.

    Each repair keeps the original task and only the latest rejected answer.
    Rejected answers and errors remain in the stage's scratch directory.
    Exhaustion raises ModelFormatError without a report or a state change.
    Cancellation, transport failures, and tool errors propagate unchanged.
    """
    original = normalize_turns(messages)
    turns = original
    last_error = None
    for attempt in range(1, 11):
        tool = context.runtime.tools["llm_complete"]
        raw = tool(turns, **generation)
        if not isinstance(raw, str):
            raise TypeError("llm_complete must return the completion text as a string.")
        try:
            return schema.model_validate(extract_json(raw))
        except (StageOutputError, ValidationError) as error:
            last_error = error
            context.runtime.environment.write_text(
                f"scratch/model-output-rejection-{attempt:02}.json",
                json.dumps({"attempt": attempt, "schema": schema.__name__, "raw": raw,
                            "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False),
            )
            # These roles propose data only. Repair repeats no accepted action.
            # Keep the full task without accumulating ten rejected answers.
            turns = [*original, Turn(Role.Assistant, raw), Turn(Role.User,
                f"The answer failed validation: {error}\n"
                "Return one corrected JSON object only. Preserve the task facts and source IDs. "
                "Use short prose fields. Escape quotation marks inside strings. Do not use Markdown fences.")]
    raise ModelFormatError(f"The stage model output failed validation after ten attempts: {last_error}") from None


def write_report(environment, report: BaseModel) -> None:
    environment.write_text(
        "scratch/report.yaml",
        yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
    )
    environment.move("scratch/report.yaml", "outbox/report.yaml")


__all__ = [
    "StageOutputError",
    "ModelFormatError",
    "complete_json",
    "extract_json",
    "load_yaml",
    "read_request",
    "write_report",
]
