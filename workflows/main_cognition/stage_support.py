"""Shared helpers for the eight cognitive stage roles.

Each stage keeps its own prompts, schema, and finalizer. This module holds only
the mechanical plumbing that is identical across stages: read the request, call
the injected ``llm_complete`` tool, extract and validate the JSON object, and
write ``outbox/report.yaml``. It contains no stage prompts and no cognitive
policy.
"""
from __future__ import annotations

import asyncio
import copy
import json
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class StageOutputError(RuntimeError):
    """The model did not return a valid JSON object for this stage."""


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


def complete_json(context, messages: list[dict], schema: type[BaseModel], **generation):
    """Call the injected model tool and validate its JSON into ``schema``. Tries it 10 times."""
    temp_list = copy.deepcopy(messages)
    error = None
    for _ in range(10):
        tool = context.runtime.tools["llm_complete"]
        raw = tool(temp_list, **generation)
        if not isinstance(raw, str):
            raise StageOutputError("llm_complete must return the completion text as a string.")
        payload = extract_json(raw)
        try:
            return schema.model_validate(payload)
        except ValidationError as error:
            temp_list.append({"role": "assistant", "content": raw})
            temp_list.append({"role": "user", "content": f"This is invalid JSON, '{error}'. "
                                                         f"Try again and and ONLY output the JSON string."})
    raise StageOutputError(f"The stage model output failed validation: {error}") from None


def write_report(environment, report: BaseModel) -> None:
    environment.write_text(
        "scratch/report.yaml",
        yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
    )
    environment.move("scratch/report.yaml", "outbox/report.yaml")


__all__ = [
    "StageOutputError",
    "complete_json",
    "extract_json",
    "load_yaml",
    "read_request",
    "write_report",
]
