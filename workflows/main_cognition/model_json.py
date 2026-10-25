"""Shared model-output plumbing for Python roles that call the model.

The main cognition phases and the inner-thought stages use these helpers.
They read the request, call the injected ``llm_complete`` tool, extract and
validate the JSON object, and write ``outbox/report.yaml``. The module holds
no prompts and no cognitive policy.

Only invalid model output gets a repair turn here. Transport errors come from
the model tool and propagate unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

from pm.utils import yaml_util
from pydantic import BaseModel, Field
from collections.abc import Callable, Sequence
from pm.llm.prompt_encoding import TurnInput, normalize_turns
from pm.context_render.turns import Role, Turn

from pm.llm.structured import (
    JsonOutputError as StageOutputError, ModelFormatError,
    complete_json as validated_json, extract_json, context_measure,
)
# The shard labels belong to the shard layout, which lives in pm.context_render.story.
from pm.context_render.story import ShardLabels


def load_yaml(path: str | Path) -> dict:
    return yaml_util.safe_load(Path(path).read_text(encoding="utf-8"))


class PhasePrompts(BaseModel):
    """The prompt texts of one phase, read from its YAML file.

    Every phase has ``system_prompt`` and ``user_template``. The other fields
    exist only in the files of the phases that use them.
    """

    system_prompt: str = ""
    user_template: str = ""
    repair_template: str = ""
    job_template: str = ""
    reality_labels: dict[str, str] = Field(default_factory=dict)
    reality_rules: dict[str, str] = Field(default_factory=dict)
    # The shared story prefix (story.yaml) and the per-call parts of the codelets phase.
    story_template: str = ""
    story_ack: str = ""
    shard_template: str = ""
    shard_labels: ShardLabels = Field(default_factory=ShardLabels)
    # The attention phase: the mental aperture texts of the old code.
    aperture_narrow: str = ""
    aperture_broad: str = ""
    aperture_default: str = ""
    view_template: str = ""
    conversion_template: str = ""
    selector_template: str = ""
    echo_template: str = ""
    # The appraise phase: the expectation check. The route phase: the anticipation call.
    check_template: str = ""
    anticipation_template: str = ""
    # The goals and decide phases: the regulation mode in story words, by mode.
    regulation_words: dict[str, str] = Field(default_factory=dict)
    # The decide and route phases: a stimulus group in story words, by group kind.
    part_entity: str = ""
    part_workflow: str = ""
    part_self: str = ""
    # The route phase: the routed action in words, by action type.
    action_words: dict[str, str] = Field(default_factory=dict)


def load_prompts(path: str | Path) -> PhasePrompts:
    """Read and validate one prompt file. A missing or invalid file raises."""
    return PhasePrompts.model_validate(load_yaml(path))


def read_request(environment) -> dict:
    return yaml_util.safe_load(environment.read_text("inbox/request.yaml"))


def complete_json(context, messages: Sequence[TurnInput], schema: type[BaseModel], *,
                  transform: Callable[[list[Turn]], list[Turn]] | None = None,
                  validation_context: dict | None = None, **generation: object):
    """Repair model output with the shared runtime policy. Keep each rejection.

    The transform runs before completion and measurement. It must retain all
    task evidence. Context exhaustion raises ModelFormatError before a commit.
    """
    tool = context.runtime.tools["llm_complete"]
    cancel = generation.pop("cancel", context.runtime.stop_event if hasattr(context.runtime, "stop_event") else None)

    def transformed(turns):
        turns = normalize_turns(turns)
        return transform(turns) if transform is not None else turns

    def complete(turns, **settings):
        settings.pop("cancel", None)
        return tool(transformed(turns), **settings)

    original_measure = context.runtime.tools.get("llm_measure_context") or context_measure(tool)

    def measure(turns, **settings):
        if original_measure is None:
            raise ValueError("The model tool must provide context measurement for JSON repair.")
        return original_measure(transformed(turns), **settings)

    def rejected(attempt, raw, problem):
        context.runtime.environment.write_text(
            f"scratch/model-output-rejection-{attempt:02}.json",
            json.dumps({"attempt": attempt, "schema": schema.__name__, "raw": raw,
                        "error": problem}, ensure_ascii=False))

    return validated_json(complete, messages, schema, measure=measure, cancel=cancel,
                          validation_context=validation_context, on_rejection=rejected, **generation)


def complete_text(context, messages: Sequence[TurnInput], *, attempts: int, **generation: object) -> str:
    """Return the stripped prose answer of one model call.

    Prose has no schema, so the only invalid output is an empty answer. An
    empty answer is retried with the same turns, up to ``attempts`` calls.
    Each empty answer leaves a record in the role's scratch directory.
    Exhaustion raises ModelFormatError. Cancellation, transport failures, and
    tool errors propagate unchanged.
    """
    turns = normalize_turns(messages)
    for attempt in range(1, attempts + 1):
        raw = context.runtime.tools["llm_complete"](turns, **generation)
        if not isinstance(raw, str):
            raise TypeError("llm_complete must return the completion text as a string.")
        if raw.strip():
            return raw.strip()
        context.runtime.environment.write_text(
            f"scratch/model-output-empty-{attempt:02}.json",
            json.dumps({"attempt": attempt, "raw": raw, "error": "empty prose answer"}, ensure_ascii=False),
        )
    raise ModelFormatError(f"The model returned empty prose {attempts} times.")


def write_report(environment, report: BaseModel) -> None:
    environment.write_text(
        "scratch/report.yaml",
        yaml_util.safe_dump(report.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
    )
    environment.move("scratch/report.yaml", "outbox/report.yaml")


__all__ = [
    "StageOutputError",
    "ModelFormatError",
    "complete_json",
    "complete_text",
    "extract_json",
    "PhasePrompts",
    "ShardLabels",
    "load_prompts",
    "load_yaml",
    "read_request",
    "write_report",
]
