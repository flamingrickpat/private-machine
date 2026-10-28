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
from typing import cast
from threading import Event
from pm_workflows.python_role import RoleContext
from pm_workflows.working_target import WorkingEnvironment
from pm.llm.contracts import ContextMeasurement, ContextMeasurer, ModelTool, WorkflowCompletion

from pm.utils import yaml_util
from pydantic import BaseModel, Field, TypeAdapter
from collections.abc import Callable, Sequence
from pm.llm.prompt_encoding import TurnInput, normalize_turns
from pm.context_render.turns import Turn

from pm.llm.structured import (
    JsonOutputError as StageOutputError, ModelFormatError,
    complete_json as validated_json, extract_json, context_measure,
)
# The shard labels belong to the shard layout, which lives in pm.context_render.story.
from pm.context_render.story import ShardLabels


def load_yaml(path: str | Path) -> dict[str, object]:
    return TypeAdapter(dict[str, object]).validate_python(yaml_util.safe_load(Path(path).read_text(encoding="utf-8")))


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


def read_request(environment: WorkingEnvironment) -> dict[str, object]:
    return TypeAdapter(dict[str, object]).validate_python(yaml_util.safe_load(environment.read_text("inbox/request.yaml")))


def complete_json[T: BaseModel](context: RoleContext, messages: Sequence[TurnInput], schema: type[T], *,
                  transform: Callable[[list[Turn]], list[Turn]] | None = None,
                  validation_context: dict[str, object] | None = None,
                  validate: Callable[[dict[str, object], str], T] | None = None,
                  cancel: Event | None = None, **generation: object) -> T:
    """Repair model output with the shared runtime policy. Keep each rejection.

    The transform runs before completion and measurement. It must retain all
    task evidence. Context exhaustion raises ModelFormatError before a commit.
    """
    assert context.runtime is not None
    runtime = context.runtime
    tool = cast(WorkflowCompletion, runtime.tools[ModelTool.Complete])
    selected_cancel = runtime.stop_event if cancel is None else cancel

    def transformed(turns: Sequence[TurnInput]) -> list[Turn]:
        turns = normalize_turns(turns)
        return transform(turns) if transform is not None else turns

    def complete(turns: Sequence[TurnInput], *, cancel: Event | None = None, **settings: object) -> str:
        return tool(transformed(turns), **settings)

    original_measure = (cast(ContextMeasurer, runtime.tools[ModelTool.MeasureContext])
                        if ModelTool.MeasureContext in runtime.tools else context_measure(tool))

    def measure(turns: Sequence[TurnInput], *, cancel: Event | None = None, **settings: object) -> ContextMeasurement:
        if original_measure is None:
            raise ValueError("The model tool must provide context measurement before inference.")
        return original_measure(transformed(turns), cancel=cancel, **settings)

    def rejected(attempt: int, raw: str, problem: str) -> None:
        runtime.environment.write_text(
            f"scratch/model-output-rejection-{attempt:02}.json",
            json.dumps({"attempt": attempt, "schema": schema.__name__, "raw": raw,
                        "error": problem}, ensure_ascii=False))

    return validated_json(complete, messages, schema, measure=measure, cancel=selected_cancel,
                          validation_context=validation_context, on_rejection=rejected, validate=validate, **generation)


def complete_text(context: RoleContext, messages: Sequence[TurnInput], *, attempts: int, **generation: object) -> str:
    """Return the stripped prose answer of one model call.

    Prose has no schema, so the only invalid output is an empty answer. An
    empty answer is retried with the same turns, up to ``attempts`` calls.
    Each empty answer leaves a record in the role's scratch directory.
    Exhaustion raises ModelFormatError. Cancellation, transport failures, and
    tool errors propagate unchanged.
    """
    assert context.runtime is not None
    turns = normalize_turns(messages)
    tool = cast(WorkflowCompletion, context.runtime.tools[ModelTool.Complete])
    for attempt in range(1, attempts + 1):
        raw = tool(turns, **generation)
        if raw.strip():
            return raw.strip()
        context.runtime.environment.write_text(
            f"scratch/model-output-empty-{attempt:02}.json",
            json.dumps({"attempt": attempt, "raw": raw, "error": "empty prose answer"}, ensure_ascii=False),
        )
    raise ModelFormatError(f"The model returned empty prose {attempts} times.")


def write_report(environment: WorkingEnvironment, report: BaseModel) -> None:
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
