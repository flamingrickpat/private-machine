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
import re
from pathlib import Path

from pm.utils import yaml_util
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from collections.abc import Callable, Sequence
from pm.llm.prompt_encoding import TurnInput, normalize_turns
from pm.context_render.turns import Role, Turn

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class StageOutputError(RuntimeError):
    """The model did not return a valid JSON object for this stage."""


class ModelFormatError(StageOutputError):
    """Repeated model output failed syntax or schema validation before any commit."""


def load_yaml(path: str | Path) -> dict:
    return yaml_util.safe_load(Path(path).read_text(encoding="utf-8"))


class ShardLabels(BaseModel):
    """The words that the shared story prefix uses for the workspace shards.

    The texts can hold ``{companion_name}``, ``{entity_name}``, and
    ``{workflow_name}``.
    """

    model_config = ConfigDict(extra="forbid")

    self_title: str = ""
    self_tail: str = ""
    entity_title: str = ""
    entity_tail: str = ""
    workflow_title: str = ""
    workflow_tail: str = ""
    empty_items: str = ""
    empty_tail: str = ""
    # A public reply shows the verbatim lines of a tail in its transcript.
    verbatim_tail: str = ""
    # The attention phase: what the awareness of each shard kind is about.
    self_focus: str = ""
    entity_focus: str = ""
    workflow_focus: str = ""


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


def complete_json(context, messages: Sequence[TurnInput], schema: type[BaseModel], *,
                  transform: Callable[[list[Turn]], list[Turn]] | None = None,
                  validation_context: dict | None = None, **generation: object):
    """Validate model JSON with at most ten syntax or schema attempts.

    Each repair keeps the original task and only the latest rejected answer.
    ``transform`` changes the turns of every attempt just before the model
    call, for example to insert private grounding context.
    ``validation_context`` is the pydantic validation context of the schema,
    for example the workflow names that a route answer may use.
    Rejected answers and errors remain in the role's scratch directory.
    Exhaustion raises ModelFormatError without a report or a state change.
    Cancellation, transport failures, and tool errors propagate unchanged.
    """
    original = normalize_turns(messages)
    turns = original
    last_error = None
    for attempt in range(1, 11):
        tool = context.runtime.tools["llm_complete"]
        raw = tool(transform(list(turns)) if transform is not None else turns, **generation)
        if not isinstance(raw, str):
            raise TypeError("llm_complete must return the completion text as a string.")
        try:
            return schema.model_validate(extract_json(raw), context=validation_context)
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
