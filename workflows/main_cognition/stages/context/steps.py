"""Context role.

Entry point: ``run(context)``. Builds the context messages from the request
and its triage, appraisal, and memory-candidate payloads, validates the model
JSON into :class:`ContextResult`, checks that every selected candidate id came
from the supplied candidates, and writes ``outbox/report.yaml``. An invalid ID
gets one repair turn. A failed repair retains only supplied memory sources and
records the dropped IDs. Current conversation can then continue without false
memory references. The message builder and selection check are pure.
"""
from __future__ import annotations

import json
from pathlib import Path
from pm.context_render import render_context
from pm.context_render.turns import Role, Turn, format_content

from workflows.main_cognition.schemas import ContextResult
from workflows.main_cognition.stage_support import (
    StageOutputError, complete_json, load_yaml, read_request, write_report,
)

WORKFLOW_DIR = Path(__file__).resolve().parent


def load_prompts(path: Path | None = None) -> dict:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml")


def _field(item, name, default=None):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def render_memories(candidates: list) -> str:
    lines = []
    for candidate in candidates:
        lines.append(
            f"- knoxel_id={_field(candidate, 'knoxel_id')} "
            f"kind={_field(candidate, 'memory_kind', 'attributed_record')} "
            f"evidence_role={_field(candidate, 'evidence_role', 'background')} "
            f"scope={_field(candidate, 'scope', {})} "
            f"source_ids={_field(candidate, 'source_ids', [])} "
            f"historical={_field(candidate, 'historical', False)} "
            f"text={_field(candidate, 'text', '')!r}"
        )
    return "\n".join(lines) or "(none)"


def build_messages(request: dict, prompts: dict) -> list[Turn]:
    system = prompts["system_prompt"].format(
        companion_name=request.get("companion_name", "Spamuel"),
    )
    user = format_content(prompts["user_template"],
        companion_name=request.get("companion_name", "Spamuel"),
        user_name=request.get("user_name", "Jesse"),
        state_summary=request.get("state_summary") or "(no recorded state update)",
        triage_summary=render_context(request.get("triage") or {}),
        appraisal_summary=render_context(request.get("appraisal") or {}),
        active_goals="\n".join(f"- {goal}" for goal in request.get("active_goals", []))
        or "(none)",
        capabilities="\n".join(f"- {capability}" for capability in request.get("capabilities", []))
        or "(none)",
        memory_candidates=render_memories(request.get("memory_candidates", [])),
        stimuli=render_context(request.get("stimuli", [])),
    )
    return [
        Turn(Role.System, system.strip()),
        Turn(Role.User, user.strip()),
    ]


def validate_selection(result: ContextResult, candidates: list) -> None:
    """Raise if the model selected an id outside the supplied candidate set."""
    allowed = {_field(candidate, "knoxel_id") for candidate in candidates}
    for candidate_id in result.candidate_ids:
        if candidate_id not in allowed:
            raise StageOutputError(
                f"Context selected knoxel_id {candidate_id} that is not a "
                "supplied memory candidate."
            )


def retain_valid_selection(result: ContextResult, candidates: list) -> tuple[ContextResult, list[int]]:
    """Drop unavailable memory IDs after a failed repair, retaining real sources.

    Rebuild memory text from the supplied snapshot. This avoids promoting an
    unsupported model summary when its reference cannot be resolved. Current
    stimuli remain in the cycle, so failed recall does not block conversation.
    Return the repaired result and dropped IDs for the diagnostic receipt.
    """
    sources = {_field(item, "knoxel_id"): _field(item, "text", "") for item in candidates}
    retained = list(dict.fromkeys(key for key in result.candidate_ids if key in sources))
    dropped = [key for key in result.candidate_ids if key not in sources]
    return result.model_copy(update={"candidate_ids": retained,
        "memories": [str(sources[key]) for key in retained]}), dropped


def run(context) -> dict:
    request = read_request(context.runtime.environment)
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    result = complete_json(context, messages, ContextResult, temperature=0.1, enable_thinking=False)
    candidates = request.get("memory_candidates", [])
    try:
        validate_selection(result, candidates)
    except StageOutputError as error:
        # The public domain context also contains IDs. They are not a grant
        # for memory selection. Retain the invalid response and repair once.
        context.runtime.environment.write_text("outbox/selection-rejection.json", json.dumps({
            "error": str(error), "rejected_result": result.model_dump(mode="json"),
            "allowed_ids": [_field(item, "knoxel_id") for item in candidates]}, indent=2))
        recovery = [*messages, {"role": "assistant", "content": result.model_dump_json()},
                    {"role": "user", "content": str(error) + " Select only from these candidate IDs: "
                     + str([_field(item, "knoxel_id") for item in candidates])
                     + ". IDs in other context fields are not granted. Return an empty selection if none fits."}]
        result = complete_json(context, recovery, ContextResult, temperature=0.1,
                               enable_thinking=False)
        try:
            validate_selection(result, candidates)
        except StageOutputError as repair_error:
            rejected = result.model_dump(mode="json")
            result, dropped = retain_valid_selection(result, candidates)
            context.runtime.environment.write_text("outbox/selection-recovery.json", json.dumps({
                "error": str(repair_error), "rejected_result": rejected,
                "dropped_ids": dropped, "retained_ids": result.candidate_ids,
                "outcome": "continued_with_supplied_memories"}, indent=2))
    write_report(context.runtime.environment, result)
    return {
        "status": "done",
        "summary": f"selected {len(result.candidate_ids)} memories and "
                   f"{len(result.active_goals)} goals",
    }


__all__ = [
    "build_messages",
    "load_prompts",
    "render_memories",
    "run",
    "retain_valid_selection",
    "validate_selection",
]
