"""Goals role.

Entry point: ``run(context)``. Builds the goals messages from the request, its
appraisal, its context, and the current goal list, validates the model JSON
into :class:`GoalsResult`, rejects any unsupported change kind, and writes
``outbox/report.yaml``. The message builder and the change guard are pure and
tested without a kernel, model, or Ghost.
"""
from __future__ import annotations

import json
from pathlib import Path

from workflows.main_cognition.schemas import GoalsResult
from workflows.main_cognition.stage_support import (
    StageOutputError, complete_json, load_yaml, read_request, write_report,
)

WORKFLOW_DIR = Path(__file__).resolve().parent

ALLOWED_CHANGES = ("retain", "add", "complete", "reprioritize", "defer", "refuse")


def load_prompts(path: Path | None = None) -> dict:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml")


def render_block(value) -> str:
    if value is None:
        return "(none)"
    return json.dumps(value, indent=2, sort_keys=False, default=str)


def render_goals(goals: list[str]) -> str:
    return "\n".join(f"- {goal}" for goal in goals) or "(none)"


def build_messages(request: dict, prompts: dict) -> list[dict]:
    goals = list(request.get("current_goals", request.get("active_goals", [])))
    system = prompts["system_prompt"].format(
        companion_name=request.get("companion_name", "Spamuel"),
    )
    user = prompts["user_template"].format(
        companion_name=request.get("companion_name", "Spamuel"),
        user_name=request.get("user_name", "Jesse"),
        state_summary=request.get("state_summary") or "(no recorded state update)",
        affect=render_block(request.get("affect")),
        current_goals=render_goals(goals),
        appraisal=render_block(request.get("appraisal")),
        context=render_block(request.get("context")),
    )
    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": user.strip()},
    ]


def validate_changes(result: GoalsResult) -> None:
    """Raise if the model proposed a change kind outside the allowed set."""
    for change in result.changes:
        if change.change not in ALLOWED_CHANGES:
            raise StageOutputError(
                f"Goals proposed an unsupported change {change.change!r}; "
                f"allowed changes are {', '.join(ALLOWED_CHANGES)}."
            )


def run(context) -> dict:
    request = read_request(context.runtime.environment)
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    result = complete_json(context, messages, GoalsResult, temperature=0.3)
    validate_changes(result)
    write_report(context.runtime.environment, result)
    return {
        "status": "done",
        "summary": f"applied {len(result.changes)} goal changes",
    }


__all__ = [
    "ALLOWED_CHANGES",
    "build_messages",
    "load_prompts",
    "render_block",
    "render_goals",
    "run",
    "validate_changes",
]
