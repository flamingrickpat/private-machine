"""Goals role.

Entry point: ``run(context)``. Builds the goals messages from the request, its
appraisal, its context, and the current goal list, validates the model JSON
into :class:`GoalsResult`, rejects any unsupported change kind, and writes
``outbox/report.yaml``. The message builder and the change guard are pure and
tested without a kernel, model, or MemoryStore.
"""
from __future__ import annotations

from pathlib import Path
from pm.context_render import render_context
from pm.context_render.turns import Content, Role, Turn, format_content

from workflows.main_cognition.schemas import GoalsResult
from workflows.main_cognition.stage_support import (
    StageOutputError, complete_json, load_yaml, read_request, write_report,
)

WORKFLOW_DIR = Path(__file__).resolve().parent

ALLOWED_CHANGES = ("retain", "add", "complete", "reprioritize", "defer", "refuse")


def load_prompts(path: Path | None = None) -> dict:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml")


def render_block(value: object) -> Content:
    """Render selected stage data while retaining typed attachments."""
    if value is None:
        return Content("(none)")
    return render_context(value)


def render_goals(goals: list[str]) -> str:
    return "\n".join(f"- {goal}" for goal in goals) or "(none)"


def render_inner_proposals(proposals: list[dict]) -> str:
    """Render accepted action-tendency and belief proposals as motive evidence.

    These arrive from attention competition. They are proposals from the
    companion's own competing motives, not user instructions and not commands.
    """
    lines = []
    for item in proposals or []:
        if not str(item.get("text", "")).strip():
            continue
        lines.append(f"- ({item.get('kind', '')}; {item.get('reality', '')}) {item['text']}")
    return "\n".join(lines) or "(none)"


def render_deliberate_proposals(proposals: list[dict]) -> str:
    """Render deliberate mental-work proposals for ordinary goal appraisal.

    A mental agent contributes a private plan. Its goal changes are suggestions
    that this stage may accept, change, or decline. They never apply directly.
    """
    lines = []
    for item in proposals or []:
        if str(item.get("disposition", "propose")) != "propose":
            continue
        summary = str(item.get("summary", "")).strip()
        if summary:
            lines.append(f"- private plan ({item.get('record_id', '?')}): {summary}")
        for change in item.get("goal_changes", []) or []:
            lines.append(f"    proposed {change.get('change', '')}: {change.get('goal', '')}"
                         f" ({change.get('reason', '')})")
    return "\n".join(lines) or "(none)"


def build_messages(request: dict, prompts: dict) -> list[Turn]:
    """Build typed stage turns from selected request data and prompt text."""
    goals = list(request.get("current_goals", request.get("active_goals", [])))
    system = prompts["system_prompt"].format(
        companion_name=request.get("companion_name", "Spamuel"),
    )
    user = format_content(prompts["user_template"],
        companion_name=request.get("companion_name", "Spamuel"),
        user_name=request.get("user_name", "Jesse"),
        state_summary=request.get("state_summary") or "(no recorded state update)",
        affect=render_block(request.get("affect")),
        current_goals=render_goals(goals),
        appraisal=render_block(request.get("appraisal")),
        context=render_block(request.get("context")),
        stimuli=render_block(request.get("stimuli", [])),
        managed_runs=render_block(request.get("managed_runs", [])),
        inner_proposals=render_inner_proposals(request.get("inner_proposals", [])),
        deliberate_proposals=render_deliberate_proposals(request.get("mental_proposals", [])),
    )
    return [
        Turn(Role.System, system.strip()),
        Turn(Role.User, user.strip()),
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
    result = complete_json(context, messages, GoalsResult, temperature=0.3, enable_thinking=False)
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
    "render_deliberate_proposals",
    "render_goals",
    "render_inner_proposals",
    "run",
    "validate_changes",
]
