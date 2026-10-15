"""Private thought role.

Entry point: ``run(context)``. Performs the chosen operation over the frozen
private candidates and writes the validated ``ThoughtResult``. The message
builder is pure and tested without a kernel, model, or MemoryStore.
"""
from __future__ import annotations

from pathlib import Path

from pm.context_render.turns import Role, Turn, format_content

from workflows.inner_thought.operations import validate_operation
from workflows.inner_thought.schemas import InnerRequest, ThoughtResult, TriggerChoice
from workflows.main_cognition.stage_support import complete_json, load_yaml, read_request, write_report

WORKFLOW_DIR = Path(__file__).resolve().parent


def load_prompts(path: Path | None = None) -> dict:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml")


def _render_candidates(request: InnerRequest, chosen: set[int]) -> str:
    lines = []
    for item in request.candidates:
        marker = "*" if item.knoxel_id in chosen else ""
        lines.append(f"- id={item.knoxel_id} kind={item.kind} revision={item.revision}{marker} "
                     f"text={item.text!r}")
    for item in request.active_goal_records:
        lines.append(f"- id={item['knoxel_id']} kind=goal revision={item.get('revision', 1)} "
                     f"text={item.get('text', '')!r}")
    for item in request.pending_expectations:
        lines.append(f"- id={item['knoxel_id']} kind=expectation text={item.get('content', '')!r}")
    return "\n".join(lines) or "(none)"


def _render_inspection(request: InnerRequest) -> str:
    """Render the selected implementation evidence as compact text.

    The host supplies this evidence from a real source grant and real records.
    A missing or failed inspection becomes an explicit limit, never a guess.
    """
    if not request.inspection:
        return "(no implementation evidence was supplied for this cycle)"
    if request.inspection.get("status") != "ok":
        return f"(inspection unavailable: {request.inspection.get('reason', 'unknown')})"
    lines: list[str] = []
    overlay = request.inspection.get("capability_overlay") or {}
    for item in overlay.get("capabilities", []):
        lines.append(f"- capability={item.get('name')} state={item.get('state')} "
                     f"summary={item.get('summary', '')!r}")
    for episode in request.inspection.get("episodes", [])[:3]:
        lines.append(f"- episode={episode.get('episode_id')} loop={episode.get('loop')} "
                     f"operation={episode.get('operation')} outcome={episode.get('outcome')} "
                     f"summary={str(episode.get('summary', ''))[:140]!r}")
    answers = request.inspection.get("newest_episode_answers") or {}
    if answers:
        rendered = "; ".join(f"{key}: {value.get('answer', '')}" for key, value in answers.items())
        lines.append(f"- newest episode answers: {rendered}")
    return "\n".join(lines) or "(no implementation evidence was supplied for this cycle)"


def build_messages(request: dict, prompts: dict) -> list[Turn]:
    """Build typed turns for one thought operation."""
    parsed = InnerRequest.model_validate(request)
    trigger = TriggerChoice.model_validate(request.get("trigger") or {})
    operation = validate_operation(str(request.get("operation", {}).get("name", trigger.operation or "none")))
    system = prompts["system_prompt"].format(
        companion_name=parsed.companion_name,
        operation_name=operation.name, operation_group=operation.group,
        operation_mode=operation.mode, operation_focus=operation.temporal_focus,
        operation_description=operation.description,
    )
    chosen = {int(item) for item in trigger.candidate_ids}
    user = format_content(prompts["user_template"],
        companion_name=parsed.companion_name,
        state_summary=parsed.state_summary or "(no recorded state update)",
        trigger_reason=trigger.reason or parsed.trigger_candidates[0].cause if parsed.trigger_candidates else "(no trigger)",
        candidates=_render_candidates(parsed, chosen),
        goals="\n".join(f"- {item['text']}" for item in parsed.active_goal_records) or "(none)",
        expectations="\n".join(f"- {item.get('content', '')}" for item in parsed.pending_expectations) or "(none)",
        thoughts="\n".join(f"- {item.get('text', '')!r}" for item in parsed.recent_thoughts[-6:]) or "(none)",
        inspection=_render_inspection(parsed),
    )
    return [Turn(Role.System, system.strip()), Turn(Role.User, user.strip())]


def run(context) -> dict:
    request = read_request(context.runtime.environment)
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    result = complete_json(context, messages, ThoughtResult, temperature=0.4)
    write_report(context.runtime.environment, result)
    return {
        "status": "done",
        "summary": f"performed thought with reality {result.reality}",
    }


__all__ = ["build_messages", "load_prompts", "run"]
