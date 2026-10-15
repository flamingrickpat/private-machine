"""Private integration role.

Entry point: ``run(context)``. Reads the performed thought and proposes its
accepted consequences as a validated ``IntegrationResult``. The message
builder is pure and tested without a kernel, model, or MemoryStore.
"""
from __future__ import annotations

from pathlib import Path

from pm.context_render.turns import Role, Turn, format_content

from workflows.inner_thought.schemas import InnerRequest, IntegrationResult, ThoughtResult, TriggerChoice
from workflows.main_cognition.stage_support import complete_json, load_yaml, read_request, write_report

WORKFLOW_DIR = Path(__file__).resolve().parent


def load_prompts(path: Path | None = None) -> dict:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml")


def build_messages(request: dict, prompts: dict) -> list[Turn]:
    """Build typed turns for one integration decision."""
    parsed = InnerRequest.model_validate(request)
    trigger = TriggerChoice.model_validate(request.get("trigger") or {})
    thought = ThoughtResult.model_validate(request.get("thought") or {})
    operation = (request.get("operation") or {}).get("name", "")
    system = prompts["system_prompt"].format(companion_name=parsed.companion_name)
    user = format_content(prompts["user_template"],
        companion_name=parsed.companion_name,
        state_summary=parsed.state_summary or "(no recorded state update)",
        reality=thought.reality,
        operation=operation or trigger.operation,
        thought=thought.thought or "(empty)",
        evidence_summary=thought.evidence_summary or "(none)",
        interpretation=thought.interpretation or "(none)",
        goals="\n".join(f"- id={item['knoxel_id']} {item['text']}" for item in parsed.active_goal_records) or "(none)",
        expectations="\n".join(f"- id={item['knoxel_id']} {item.get('content', '')}"
                               f" (pending since tick {item.get('tick_id', 0)})"
                               for item in parsed.pending_expectations) or "(none)",
    )
    return [Turn(Role.System, system.strip()), Turn(Role.User, user.strip())]


def run(context) -> dict:
    request = read_request(context.runtime.environment)
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    result = complete_json(context, messages, IntegrationResult, temperature=0.2, enable_thinking=False)
    write_report(context.runtime.environment, result)
    return {
        "status": "done",
        "summary": result.summary or "integrated the private thought",
    }


__all__ = ["build_messages", "load_prompts", "run"]
