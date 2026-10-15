"""Private trigger role.

Entry point: ``run(context)``. Reads the fired triggers and the private
candidates the host derived, lets the model pick one operation, and writes the
validated ``TriggerChoice`` to ``outbox/report.yaml``. The message builder is
pure and tested without a kernel, model, or MemoryStore.
"""
from __future__ import annotations

from pathlib import Path

from pm.context_render.turns import Role, Turn, format_content

from workflows.inner_thought.operations import OPERATIONS
from workflows.inner_thought.schemas import InnerRequest, TriggerChoice
from workflows.main_cognition.stage_support import complete_json, load_yaml, read_request, write_report

WORKFLOW_DIR = Path(__file__).resolve().parent


def load_prompts(path: Path | None = None) -> dict:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml")


def _render_triggers(request: InnerRequest) -> str:
    lines = []
    for candidate in request.trigger_candidates:
        allowed = ", ".join(name for name in candidate.operations if name in OPERATIONS) or "none"
        lines.append(f"- {candidate.name} (urgency {candidate.urgency:.2f}): {candidate.cause}. "
                     f"Allowed operations: {allowed}.")
    return "\n".join(lines) or "(none)"


def _render_candidates(request: InnerRequest) -> str:
    lines = []
    for item in request.candidates:
        lines.append(f"- id={item.knoxel_id} kind={item.kind} urgency={item.urgency:.2f} "
                     f"text={item.text!r}")
    return "\n".join(lines) or "(none)"


def _render_thoughts(request: InnerRequest) -> str:
    lines = []
    for item in request.recent_thoughts[-6:]:
        lines.append(f"- tick {item.get('tick_id', 0)} operation={item.get('operation', '')}: {item.get('text', '')!r}")
    return "\n".join(lines) or "(none)"


def build_messages(request: dict, prompts: dict) -> list[Turn]:
    """Build typed turns from the fired triggers and private candidates."""
    parsed = InnerRequest.model_validate(request)
    system = prompts["system_prompt"].format(companion_name=parsed.companion_name)
    user = format_content(prompts["user_template"],
        companion_name=parsed.companion_name,
        user_name=parsed.user_name,
        state_summary=parsed.state_summary or "(no recorded state update)",
        idle="unknown" if parsed.seconds_since_user_input is None else f"{parsed.seconds_since_user_input:.0f}",
        goals="\n".join(f"- {item['text']}" for item in parsed.active_goal_records) or "(none)",
        expectations="\n".join(f"- {item.get('content', '')}" for item in parsed.pending_expectations) or "(none)",
        thoughts=_render_thoughts(parsed),
        triggers=_render_triggers(parsed),
        candidates=_render_candidates(parsed),
    )
    return [Turn(Role.System, system.strip()), Turn(Role.User, user.strip())]


def run(context) -> dict:
    request = read_request(context.runtime.environment)
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    result = complete_json(context, messages, TriggerChoice, temperature=0.1, enable_thinking=False)
    write_report(context.runtime.environment, result)
    return {
        "status": "done",
        "summary": f"chose operation {result.operation or 'none'} from trigger {result.trigger or 'none'}",
    }


__all__ = ["build_messages", "load_prompts", "run"]
