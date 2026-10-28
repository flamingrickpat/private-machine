"""Private trigger role.

Entry point: ``run(context)``. Reads the fired triggers and the private
candidates the host derived, lets the model pick one operation, and writes the
validated ``TriggerChoice`` to ``outbox/report.yaml``. The message builder is
pure and tested without a kernel, model, or MemoryStore.
"""
from __future__ import annotations

from pathlib import Path
from workflows.inner_thought.rendering import candidates as render_candidates, sources as render_sources

from pm.constants import Sampler
from pm.context_render.turns import Content, Role, Turn, format_content

from workflows.inner_thought.operations import OPERATIONS
from workflows.inner_thought.schemas import InnerRequest, TriggerChoice
from workflows.main_cognition.model_json import complete_json, load_yaml, read_request, write_report

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


def _render_candidates(request: InnerRequest) -> Content:
    return render_candidates(request.candidates)


def _render_thoughts(request: InnerRequest) -> Content:
    return render_sources(request.recent_thoughts[-6:], "private_thought")


def build_messages(request: dict, prompts: dict) -> list[Turn]:
    """Build typed turns from the fired triggers and private candidates."""
    parsed = InnerRequest.model_validate(request)
    system = prompts["system_prompt"].format(companion_name=parsed.companion_name)
    user = format_content(prompts["user_template"],
        companion_name=parsed.companion_name,
        user_name=parsed.user_name,
        state_summary=parsed.state_summary or "(no recorded state update)",
        idle="unknown" if parsed.seconds_since_user_input is None else f"{parsed.seconds_since_user_input:.0f}",
        goals=render_sources(parsed.active_goal_records, "goal"),
        expectations=render_sources(parsed.pending_expectations, "expectation"),
        thoughts=_render_thoughts(parsed),
        triggers=_render_triggers(parsed),
        candidates=_render_candidates(parsed),
    )
    import json
    user = user + "\nPrivate episode and operation suggestions:\n" + json.dumps(parsed.episode)
    from pm.context_render import render_context
    from workflows.reflection.rendering import status
    user = user + "\nPrivate experiment status:\n" + render_context(status(parsed.reflection))
    return [Turn(Role.System, system.strip()), Turn(Role.User, user.strip())]


def run(context) -> dict:
    request = read_request(context.runtime.environment)
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    result = complete_json(context, messages, TriggerChoice, temperature=Sampler.TempInstruct, enable_thinking=False)
    write_report(context.runtime.environment, result)
    return {
        "status": "done",
        "summary": f"chose operation {result.operation or 'none'} from trigger {result.trigger or 'none'}",
    }


__all__ = ["build_messages", "load_prompts", "run"]
