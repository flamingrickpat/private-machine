"""Private trigger role.

Entry point: ``run(context)``. Reads the fired triggers and the private
candidates the host derived, lets the model pick one operation, and writes the
validated ``TriggerChoice`` to ``outbox/report.yaml``. The message builder is
pure and tested without a kernel, model, or MemoryStore.
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Literal
from pydantic import BaseModel, ConfigDict
from workflows.inner_thought.rendering import candidates as render_candidates, sources as render_sources

from pm.constants import Sampler
from pm_workflows.python_role import RoleContext
from pm.context_render.turns import Content, Role, Turn, format_content

from workflows.inner_thought.operations import OPERATIONS
from workflows.inner_thought.schemas import InnerRequest, TriggerChoice
from workflows.inner_thought.schemas import PrivateAttentionContext, ThoughtAttentionDecision
from workflows.cognitive_control.attention import validate_private_attention
from workflows.main_cognition.model_json import complete_json, load_yaml, read_request, write_report

WORKFLOW_DIR = Path(__file__).resolve().parent


class TriggerPrompts(BaseModel):
    """Required prompt texts for legacy triggers and separate private attention."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    system_prompt: str
    user_template: str
    attention_system_prompt: str
    attention_user_template: str


class AttentionStageRequest(BaseModel):
    """Frozen private sources beside the shared owner's actual competing context."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    stage_kind: Literal["private_attention"] = "private_attention"
    request: InnerRequest
    attention: PrivateAttentionContext


def load_prompts(path: Path | None = None) -> TriggerPrompts:
    """Read both selection contracts without defaulting missing prompt texts."""
    return TriggerPrompts.model_validate(load_yaml(path or WORKFLOW_DIR / "prompts.yaml"))


def _render_triggers(request: InnerRequest) -> str:
    lines: list[str] = []
    for candidate in request.trigger_candidates:
        allowed = ", ".join(name for name in candidate.operations if name in OPERATIONS) or "none"
        lines.append(f"- {candidate.name} (urgency {candidate.urgency:.2f}): {candidate.cause}. "
                     f"Allowed operations: {allowed}.")
    return "\n".join(lines) or "(none)"


def _render_candidates(request: InnerRequest) -> Content:
    return render_candidates(request.candidates)


def _render_thoughts(request: InnerRequest) -> Content:
    return render_sources(request.recent_thoughts[-6:], "private_thought")


def build_messages(request: dict[str, object], prompts: TriggerPrompts) -> list[Turn]:
    """Build typed turns from the fired triggers and private candidates."""
    parsed = InnerRequest.model_validate(request)
    system = prompts.system_prompt.format(companion_name=parsed.companion_name)
    user = format_content(prompts.user_template,
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
    episode = parsed.episode.model_dump_json()
    user = user + "\nPrivate episode and operation suggestions:\n" + episode
    from pm.context_render import render_context
    from workflows.reflection.rendering import status
    user = user + "\nPrivate experiment status:\n" + render_context(status(parsed.reflection))
    return [Turn(Role.System, system.strip()), Turn(Role.User, user.strip())]


def build_attention_messages(request: AttentionStageRequest, prompts: TriggerPrompts) -> list[Turn]:
    """Compare actual committed proposals under accepted shared guidance."""
    system = prompts.attention_system_prompt.format(companion_name=request.request.companion_name)
    delivered = "\n".join(f"- {operation.name}: {operation.description}" for operation in OPERATIONS.values())
    user = prompts.attention_user_template.format(
        request=request.request.model_dump_json(), attention=request.attention.model_dump_json(),
        operations=delivered,
        schema=json.dumps(ThoughtAttentionDecision.model_json_schema(), ensure_ascii=False))
    return [Turn(Role.System, system.strip()), Turn(Role.User, user.strip())]


def run(context: RoleContext) -> dict[str, str]:
    """Choose one of the deterministic trigger candidates, or none."""
    assert context.runtime is not None
    request = read_request(context.runtime.environment)
    prompts = load_prompts()
    if request.get("stage_kind") == "private_attention":
        selection = AttentionStageRequest.model_validate(request)

        def validate(payload: dict[str, object], raw: str) -> ThoughtAttentionDecision:
            """Retain supplied identities and accepted admission through model repair."""
            return validate_private_attention(selection.attention, ThoughtAttentionDecision.model_validate(payload))

        decision = complete_json(context, build_attention_messages(selection, prompts), ThoughtAttentionDecision,
                                 temperature=Sampler.TempInstruct, enable_thinking=False, validate=validate)
        write_report(context.runtime.environment, decision)
        return {"status": "done", "summary": f"private attention {decision.disposition}: {decision.reason}"}
    messages = build_messages(request, prompts)
    result = complete_json(context, messages, TriggerChoice, temperature=Sampler.TempInstruct, enable_thinking=False)
    write_report(context.runtime.environment, result)
    return {
        "status": "done",
        "summary": f"chose operation {result.operation or 'none'} from trigger {result.trigger or 'none'}",
    }


__all__ = ["AttentionStageRequest", "build_attention_messages", "build_messages", "load_prompts", "run"]
