"""Action selection role.

Entry point: ``run(context)``. Builds the action-selection messages from the
workspace frame, the goal changes, the available workflows, and the companion's
real capabilities, then validates the model JSON into
:class:`ActionSelectionResult`. The message builder and the guards are pure and
tested without a kernel, model, or MemoryStore.

``ActionIntent`` carries no id. This stage never invents one. ``depends_on``
entries are the labels the model wrote (the workflow name of another action in
the same result). The parent assigns the durable ``action-<index>`` ids.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from pm.constants import Sampler
from pm.context_render import render_context
from pm.context_render.turns import Content, Role, Turn, format_content

from workflows.main_cognition.schemas import ActionSelectionResult
from workflows.main_cognition.stage_support import (
    StageOutputError, complete_json, load_yaml, read_request, write_report,
)

WORKFLOW_DIR = Path(__file__).resolve().parent

ACTION_TYPES = (
    "Reply",
    "ToolCall",
    "ToolCallAndReply",
    "Ignore",
    "Idle",
    "Sleep",
)

# Claimed capability -> words that reveal a claim of it. The same words must
# appear in the declared capability list, or the claim is rejected.
CAPABILITY_MARKERS = {
    "vision": ("see", "seeing", "look", "watching", "camera", "vision", "observe"),
    "hearing": ("hear", "hearing", "listen", "audio", "microphone"),
    "mobility": ("walk", "navigate", "move", "travel", "go to"),
    "web": ("browse", "search the web", "internet", "online"),
}


def load_prompts(path: Path | None = None) -> dict:
    return load_yaml(path or WORKFLOW_DIR / "prompts.yaml")


def workflow_names(available_workflows: list[dict]) -> list[str]:
    return [str(workflow.get("name", "")) for workflow in available_workflows]


def render_workflows(available_workflows: list[dict]) -> str:
    lines = [
        f"- {workflow.get('name', '')}: {workflow.get('description', '')}"
        for workflow in available_workflows
    ]
    return "\n".join(lines) or "(none)"


def render_capabilities(capabilities: list[str]) -> str:
    return "\n".join(f"- {capability}" for capability in capabilities) or "(none)"


def render_inner_proposals(proposals: list[dict]) -> str:
    """Render accepted want, imagination, and steady-yourself proposals.

    They come from the companion's own attention competition. Each line keeps
    its reality label, so an imagined outcome never reads as an event.
    """
    lines = []
    for item in proposals or []:
        if not str(item.get("text", "")).strip():
            continue
        lines.append(f"- ({item.get('kind', '')}; {item.get('reality', '')}) {item['text']}")
    return "\n".join(lines) or "(none)"


def render_block(value: object) -> Content:
    """Render selected stage data while retaining typed attachments."""
    if value is None:
        return Content("(none)")
    return render_context(value)


def build_messages(request: dict, prompts: dict) -> list[Turn]:
    available_workflows = list(request.get("available_workflows", []))
    system = prompts["system_prompt"].format(
        companion_name=request.get("companion_name", "Spamuel"),
        wait_default_seconds=request.get("wait_default_seconds", 1800),
    )
    user = format_content(prompts["user_template"],
        companion_name=request.get("companion_name", "Spamuel"),
        user_name=request.get("user_name", "Jesse"),
        state_summary=request.get("state_summary") or "(no recorded state update)",
        affect=render_block(request.get("affect")),
        active_goals=", ".join(request.get("active_goals", [])) or "(none)",
        workspace=render_block(request.get("workspace")),
        goals=render_block(request.get("goals")),
        context=render_block(request.get("context")),
        stimuli=render_block(request.get("stimuli", [])),
        managed_runs=render_block(request.get("managed_runs", [])),
        available_workflows=render_workflows(available_workflows),
        capabilities=render_capabilities(list(request.get("capabilities", []))),
        inner_proposals=render_inner_proposals(request.get("inner_proposals", [])),
    )
    return [
        Turn(Role.System, system.strip()),
        Turn(Role.User, user.strip()),
    ]


def validate_actions(result: ActionSelectionResult, available_workflows: list[dict]) -> None:
    """Reject an unsupported action type or an unavailable workflow name."""
    allowed = set(workflow_names(available_workflows))
    for action in result.actions:
        if action.action_type not in ACTION_TYPES:
            raise StageOutputError(
                f"Action selection used unsupported action_type "
                f"{action.action_type!r}. Allowed: {', '.join(ACTION_TYPES)}."
            )
        if action.action_type in {"Ignore", "Idle", "Sleep"} and not action.workflow_name:
            continue
        if action.workflow_name not in allowed:
            raise StageOutputError(
                f"Action selection named workflow {action.workflow_name!r} that "
                "is not available."
            )


def _mentions(text: str, markers: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(marker)}", text) for marker in markers)


# A refusal names the capability it lacks: "I can't walk to the store". A denial
# is not a claim, so a marker only counts when no negation precedes it.
_NEGATION = re.compile(
    r"\b(?:no|not|never|none|without|cannot|can not|can't|won't|will not|don't|do not|"
    r"doesn't|does not|didn't|did not|isn't|is not|aren't|are not|unable|incapable|"
    r"lack|lacks|lacking)\b"
)


def _claims(text: str, markers: tuple[str, ...]) -> bool:
    """True if a marker is used affirmatively rather than inside a denial."""
    for marker in markers:
        for match in re.finditer(rf"\b{re.escape(marker)}", text):
            before = text[max(0, match.start() - 30):match.start()]
            if not _NEGATION.search(before):
                return True
    return False


def check_capabilities(result: ActionSelectionResult, capabilities: list[str], public_context: dict | None = None) -> None:
    """Reject action content that claims a capability the companion lacks.

    A refusal that names the missing capability is not a claim.
    """
    if public_context:
        # Production uses domain evidence and final semantic claim review.
        # A verb alone cannot distinguish game movement, metaphor, or refusal.
        return
    available = " ".join(str(capability) for capability in capabilities).lower()
    for action in result.actions:
        content = (action.content or "").lower()
        for capability, markers in CAPABILITY_MARKERS.items():
            if not _claims(content, markers):
                continue
            if _mentions(available, markers):
                continue
            raise StageOutputError(
                f"Action content claims capability {capability!r} that is not "
                "available."
            )


def run(context) -> dict:
    request = read_request(context.runtime.environment)
    prompts = load_prompts()
    messages = build_messages(request, prompts)
    # Earlier stages already supply appraisal and the selected workspace.
    # Hidden reasoning exhausted a real completion during a teasing choice.
    result = complete_json(context, messages, ActionSelectionResult, temperature=Sampler.TempInstruct,
                           enable_thinking=False)
    validate_actions(result, list(request.get("available_workflows") or []))
    check_capabilities(result, list(request.get("capabilities") or []), request.get("public_context"))
    write_report(context.runtime.environment, result)
    return {
        "status": "done",
        "summary": f"selected {len(result.actions)} actions",
    }


__all__ = [
    "ACTION_TYPES",
    "CAPABILITY_MARKERS",
    "build_messages",
    "check_capabilities",
    "load_prompts",
    "render_block",
    "render_capabilities",
    "render_inner_proposals",
    "render_workflows",
    "run",
    "validate_actions",
    "workflow_names",
]
