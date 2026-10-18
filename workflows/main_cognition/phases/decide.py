"""Decide phase: choose the action intents of this cycle.

This phase ports the old action selection stage. The model turns the
workspace frame, the goal changes, the available workflows, and the real
capabilities into zero or more action intents. An unsupported action type
or an unavailable workflow name fails validation. Without public context,
a content claim of a missing capability fails too.

The phase records each intent as an in-character decision of the primary
stimulus group. Step 8 of A19F splits this phase into one in-character
decision for each group and a separate routing call.
"""
from __future__ import annotations

import re

from pm.context_render import render_context
from pm.context_render.turns import Turn, format_content
from pm.records.cycle_records import ActionDecision

from workflows.main_cognition.cycle_work import CycleWork, primary_group
from workflows.main_cognition.model_json import PhasePrompts, StageOutputError
from workflows.main_cognition.phases.goals import render_inner_proposals
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import ActionSelectionResult, WorkflowOption
from workflows.main_cognition.story_prefix import StoryPrefix, conscious_percepts
from workflows.public_speech.public_context import PublicContext

ACTION_TYPES = (
    "Reply",
    "ToolCall",
    "ToolCallAndReply",
    "Ignore",
    "Idle",
    "Sleep",
)

#: Action types that make no dispatch. They stay in the decision record only.
QUIET_ACTIONS = frozenset({"Ignore", "Idle", "Sleep"})

#: The action space when the configuration names no workflows.
DEFAULT_WORKFLOWS = [
    WorkflowOption(name="dialogue_story", description="Generate one in-character Story reply."),
    WorkflowOption(name="dialogue_assistant", description="Assistant reply with scoped tools."),
    WorkflowOption(name="self_management", description="Start, status, update, or terminate a managed run."),
    WorkflowOption(name="local_task", description="Run one small real local task."),
    WorkflowOption(name="minecraft", description="Run a sustained Minecraft specialist with exclusive body ownership."),
]

# Claimed capability -> words that reveal a claim of it. The same words must
# appear in the declared capability list, or the claim is rejected.
CAPABILITY_MARKERS = {
    "vision": ("see", "seeing", "look", "watching", "camera", "vision", "observe"),
    "hearing": ("hear", "hearing", "listen", "audio", "microphone"),
    "mobility": ("walk", "navigate", "move", "travel", "go to"),
    "web": ("browse", "search the web", "internet", "online"),
}


def available_workflows(work: CycleWork) -> list[WorkflowOption]:
    """Return the configured workflows, or the default action space."""
    return work.request.available_workflows or DEFAULT_WORKFLOWS


def render_workflows(workflows: list[WorkflowOption]) -> str:
    lines = [f"- {workflow.name}: {workflow.description}" for workflow in workflows]
    return "\n".join(lines) or "(none)"


def render_capabilities(capabilities: list[str]) -> str:
    return "\n".join(f"- {capability}" for capability in capabilities) or "(none)"


def build_messages(work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix) -> list[Turn]:
    """Return the shared story prefix and then the action instruction as the last turn.

    The prefix shows the workspace shards that attention updated. Step 8 of
    A19F splits this call into an in-character decision and routing.
    """
    request = work.request
    system = prompts.system_prompt.format(
        companion_name=request.companion_name,
        wait_default_seconds=request.wait_default_seconds,
    )
    user = format_content(prompts.user_template,
        companion_name=request.companion_name,
        user_name=request.user_name,
        affect=work.affect.felt,
        regulation=work.affect.regulation,
        active_goals=", ".join(request.active_goals) or "(none)",
        workspace=render_context(work.workspace),
        goals=render_context(work.goals),
        context=render_context(work.context),
        stimuli=render_context(request.stimuli),
        managed_runs=render_context(request.managed_runs),
        available_workflows=render_workflows(available_workflows(work)),
        capabilities=render_capabilities(request.capabilities),
        inner_proposals=render_inner_proposals(conscious_percepts(work)),
    )
    return prefix.ask(f"{system.strip()}\n\n{user.text.strip()}")


def validate_actions(result: ActionSelectionResult, workflows: list[WorkflowOption]) -> None:
    """Reject an unsupported action type or an unavailable workflow name."""
    allowed = {workflow.name for workflow in workflows}
    for action in result.actions:
        if action.action_type not in ACTION_TYPES:
            raise StageOutputError(
                f"Action selection used unsupported action_type "
                f"{action.action_type!r}. Allowed: {', '.join(ACTION_TYPES)}."
            )
        if action.action_type in QUIET_ACTIONS and not action.workflow_name:
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


def check_capabilities(result: ActionSelectionResult, capabilities: list[str],
                       public_context: PublicContext | None = None) -> None:
    """Reject action content that claims a capability the companion lacks.

    A refusal that names the missing capability is not a claim. With public
    context, the domain evidence and the final claim review of public speech
    decide instead, because a verb alone cannot tell game movement, metaphor,
    and refusal apart.
    """
    if public_context is not None:
        return
    available = " ".join(capabilities).lower()
    for action in result.actions:
        content = action.content.lower()
        for capability, markers in CAPABILITY_MARKERS.items():
            if not _claims(content, markers):
                continue
            if _mentions(available, markers):
                continue
            raise StageOutputError(
                f"Action content claims capability {capability!r} that is not "
                "available."
            )


def run(context, work: CycleWork) -> CycleWork:
    # Earlier phases already supply appraisal and the selected workspace.
    # Hidden reasoning exhausted a real completion during a teasing choice.
    prefix = StoryPrefix(work)
    result = complete(context, work, build_messages(work, phase_prompts("decide"), prefix), ActionSelectionResult,
                      grounded=True, emotion=False, grounding_at=len(prefix.turns))
    validate_actions(result, available_workflows(work))
    check_capabilities(result, work.request.capabilities, work.request.public_context)
    work.action_selection = result
    group = primary_group(work)
    if group is not None:
        work.state.decisions = [ActionDecision(group=group.key, text=action.content, variants=[action.content],
                                               stimulus_ids=list(group.stimulus_ids))
                                for action in result.actions]
    work.receipt("decide", f"selected {len(result.actions)} actions")
    return work


__all__ = [
    "ACTION_TYPES",
    "CAPABILITY_MARKERS",
    "DEFAULT_WORKFLOWS",
    "QUIET_ACTIONS",
    "available_workflows",
    "build_messages",
    "check_capabilities",
    "primary_group",
    "render_capabilities",
    "render_workflows",
    "run",
    "validate_actions",
]
