"""Goals phase: propose goal changes for this cycle.

The model reviews the current goals against the appraisal and the selected
context. It may keep, add, complete, reprioritize, defer, or refuse goals.
The call starts with the shared story prefix, so it reads the workspace
shards that attention just updated. The codelet percepts in those shards and
the private plans from deliberate mental work enter as suggestions, never as
commands. Any change kind outside :data:`ALLOWED_CHANGES` fails validation.
The commit applies the changes. This phase never writes memory.

Step 7 of A19F rewrites this phase as a story-framed goal call. Until then
the old instruction text follows the prefix.
"""
from __future__ import annotations

from pm.context_render import render_context
from pm.context_render.turns import Turn, format_content

from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts, StageOutputError
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import GoalsResult, MentalProposal
from workflows.main_cognition.story_prefix import StoryPrefix, conscious_percepts

ALLOWED_CHANGES = ("retain", "add", "complete", "reprioritize", "defer", "refuse")

def render_inner_proposals(lines: list[str]) -> str:
    """Render the conscious codelet percepts as motive evidence.

    These are the companion's own competing impulses that attention put into
    the workspace. They are not user instructions and not commands.
    """
    return "\n".join(f"- {line}" for line in lines) or "(none)"


def render_deliberate_proposals(proposals: list[MentalProposal]) -> str:
    """Render deliberate mental-work proposals for ordinary goal appraisal.

    A mental agent contributes a private plan. Its goal changes are suggestions
    that this phase may accept, change, or decline. They never apply directly.
    """
    lines = []
    for item in proposals:
        if item.disposition != "propose":
            continue
        summary = item.summary.strip()
        if summary:
            lines.append(f"- private plan ({item.record_id}): {summary}")
        for change in item.goal_changes:
            lines.append(f"    proposed {change.change}: {change.goal} ({change.reason})")
    return "\n".join(lines) or "(none)"


def render_goals(goals: list[str]) -> str:
    return "\n".join(f"- {goal}" for goal in goals) or "(none)"


def build_messages(work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix) -> list[Turn]:
    """Return the shared story prefix and then the goal instruction as the last turn."""
    request = work.request
    system = prompts.system_prompt.format(companion_name=request.companion_name)
    user = format_content(prompts.user_template,
        companion_name=request.companion_name,
        user_name=request.user_name,
        affect=work.affect.felt,
        regulation=work.affect.regulation,
        current_goals=render_goals(request.active_goals),
        appraisal=render_context(work.appraisal_view()),
        context=render_context(work.context),
        managed_runs=render_context(request.managed_runs),
        inner_proposals=render_inner_proposals(conscious_percepts(work)),
        deliberate_proposals=render_deliberate_proposals(request.mental_proposals),
    )
    return prefix.ask(f"{system.strip()}\n\n{user.text.strip()}")


def validate_changes(result: GoalsResult) -> None:
    """Raise if the model proposed a change kind outside the allowed set."""
    for change in result.changes:
        if change.change not in ALLOWED_CHANGES:
            raise StageOutputError(
                f"Goals proposed an unsupported change {change.change!r}; "
                f"allowed changes are {', '.join(ALLOWED_CHANGES)}."
            )


def run(context, work: CycleWork) -> CycleWork:
    prefix = StoryPrefix(work)
    result = complete(context, work, build_messages(work, phase_prompts("goals"), prefix), GoalsResult,
                      grounded=True, emotion=False, grounding_at=len(prefix.turns))
    validate_changes(result)
    work.goals = result
    work.receipt("goals", f"proposed {len(result.changes)} goal changes")
    return work


__all__ = [
    "ALLOWED_CHANGES",
    "build_messages",
    "render_deliberate_proposals",
    "render_goals",
    "render_inner_proposals",
    "run",
    "validate_changes",
]
