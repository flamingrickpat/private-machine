"""Goals phase: propose goal changes for this cycle.

The call is story-framed (A19F step 7). It starts with the shared story
prefix, so it reads the workspace shards that attention just updated and the
subjective experience in the self tail. The goal instruction comes last.
The model reviews the current goals against the appraisal and the selected
memories. It may keep, add, complete, reprioritize, defer, or refuse goals.
The codelet percepts in the shards and the private plans from deliberate
mental work enter as the companion's own urges, never as commands. Any
change kind outside :data:`ALLOWED_CHANGES` fails validation. The commit
applies the changes. This phase never writes memory.
"""
from __future__ import annotations

from pydantic import create_model, model_validator

from pm.context_render import render_context
from pm.context_render.turns import Turn, format_content

from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts
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
    user = format_content(prefix.fill(prompts.user_template),
        affect=work.affect.felt or work.state_summary or "(no feeling recorded)",
        regulation_words=prompts.regulation_words[work.affect.regulation],
        current_goals=render_goals(request.active_goals),
        appraisal=work.appraisal.feeling or "(no judgement recorded)",
        context=render_goals(work.context.memories),
        managed_runs=render_context(request.managed_runs) if request.managed_runs else "(none)",
        inner_proposals=render_inner_proposals(conscious_percepts(work)),
        deliberate_proposals=render_deliberate_proposals(request.mental_proposals),
    )
    return prefix.ask(user.text)


def validate_changes(result: GoalsResult) -> GoalsResult:
    """Return ``result``, or raise ValueError for a change kind outside :data:`ALLOWED_CHANGES`.

    The call validates its answer with this function, so an invalid kind
    is invalid model output and gets a repair turn (a replay once returned
    an empty kind and stopped the cycle).
    """
    for change in result.changes:
        if change.change not in ALLOWED_CHANGES:
            raise ValueError(f"Goal change {change.change!r} is not allowed. "
                             f"Use one of: {', '.join(ALLOWED_CHANGES)}.")
    return result


#: The answer model of the goals call: ``GoalsResult`` with the change kinds checked.
CheckedGoals = create_model("CheckedGoals", __base__=GoalsResult,
                            __validators__={"allowed": model_validator(mode="after")(validate_changes)})


def run(context, work: CycleWork) -> CycleWork:
    prefix = StoryPrefix(work)
    answer = complete(context, work, build_messages(work, phase_prompts("goals"), prefix), CheckedGoals,
                      grounded=True, emotion=False, grounding_at=len(prefix.turns))
    result = GoalsResult.model_validate(answer.model_dump())
    work.goals = result
    work.receipt("goals", f"proposed {len(result.changes)} goal changes")
    return work


__all__ = [
    "ALLOWED_CHANGES",
    "CheckedGoals",
    "build_messages",
    "render_deliberate_proposals",
    "render_goals",
    "render_inner_proposals",
    "run",
    "validate_changes",
]
