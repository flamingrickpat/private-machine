"""Triage phase: rate each stimulus and each stimulus group.

The call is story-framed (A19F step 7): it starts with the shared story
prefix and ends with the triage instruction. The model classifies each
stimulus by relevance, urgency, and input kind, and sets the codelet budget
of the cycle. This decides how much effort the rest of the cycle spends.
The phase then gives each stimulus group the highest category of its
stimuli. A group whose stimuli are all ``Insignificant`` is marked
insignificant and gets no decision. The phase never writes memory.
"""
from __future__ import annotations

from pm.constants import CODELET_BUDGET_DEFAULT, CODELET_BUDGET_MAX
from pm.context_render.turns import Turn, format_content

from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.intake import group_key
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import TRIAGE_CATEGORIES, TriageResult
from workflows.main_cognition.story_prefix import StoryPrefix

#: Triage categories from the least to the most important.
CATEGORIES = TRIAGE_CATEGORIES


def render_stimuli(work: CycleWork, prefix: StoryPrefix) -> str:
    """Render each stimulus with its id, source, kind, text, and the story part of its group."""
    lines = []
    for stimulus in work.request.stimuli:
        part = prefix.label(group_key(stimulus, work.request))
        lines.append(f"- id={stimulus.knoxel_id} source={stimulus.source} "
                     f"kind={stimulus.kind} text={stimulus.text!r} part={part}")
    return "\n".join(lines) or "(none)"


def build_messages(work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix) -> list[Turn]:
    """Return the shared story prefix and then the triage instruction as the last turn."""
    request = work.request
    user = format_content(prefix.fill(prompts.user_template),
        state_summary=work.state_summary or "(no feeling recorded yet)",
        active_goals="; ".join(request.active_goals) or "(nothing in particular)",
        stimuli=render_stimuli(work, prefix),
        budget_max=CODELET_BUDGET_MAX,
        budget_default=CODELET_BUDGET_DEFAULT,
    )
    return prefix.ask(user.text)


def rate_groups(work: CycleWork) -> None:
    """Copy the triage categories onto the stimulus groups of the state.

    A group gets the highest category of its stimuli. A group without a
    rated stimulus keeps an empty category and stays significant.
    """
    for group in work.state.stimulus_groups:
        categories = [item.category for item in work.triage.items if item.knoxel_id in group.stimulus_ids]
        if not categories:
            continue
        ranks = [CATEGORIES.index(name) for name in categories]
        group.category = CATEGORIES[max(ranks)]
        group.insignificant = max(ranks) == 0


def run(context, work: CycleWork) -> CycleWork:
    prefix = StoryPrefix(work)
    result = complete(context, work, build_messages(work, phase_prompts("triage"), prefix), TriageResult,
                      grounded=True, emotion=True, grounding_at=len(prefix.turns))
    work.triage = result
    rate_groups(work)
    work.receipt("triage", f"triaged {len(result.items)} stimuli")
    return work


__all__ = ["CATEGORIES", "build_messages", "rate_groups", "render_stimuli", "run"]
