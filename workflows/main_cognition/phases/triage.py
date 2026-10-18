"""Triage phase: rate each stimulus and each stimulus group.

The model classifies each stimulus by relevance, urgency, and input kind.
This decides how much effort the rest of the cycle spends. The phase then
gives each stimulus group the highest category of its stimuli. A group whose
stimuli are all ``Insignificant`` is marked insignificant. It never writes
memory.
"""
from __future__ import annotations

from pm.context_render.turns import Role, Turn, format_content

from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import Stimulus, TriageResult

#: Triage categories from the least to the most important.
CATEGORIES = ("Insignificant", "Moderate", "Significant", "Critical")


def render_stimuli(stimuli: list[Stimulus]) -> str:
    lines = []
    for stimulus in stimuli:
        lines.append(f"- id={stimulus.knoxel_id} source={stimulus.source} "
                     f"kind={stimulus.kind} text={stimulus.text!r}")
    return "\n".join(lines) or "(none)"


def build_messages(work: CycleWork, prompts: PhasePrompts) -> list[Turn]:
    """Build typed turns from the current stimuli and prompt text."""
    request = work.request
    system = prompts.system_prompt.format(companion_name=request.companion_name)
    user = format_content(prompts.user_template,
        companion_name=request.companion_name,
        user_name=request.user_name,
        state_summary=work.state_summary or "(no recorded state update)",
        active_goals=", ".join(request.active_goals) or "(none)",
        stimuli=render_stimuli(request.stimuli),
    )
    return [
        Turn(Role.System, system.strip()),
        Turn(Role.User, user.strip()),
    ]


def rate_groups(work: CycleWork) -> None:
    """Copy the triage categories onto the stimulus groups of the state.

    An unknown category ranks as ``Insignificant``. A group without a rated
    stimulus keeps an empty category and stays significant.
    """
    for group in work.state.stimulus_groups:
        categories = [item.category for item in work.triage.items if item.knoxel_id in group.stimulus_ids]
        if not categories:
            continue
        ranks = [CATEGORIES.index(name) if name in CATEGORIES else 0 for name in categories]
        group.category = CATEGORIES[max(ranks)]
        group.insignificant = max(ranks) == 0


def run(context, work: CycleWork) -> CycleWork:
    result = complete(context, work, build_messages(work, phase_prompts("triage")), TriageResult,
                      grounded=True, emotion=True)
    work.triage = result
    rate_groups(work)
    work.receipt("triage", f"triaged {len(result.items)} stimuli")
    return work


__all__ = ["CATEGORIES", "build_messages", "rate_groups", "render_stimuli", "run"]
