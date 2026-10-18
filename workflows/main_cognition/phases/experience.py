"""Experience phase: frame the workspace and its subjective experience.

This phase ports the old workspace stage. The call starts with the shared
story prefix, so the model reads the workspace shards that attention just
updated. The instruction lists what entered awareness in this cycle, the
goal changes, and the candidate records. The model returns the frame
reason, the focus ids, the control mode, and the subjective experience.
Focus ids must come from the items of the shown shards or the stimuli. An
invalid frame gets one correction.

Step 7 of A19F replaces this phase with the story-framed subjective
experience in the self shard.
"""
from __future__ import annotations

import json

from pydantic import BaseModel

from pm.context_render import render_context
from pm.context_render.turns import Role, Turn, format_content

from workflows.main_cognition.csm_view import render_story
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts, StageOutputError
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import WorkspaceResult
from workflows.main_cognition.story_prefix import StoryPrefix, shown_shards

CONTROL_MODES = ("reactive", "reflective", "blended")


class CandidateText(BaseModel):
    """One record that the frame may focus on."""

    knoxel_id: int = 0
    text: str = ""


def candidate_texts(work: CycleWork) -> list[CandidateText]:
    """Return the items of the shown shards and the stimuli, one entry for each id."""
    items = [CandidateText(knoxel_id=item.record_id, text=work.records[item.record_id].content)
             for shard in shown_shards(work) for item in shard.items]
    items += [CandidateText(knoxel_id=stimulus.knoxel_id, text=stimulus.text) for stimulus in work.request.stimuli]
    return list({item.knoxel_id: item for item in items}.values())


def allowed_focus_ids(candidates: list[CandidateText]) -> set[int]:
    """Return the ids that the frame may focus on."""
    return {item.knoxel_id for item in candidates}


def render_candidates(candidates: list[CandidateText]) -> str:
    lines = [f"- id={item.knoxel_id} text={item.text!r}" for item in candidates]
    return "\n".join(lines) or "(none)"


def ensure_valid_result(result: WorkspaceResult, allowed: set[int]) -> None:
    """Reject focus ids outside ``allowed`` and unknown control modes."""
    for focus_id in result.focus_ids:
        if focus_id not in allowed:
            raise StageOutputError(
                f"Workspace focused on id {focus_id} that was not an allowed candidate."
            )
    if result.control_mode not in CONTROL_MODES:
        raise StageOutputError(
            f"Workspace chose control_mode {result.control_mode!r}; "
            f"expected one of {', '.join(CONTROL_MODES)}."
        )


def render_accepted(work: CycleWork) -> str:
    """Return the story lines of the items that entered a shard in this cycle, in world-time order."""
    entered = [item.record_id for change in work.attention.changes for item in change.entered]
    return render_story(work.records, entered, with_id=False) or "(nothing new)"


def build_messages(work: CycleWork, prompts: PhasePrompts, candidates: list[CandidateText],
                   allowed: set[int]) -> list[Turn]:
    """Return the shared story prefix and then the frame instruction as the last turn."""
    request = work.request
    system = prompts.system_prompt.format(companion_name=request.companion_name)
    user = format_content(prompts.user_template,
        companion_name=request.companion_name,
        user_name=request.user_name,
        affect=work.affect.felt,
        regulation=work.affect.regulation,
        active_goals=", ".join(request.active_goals) or "(none)",
        accepted=render_accepted(work),
        goals=render_context(work.goals),
        candidates=render_candidates(candidates),
    )
    instruction = f"{system.strip()}\n\n{user.text.strip()}\nAllowed focus ids: {json.dumps(sorted(allowed))}"
    return StoryPrefix(work).ask(instruction)


def run(context, work: CycleWork) -> CycleWork:
    candidates = candidate_texts(work)
    allowed = allowed_focus_ids(candidates)
    messages = build_messages(work, phase_prompts("experience"), candidates, allowed)
    # No invalid frame reaches commit. Offer one correction for a model that
    # selected an ID outside the workspace.
    for attempt in range(2):
        result = complete(context, work, messages, WorkspaceResult)
        try:
            ensure_valid_result(result, allowed)
            break
        except StageOutputError:
            if attempt == 1:
                raise
            messages.extend([
                Turn(Role.Assistant, result.model_dump_json()),
                Turn(Role.User, "This frame uses an invalid focus ID or control mode. "
                     f"Use only focus IDs from {sorted(allowed)} and control modes from {CONTROL_MODES}."),
            ])
    work.workspace = result
    work.receipt("experience", f"assembled a {result.control_mode} frame over {len(result.focus_ids)} focus ids")
    return work


__all__ = [
    "CONTROL_MODES",
    "CandidateText",
    "allowed_focus_ids",
    "build_messages",
    "candidate_texts",
    "ensure_valid_result",
    "render_accepted",
    "render_candidates",
    "run",
]
