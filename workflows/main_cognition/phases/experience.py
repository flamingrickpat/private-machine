"""Experience phase: frame the workspace and its subjective experience.

This phase ports the old workspace stage. The model reads the accepted
attention winners, the goal changes, and the candidate texts. It returns the
frame reason, the focus ids, the control mode, and the subjective
experience. Focus ids must come from the candidates and, when attention
selected ids, from that selection. An invalid frame gets one correction.

Step 7 of A19F replaces this phase with the story-framed subjective
experience in the self shard.
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field

from pm.context_render import render_context
from pm.context_render.turns import Role, Turn, format_content

from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts, StageOutputError
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import AttentionResult, WorkspaceResult

CONTROL_MODES = ("reactive", "reflective", "blended")

ACCEPTED_LABELS = {
    "observed": "happening now",
    "remembered": "memory",
    "internal": "my own feeling or thought",
    "inferred": "guess",
    "imagined": "possibility",
}

# A16: every proposal kind names its own voice in the frame. The workspace
# model sees which inner part each accepted note came from, without seeing
# machinery like scores or codelet names.
KIND_LABELS = {
    "attention": "noticed",
    "memory_access": "remembered",
    "appraisal": "interpretation",
    "simulation": "imagination",
    "regulation": "way to stay steady",
    "action_tendency": "want",
    "belief": "belief about myself",
}


class CandidateText(BaseModel):
    """One record that the frame may focus on."""

    knoxel_id: int = 0
    text: str = ""


class CompactNomination(BaseModel):
    """The fields of one nomination that the workspace prompt shows."""

    nomination_id: str = ""
    target_ids: list[int] = Field(default_factory=list)
    status: str = ""
    score: float = 0.0
    reality: str = ""
    attention_kind: str = ""


class CompactAttention(BaseModel):
    """A short attention summary: ids, status, kinds, and scores only."""

    selected_ids: list[int] = Field(default_factory=list)
    nominations: list[CompactNomination] = Field(default_factory=list)


def candidate_texts(work: CycleWork) -> list[CandidateText]:
    """Return memories, active goals, and stimuli, one entry for each id.

    Attention can select an active goal without retrieving it as a memory, so
    the goal records are candidates too. A later entry with the same id
    replaces the text of an earlier one but keeps its position.
    """
    request = work.request
    items = [CandidateText(knoxel_id=item.knoxel_id, text=item.text) for item in request.memory_candidates]
    seen = {item.knoxel_id for item in items}
    for goal in request.active_goal_records:
        if goal.knoxel_id not in seen:
            items.append(CandidateText(knoxel_id=goal.knoxel_id, text=goal.text))
            seen.add(goal.knoxel_id)
    for stimulus in request.stimuli:
        items.append(CandidateText(knoxel_id=stimulus.knoxel_id, text=stimulus.text))
    return list({item.knoxel_id: item for item in items}.values())


def allowed_focus_ids(candidates: list[CandidateText], attention: AttentionResult) -> set[int]:
    """Candidate ids, narrowed to the attention selection when one exists."""
    allowed = {item.knoxel_id for item in candidates}
    selected = set(attention.selected_ids)
    if selected:
        allowed &= selected
    return allowed


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


def render_accepted(attention: AttentionResult) -> str:
    """List the winning nominations with their reality and kind labels."""
    lines = []
    for item in attention.nominations:
        if item.status != "selected" or not item.text.strip():
            continue
        label = ACCEPTED_LABELS[item.reality] if item.reality in ACCEPTED_LABELS else item.reality
        kind = KIND_LABELS[item.attention_kind] if item.attention_kind in KIND_LABELS else ""
        prefix = f"{label}; {kind}" if kind else label
        lines.append(f"- ({prefix}) {item.text}")
    return "\n".join(lines) or "(none)"


def compact_attention(attention: AttentionResult) -> CompactAttention:
    """Keep the workspace prompt short: ids, status, kinds, and scores only."""
    return CompactAttention(
        selected_ids=list(attention.selected_ids),
        nominations=[CompactNomination(
            nomination_id=item.nomination_id, target_ids=list(item.target_ids), status=item.status,
            score=item.score, reality=item.reality, attention_kind=item.attention_kind)
            for item in attention.nominations],
    )


def build_messages(work: CycleWork, prompts: PhasePrompts, candidates: list[CandidateText],
                   allowed: set[int]) -> list[Turn]:
    request = work.request
    system = prompts.system_prompt.format(companion_name=request.companion_name)
    user = format_content(prompts.user_template,
        companion_name=request.companion_name,
        user_name=request.user_name,
        affect=work.affect.felt,
        regulation=work.affect.regulation,
        active_goals=", ".join(request.active_goals) or "(none)",
        accepted=render_accepted(work.attention),
        attention=render_context(compact_attention(work.attention)),
        goals=render_context(work.goals),
        candidates=render_candidates(candidates),
    )
    user += "\nAllowed focus ids: " + json.dumps(sorted(allowed))
    return [
        Turn(Role.System, system.strip()),
        Turn(Role.User, user.strip()),
    ]


def run(context, work: CycleWork) -> CycleWork:
    candidates = candidate_texts(work)
    allowed = allowed_focus_ids(candidates, work.attention)
    messages = build_messages(work, phase_prompts("experience"), candidates, allowed)
    # No invalid frame reaches commit. Offer one correction for a model that
    # selected an ID outside the accepted attention boundary.
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
    "ACCEPTED_LABELS",
    "CONTROL_MODES",
    "KIND_LABELS",
    "CandidateText",
    "allowed_focus_ids",
    "build_messages",
    "candidate_texts",
    "compact_attention",
    "ensure_valid_result",
    "render_accepted",
    "render_candidates",
    "run",
]
