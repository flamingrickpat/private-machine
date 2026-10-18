"""Appraise phase: judge how the stimuli land and project the affect step.

The model appraises the triaged situation. ``affect_dynamics`` then projects
one affect step from the previous mental state: elapsed-time decay,
personality gains, the appraisal axes, explicit deltas, and the expectation
error. Later phases read the same projection, and the commit stores it, so
the accepted state and the phase input cannot diverge. The phase sets the
mental state of the cycle state. It never writes memory.
"""
from __future__ import annotations

from pm.context_render.turns import Role, Turn, format_content

from workflows.main_cognition.affect_dynamics import AffectProjection, project_affect
from workflows.main_cognition.cycle_work import CycleWork
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import AffectView, AppraisalResult


def render_triage(work: CycleWork) -> str:
    """Render each triage item with its reasoning and the matching stimulus text."""
    stimuli = {item.knoxel_id: item.text for item in work.request.stimuli}
    lines = []
    for item in work.triage.items:
        lines.append(
            f"- id={item.knoxel_id} category={item.category} "
            f"input_kind={item.input_kind} "
            f"urgency={item.urgency} relevance={item.relevance} "
            f"text={stimuli.get(item.knoxel_id, '')!r} "
            f"reasoning={item.reasoning!r}"
        )
    return "\n".join(lines) or "(none)"


def build_messages(work: CycleWork, prompts: PhasePrompts) -> list[Turn]:
    """Build typed turns from the triage result and the current input."""
    request = work.request
    system = prompts.system_prompt.format(companion_name=request.companion_name)
    user = format_content(prompts.user_template,
        companion_name=request.companion_name,
        user_name=request.user_name,
        state_summary=work.state_summary or "(no recorded state update)",
        active_goals=", ".join(request.active_goals) or "(none)",
        triage=render_triage(work),
    )
    return [
        Turn(Role.System, system.strip()),
        Turn(Role.User, user.strip()),
    ]


def project(work: CycleWork, appraisal: AppraisalResult) -> AffectProjection:
    """Project the affect step of this appraisal from the previous state.

    The user is the relationship partner only when the appraisal cites a user
    message as evidence.
    """
    request = work.request
    user_ids = {item.knoxel_id for item in request.stimuli if item.source == "user_input"}
    partner = request.relationship_entity_id if user_ids.intersection(appraisal.evidence_ids) else None
    previous = request.previous_state
    return project_affect(
        previous.mental_state.to_list() if previous is not None else None,
        appraisal,
        elapsed_seconds=request.mental_state_age_seconds,
        relationship_entity_id=partner,
        personality=request.personality_vec or None,
        expectation_error=appraisal.expectation_error,
    )


def affect_view(projection: AffectProjection) -> AffectView:
    """Return the accepted affect in the form that later prompts read."""
    relationship = projection.state.state_relationship
    return AffectView(
        summary=projection.summary,
        regulation=projection.regulation,
        attention_focus=projection.behavior["attention_focus"],
        attention_bias={
            "valence_focus": round(float(projection.attention_bias[0]), 4),
            "salience": round(float(projection.attention_bias[1]), 4),
        },
        behavior=projection.behavior,
        emotions=projection.state.state_emotions.model_dump(),
        relationship=relationship.model_dump() if relationship is not None else None,
    )


def run(context, work: CycleWork) -> CycleWork:
    result = complete(context, work, build_messages(work, phase_prompts("appraise")), AppraisalResult,
                      grounded=True, emotion=True)
    projection = project(work, result)
    work.appraisal = result
    work.affect = projection
    work.affect_view = affect_view(projection)
    work.state_summary = projection.summary
    work.state.mental_state = projection.state
    work.receipt("appraise", f"appraised {len(result.evidence_ids)} evidence ids; feeling: {result.feeling}")
    return work


__all__ = ["affect_view", "build_messages", "project", "render_triage", "run"]
