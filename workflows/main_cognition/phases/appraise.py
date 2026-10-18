"""Appraise phase: judge how the stimuli land and update the mental state.

The model appraises the triaged situation. Then the phase computes the
affect of the cycle (D27):

1. The base state is the raw latent state: the field defaults plus the sum
   of the deltas of the causal features. Its relationship sums the deltas
   of the partner entity only.
2. :func:`~workflows.main_cognition.affect.appraise_state` derives the
   appraisal state and its delta from the appraisal axes, with the partner
   of the primary stimulus group as relationship entity.
3. The new appraisal sample joins the history. The EMA readout over the
   whole history, up to the cycle time, becomes ``CycleState.mental_state``.
   So the time since the last feature lets earlier feelings fade.

Ego strength of the previous state sets the EMA half-life
(:func:`~pm.records.mental_state.reference_half_life_seconds`). The commit
stores the appraisal state and the delta on the appraisal ``Feature``, so
the next cycle reads the same sample from the store. The phase never writes
memory.
"""
from __future__ import annotations

from pm.context_render.turns import Role, Turn, format_content
from pm.records.mental_state import (
    MentalStateSample,
    compute_latent_state,
    compute_mental_state,
    compute_relationship_state,
    derive_expression,
    reference_half_life_seconds,
    sparse_delta,
)

from workflows.main_cognition.affect import appraise_state, describe_mental_state, regulation_mode
from workflows.main_cognition.cycle_work import CycleWork, primary_group
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import AppraisalAffect, AppraisalResult


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


def partner_entity(work: CycleWork) -> int | None:
    """Return the entity of the primary stimulus group, or None for a self or workflow group."""
    group = primary_group(work)
    if group is None or group.key.kind != "entity":
        return None
    return int(group.key.ref)


def update_affect(work: CycleWork, appraisal: AppraisalResult) -> AppraisalAffect:
    """Compute the appraisal affect and set the EMA mental state of the cycle.

    With ``H`` the causal feature history, ``e`` the partner entity, and
    ``t`` the cycle time:

    1. ``half_life = MENTAL_STATE_REFERENCE_MINUTES * 60 * |1 - ego_strength|``
       seconds, with the ego strength of the previous state.
    2. ``base = defaults + sum(H)`` for core affect, emotions, and needs. Its
       relationship is ``sum(H_e)`` over the samples of ``e`` only
       (:func:`~pm.records.mental_state.compute_latent_state`).
    3. ``appraisal_state, delta = appraise_state(base, answer, e)``. The
       delta is ``appraisal_state - base``. So the new raw level equals the
       appraisal state.
    4. ``state = EMA(H + [(t, delta)], t)`` with the relationship from
       ``EMA(H_e + [(t, delta)], t)``. The expression readouts follow from
       ``state``.

    ``EMA`` is :func:`~pm.records.mental_state.compute_mental_state`: the
    deviation of the raw level from its time-aware EMA baseline, scaled by
    the spread of the past deviations. The function changes
    ``work.state.mental_state`` and returns the data that the appraisal
    ``Feature`` stores.
    """
    request = work.request
    previous = request.previous_state
    half_life = reference_half_life_seconds(
        previous.mental_state.state_cognition.ego_strength if previous is not None else None)
    now = request.cycle_time
    history = request.state_history
    entity = partner_entity(work)

    base = compute_latent_state(history, personality=request.personality, entity_id=entity)
    appraisal_state, delta = appraise_state(base, appraisal, entity_id=entity)

    sample = MentalStateSample(timestamp=now, entity_ids=[entity] if entity is not None else [],
                               delta=sparse_delta(delta))
    updated = [*history, sample]
    state = compute_mental_state(updated, now=now, half_life_s=half_life, personality=request.personality)
    if entity is not None:
        state.state_relationship = compute_relationship_state(updated, entity, now=now, half_life_s=half_life)
    derive_expression(state)
    work.state.mental_state = state
    partner_name = request.user_name if entity is not None and entity == request.user_entity_id else ""
    return AppraisalAffect(
        appraisal_state=appraisal_state, delta_vec=delta, entity_id=entity,
        regulation=regulation_mode(appraisal_state, state),
        felt=describe_mental_state(state, name=request.companion_name, partner_name=partner_name),
    )


def run(context, work: CycleWork) -> CycleWork:
    result = complete(context, work, build_messages(work, phase_prompts("appraise")), AppraisalResult,
                      grounded=True, emotion=True)
    work.appraisal = result
    work.affect = update_affect(work, result)
    work.state_summary = work.affect.felt
    work.receipt("appraise", f"appraised {len(result.evidence_ids)} evidence ids; feeling: {result.feeling}")
    return work


__all__ = ["build_messages", "partner_entity", "render_triage", "run", "update_affect"]
