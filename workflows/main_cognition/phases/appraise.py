"""Appraise phase: check expectations, judge how the stimuli land, and update the mental state.

Both model calls are story-framed (A19F step 7): they start with the shared
story prefix and end with their instruction.

1. Expectation check (port of ``_appraise_stimulus_check_expectation``).
   The open expectations that the new stimuli can affect are rated for
   directness and fulfillment (:mod:`workflows.main_cognition.expectations`).
   Their valence impacts become the ``expectation_error`` of the appraisal,
   and their fear impacts join its fear delta. The commit stores the new
   fulfillment and status of each checked expectation.
2. Appraisal. The model appraises the triaged situation and reads the
   expectation outcomes.

Then the phase computes the affect of the cycle (D27):

1. The base state is the raw latent state: the field defaults plus the sum
   of the deltas of the causal features. Its relationship sums the deltas
   of the partner entity only.
2. :func:`~workflows.main_cognition.affect.appraise_state` derives the
   appraisal state and its delta from the appraisal axes, with the partner
   of the primary stimulus group as relationship entity.
3. The new appraisal sample joins the history. The EMA readout over the
   whole history, up to the cycle time, becomes ``CycleState.mental_state``.
   So the time since the last feature lets earlier feelings fade.

:func:`~pm.records.mental_state.reference_half_life_seconds` sets the EMA
half-life. The commit
stores the appraisal state and the delta on the appraisal ``Feature``, so
the next cycle reads the same sample from the store. The phase never writes
memory.
"""
from __future__ import annotations

from pm_workflows.python_role import RoleContext

from pm.context_render.turns import Turn, format_content
from pm.records.mental_state import (
    MentalStateSample,
    alpha_clamp_delta,
    alpha_limit_state_step,
    compute_mental_state,
    compute_relationship_state,
    derive_expression,
    reference_half_life_seconds,
    sparse_delta,
)

from workflows.main_cognition import expectations
from workflows.main_cognition.affect import appraise_state, describe_mental_state, regulation_mode
from workflows.main_cognition.csm_view import Embedder
from workflows.main_cognition.cycle_work import CycleWork, primary_group
from workflows.main_cognition.model_json import PhasePrompts
from workflows.main_cognition.phases.model import complete, phase_prompts
from workflows.main_cognition.schemas import AppraisalAffect, AppraisalResult, ExpectationOutcome
from workflows.main_cognition.story_prefix import StoryPrefix


def render_triage(work: CycleWork) -> str:
    """Render each triage item with its reasoning and the matching stimulus text."""
    stimuli = {item.knoxel_id: item.text for item in work.request.stimuli}
    lines = []
    for item in work.triage.items:
        lines.append(
            f"- id={item.knoxel_id} category={item.category} "
            f"input_kind={item.input_kind} "
            f"urgency={item.urgency} relevance={item.relevance} "
            f"text={stimuli[item.knoxel_id] if item.knoxel_id in stimuli else ''!r} "
            f"reasoning={item.reasoning!r}"
        )
    return "\n".join(lines) or "(none)"


def build_messages(work: CycleWork, prompts: PhasePrompts, prefix: StoryPrefix) -> list[Turn]:
    """Return the shared story prefix and then the appraisal instruction as the last turn."""
    request = work.request
    user = format_content(prefix.fill(prompts.user_template),
        state_summary=work.state_summary or "(no feeling recorded yet)",
        active_goals="; ".join(request.active_goals) or "(nothing in particular)",
        triage=render_triage(work),
        expectations=expectations.render_outcomes(work.expectations),
    )
    return prefix.ask(user.text)


def merge_expectations(result: AppraisalResult, outcomes: list[ExpectationOutcome]) -> AppraisalResult:
    """Return the appraisal with the measured effect of the expectation outcomes.

    Without outcomes the model's own ``expectation_error`` stays. With
    outcomes, the summed valence impact replaces it, and the summed fear
    impact adds to the fear delta. The values stay within their bounds.
    """
    if not outcomes:
        return result
    error, fear = expectations.effect(outcomes)
    deltas = dict(result.emotion_delta)
    deltas["fear"] = max(-1.0, min(1.0, (deltas["fear"] if "fear" in deltas else 0.0) + fear))
    return result.model_copy(update={"expectation_error": error, "emotion_delta": deltas})


def partner_entity(work: CycleWork) -> int | None:
    """Return the entity of the primary stimulus group, or None for a self or workflow group."""
    group = primary_group(work)
    if group is None or group.key.kind != "entity":
        return None
    return int(group.key.ref)


def update_affect(work: CycleWork, appraisal: AppraisalResult) -> AppraisalAffect:
    """Compute the appraisal affect and set the EMA mental state of the cycle.

    This follows the old ``ghost_codelets`` design. With ``H`` the causal
    feature history, ``e`` the partner entity, and ``t`` the cycle time:

    1. ``current = EMA(H, t)``: the readout of the state before this
       appraisal, with the relationship toward ``e`` from ``EMA(H_e, t)``.
    2. ``appraisal_state, delta = appraise_state(current, answer, e)``. The
       delta is the step from the current readout to the appraised state.
    3. ``state = EMA(H + [(t, delta)], t)``, again with the relationship of
       ``e``. The expression readouts follow from ``state``.

    ``EMA`` is :func:`~pm.records.mental_state.compute_mental_state`. Its raw
    level is the running sum of all deltas. That sum has no meaning by itself
    and can grow without limit. The EMA baseline follows the level, and only
    the deviation of the level from the baseline is read out. So a repeated
    appraisal moves the readout toward its target, and habituation brings it
    back toward the baseline over time.

    The function changes ``work.state.mental_state`` and returns the data
    that the appraisal ``Feature`` stores.
    """
    request = work.request
    half_life = reference_half_life_seconds()
    now = request.cycle_time
    history = request.state_history
    entity = partner_entity(work)

    current = compute_mental_state(history, now=now, half_life_s=half_life, personality=request.personality)
    if entity is not None:
        current.state_relationship = compute_relationship_state(history, entity, now=now, half_life_s=half_life)
    appraisal_state, delta = appraise_state(current, appraisal, entity_id=entity)
    # !!! ALPHA RUN ONLY: TEMPORARY FAILSAFE. FIX PROPERLY LATER. !!!
    # One stimulus moves one axis by a third at most. The commit stores the
    # clamped delta, so the next cycle reads the same sample.
    delta = alpha_clamp_delta(delta)

    sample = MentalStateSample(timestamp=now, entity_ids=[entity] if entity is not None else [],
                               delta=sparse_delta(delta))
    updated = [*history, sample]
    state = compute_mental_state(updated, now=now, half_life_s=half_life, personality=request.personality)
    if entity is not None:
        state.state_relationship = compute_relationship_state(updated, entity, now=now, half_life_s=half_life)
    # !!! ALPHA RUN ONLY: TEMPORARY FAILSAFE. FIX PROPERLY LATER. !!!
    # Each axis moves a third at most from the state of the last cycle. In
    # the A19F review run one stimulus moved valence from +0.78 to -1.0.
    if request.previous_state is not None:
        alpha_limit_state_step(request.previous_state.mental_state, state)
    derive_expression(state)
    work.state.mental_state = state
    partner_name = request.user_name if entity is not None and entity == request.user_entity_id else ""
    return AppraisalAffect(
        appraisal_state=appraisal_state, delta_vec=delta, entity_id=entity,
        regulation=regulation_mode(appraisal_state, state),
        felt=describe_mental_state(state, name=request.companion_name, partner_name=partner_name),
    )


def run(context: RoleContext, work: CycleWork) -> CycleWork:
    prompts = phase_prompts("appraise")
    prefix = StoryPrefix(work)
    relevant = expectations.relevant_expectations(work, Embedder(context, work.records))
    work.expectations = expectations.check(context, work, prefix, prompts, relevant)
    result = complete(context, work, build_messages(work, prompts, prefix), AppraisalResult,
                      grounded=True, emotion=True, grounding_at=len(prefix.turns))
    result = merge_expectations(result, work.expectations)
    work.appraisal = result
    work.affect = update_affect(work, result)
    work.state_summary = work.affect.felt
    work.receipt("appraise", f"checked {len(work.expectations)} expectations; appraised "
                             f"{len(result.evidence_ids)} evidence ids; feeling: {result.feeling}")
    return work


__all__ = ["build_messages", "merge_expectations", "partner_entity", "render_triage", "run", "update_affect"]
