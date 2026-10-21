"""Affect of the main cycle: appraisal state, regulation mode, and feelings in words.

D27 sets the affect model. Appraisal produces a ``Feature`` with an appraisal
state and a delta. The mental state of the cycle is the EMA-baselined readout
over the deltas of the causal features
(:func:`pm.records.mental_state.compute_mental_state`). This module holds the
parts of that flow that are behavior, not runtime math:

- :func:`appraise_state` turns one appraisal answer into the appraisal state
  and its delta,
- :func:`percept_affect` mini-appraises one codelet percept the same way,
- :func:`regulation_mode` maps the appraisal and the accepted state to a
  regulation mode,
- :func:`describe_mental_state` renders the accepted state as short story
  text for prompts: the strongest emotions, the mood, the relationship, the
  strongest urge to act, where attention drifts, and the pressing needs.

The functions are pure. They never call a model or touch the memory store.
"""
from __future__ import annotations

from pm.records.copying import pickle_copy

from typing import TYPE_CHECKING

from pm.constants import (
    ANGER_REACTIVE_THRESHOLD,
    ATTENTION_DRIFT_THRESHOLD,
    AVERSIVE_REACTIVE_THRESHOLD,
    BEHAVIOR_PRESSING,
    EXPECTATION_EMOTION_GAIN,
    EXPECTATION_VALENCE_GAIN,
    FEAR_REACTIVE_THRESHOLD,
    FEELING_HINT,
    FEELING_MAX_EMOTIONS,
    FEELING_MAX_NEEDS,
    FEELING_MAX_RELATIONSHIP,
    FEELING_MILD,
    FEELING_MODERATE,
    FEELING_STRONG,
    MOOD_AROUSAL_ALERT,
    MOOD_AROUSAL_KEYED,
    MOOD_DOMINANCE_THRESHOLD,
    MOOD_VALENCE_THRESHOLD,
    NEED_PRESSING,
    SADNESS_REFLECTIVE_THRESHOLD,
    THREAT_APPRAISAL_THRESHOLD,
    WITHDRAWAL_REFLECTIVE_THRESHOLD,
)
from pm.records.mental_state import (
    AppraisalGeneral,
    AppraisalSocial,
    FullMentalState,
    StateEmotions,
    StateExpressionExt,
    StateNeeds,
    compute_attention_bias,
    compute_state_delta,
    ema_axis_positions,
)

if TYPE_CHECKING:
    # The schemas import the public context, which imports this module.
    from workflows.main_cognition.schemas import AppraisalResult

REGULATION_REACTIVE = "reactive"
REGULATION_REFLECTIVE = "reflective"
REGULATION_BLENDED = "blended"


def _bounded(container, name: str, value: float) -> float:
    """Clamp ``value`` into the declared bounds of one field of ``container``."""
    model = type(container)
    low, high = model.get_axis_bounds(model.model_fields[name].json_schema_extra["vector_position"])
    return min(high, max(low, value))


def appraise_state(base: FullMentalState, appraisal: AppraisalResult, *, entity_id: int | None) -> tuple[FullMentalState, list[float]]:
    """Return the appraisal state and its delta vector for one appraisal answer.

    ``base`` is the EMA readout at the start of the cycle
    (:func:`~pm.records.mental_state.compute_mental_state`), with the
    relationship toward ``entity_id`` when there is one.

    Steps:

    1. Copy ``base`` and set the appraisal axes of the answer. An omitted
       axis has its field default.
    2. ``compute_state_delta`` derives core affect, emotions, expression,
       and (with ``entity_id``) the relationship from the axes.
    3. The explicit ``emotion_delta`` and ``need_delta`` add to the derived
       values. The expectation error moves valence, and joy or sadness.
       Every value stays inside its field bounds.
    4. The delta is the result minus the copy of step 1, on the axes that
       the EMA sums (``EMA_STATE_MODELS``). Other positions are 0. So the
       appraisal axes, cognition, and the relationship entity id never add
       up in the history.
    """
    probe = pickle_copy(base)
    probe.appraisal_general = appraisal.general()
    probe.appraisal_social = appraisal.social()
    derived, _ = compute_state_delta(probe, relationship_entity_id=entity_id)
    emotions = derived.state_emotions
    for name, value in appraisal.emotion_delta.items():
        setattr(emotions, name, _bounded(emotions, name, getattr(emotions, name) + value))
    needs = derived.state_needs
    for name, value in appraisal.need_delta.items():
        setattr(needs, name, _bounded(needs, name, getattr(needs, name) + value))
    error = appraisal.expectation_error
    if error:
        core = derived.state_core
        core.valence = _bounded(core, "valence", core.valence + EXPECTATION_VALENCE_GAIN * error)
        target = "joy" if error > 0 else "sadness"
        setattr(emotions, target, _bounded(emotions, target, getattr(emotions, target)
                                           + EXPECTATION_EMOTION_GAIN * abs(error)))
    return derived, ema_delta(derived, probe)


def ema_delta(after: FullMentalState, before: FullMentalState) -> list[float]:
    """Return ``after - before`` on the axes that the EMA sums. Other positions are 0."""
    axes = ema_axis_positions()
    return [new - old if pos in axes else 0.0
            for pos, (new, old) in enumerate(zip(after.to_list(), before.to_list()))]


def percept_affect(base: FullMentalState, general: AppraisalGeneral, social: AppraisalSocial, *,
                   entity_id: int | None) -> tuple[FullMentalState, list[float]]:
    """Mini-appraise one codelet percept. Return its appraisal state and delta.

    Port of the percept loop of ``ghost_codelets.cognitive_cycle``: copy the
    current state, set the appraisal axes of the percept, and run
    ``compute_state_delta`` with the group entity. ``base`` stays unchanged.
    The delta is the result minus the copy, on the EMA axes only. A percept
    does not change the mental state. Its delta can enter the EMA history
    after attention puts the percept into a workspace shard. Then the commit
    stages it as causal.
    """
    probe = pickle_copy(base)
    probe.appraisal_general = general.model_copy()
    probe.appraisal_social = social.model_copy()
    derived, _ = compute_state_delta(probe, relationship_entity_id=entity_id)
    return derived, ema_delta(derived, probe)


def regulation_mode(appraisal_state: FullMentalState, state: FullMentalState) -> str:
    """Return the regulation mode of the cycle.

    Danger preempts slow regulation: a high appraised bodily threat or
    aversive pressure, or strong fear or anger in the accepted state, makes
    the mode ``reactive``. Strong sadness, or a strong pull to go silent,
    makes it ``reflective``. Otherwise the mode is ``blended``.

    The threat axes come from ``appraisal_state``, because the EMA state
    does not sum appraisal axes. Fear, anger, sadness, and the pull to go
    silent come from the accepted ``state``. Its expression readouts must
    be set (:func:`~pm.records.mental_state.derive_expression`).
    """
    axes = appraisal_state.appraisal_general
    emotions = state.state_emotions
    if (axes.bodily_threat >= THREAT_APPRAISAL_THRESHOLD or axes.aversive_pressure >= AVERSIVE_REACTIVE_THRESHOLD
            or emotions.fear >= FEAR_REACTIVE_THRESHOLD or emotions.anger >= ANGER_REACTIVE_THRESHOLD):
        return REGULATION_REACTIVE
    if (emotions.sadness >= SADNESS_REFLECTIVE_THRESHOLD
            or state.state_expression_ext.disengage_silence >= WITHDRAWAL_REFLECTIVE_THRESHOLD):
        return REGULATION_REFLECTIVE
    return REGULATION_BLENDED


# Words for each emotion. Each word reads well after an intensity word, for
# example "mild teasing urges". ``propriety_inhibition`` is a brake, not a
# feeling, so it has no word.
EMOTION_WORDS = {
    "joy": "joy", "sadness": "sadness", "anger": "anger", "fear": "fear", "disgust": "disgust",
    "surprise": "surprise", "resentment": "resentment", "jealousy": "jealousy", "boredom": "boredom",
    "tenderness": "tenderness", "curiosity": "curiosity", "shame": "shame", "guilt": "guilt",
    "pride": "pride", "intimacy_arousal": "intimate excitement", "amusement": "amusement",
    "playfulness": "playfulness", "relief": "relief", "social_laughter": "urges to laugh along",
    "teasing_edge": "teasing urges",
}

# What each need asks for when it is short. ``closeness_need`` is a drive:
# a high value is the need. The other needs are supplies: a low value is the need.
NEED_WORDS = {
    "energy_stability": "steady energy", "processing_power": "room to think", "data_access": "information",
    "connection": "connection", "closeness_need": "closeness", "relevance": "feeling useful",
    "learning_growth": "something new to learn", "creative_expression": "creative expression",
    "autonomy": "autonomy",
}
# What each expression tendency urges, as the end of "<name> wants to ...".
# ``display_amplitude`` and the primitive pressures are not urges toward a
# person, so they have no words.
BEHAVIOR_WORDS = {
    "assertive_push": "hold a firm line",
    "confrontive_push": "push back hard",
    "affiliative_soothe": "smooth things over",
    "appeasement": "give way and placate",
    "disengage_silence": "go quiet and pull back",
}


def _intensity(value: float) -> str:
    """Return an intensity phrase for a value in [0, 1], or an empty text below a hint.

    The thresholds come from the old ``_verbalize_emotional_state``.
    """
    if value >= FEELING_STRONG:
        return "strong"
    if value >= FEELING_MODERATE:
        return "moderate"
    if value >= FEELING_MILD:
        return "mild"
    if value >= FEELING_HINT:
        return "a hint of"
    return ""


def _phrase(value: float, word: str) -> str:
    """Return the intensity phrase and the word, for example "mild boredom"."""
    return f"{_intensity(value)} {word}"


def _emotion_sentence(name: str, emotions: StateEmotions) -> str:
    """Name the strongest emotions, or say that the companion is calm."""
    ranked = sorted(((getattr(emotions, field), word) for field, word in EMOTION_WORDS.items()), reverse=True)
    active = [(value, word) for value, word in ranked[:FEELING_MAX_EMOTIONS] if _intensity(value)]
    if not active:
        return f"{name} feels calm and emotionally even."
    first, *rest = [_phrase(value, word) for value, word in active]
    if not rest:
        return f"{name} feels {first}."
    return f"{name} feels {first}, along with {' and '.join(rest)}."


def _mood_sentence(name: str, state: FullMentalState) -> str:
    """Describe core affect: valence as spirits, arousal, and dominance as footing."""
    core = state.state_core
    parts = []
    if core.valence >= MOOD_VALENCE_THRESHOLD:
        parts.append("in good spirits")
    elif core.valence <= -MOOD_VALENCE_THRESHOLD:
        parts.append("in low spirits")
    if core.arousal >= MOOD_AROUSAL_KEYED:
        parts.append("keyed up")
    elif core.arousal >= MOOD_AROUSAL_ALERT:
        parts.append("alert")
    if core.dominance >= MOOD_DOMINANCE_THRESHOLD:
        parts.append("sure of their footing")
    elif core.dominance <= -MOOD_DOMINANCE_THRESHOLD:
        parts.append("out of their depth")
    if not parts:
        return ""
    return f"Underneath, {name} is {', '.join(parts)}."


def _relationship_sentence(state: FullMentalState, partner_name: str) -> str:
    """Name the strongest relationship axes toward the partner. A negative axis uses its opposite word."""
    relationship = state.state_relationship
    if relationship is None or not partner_name:
        return ""
    signed = [
        (relationship.affection, "affection", "dislike"),
        (relationship.trust, "trust", "distrust"),
        (relationship.communication_warmth, "warmth", "coldness"),
        (relationship.conflict_tension, "friction", "friction"),
        (relationship.attachment_strength, "attachment", "attachment"),
        (relationship.aversive_association, "aversion", "aversion"),
    ]
    ranked = sorted(((abs(value), positive if value >= 0 else negative) for value, positive, negative in signed),
                    reverse=True)
    active = [_phrase(value, word) for value, word in ranked[:FEELING_MAX_RELATIONSHIP] if value >= FEELING_MILD]
    if not active:
        return ""
    return f"Toward {partner_name} there is {' and '.join(active)}."


def _behavior_sentence(name: str, expression: StateExpressionExt) -> str:
    """Name the strongest urge to act, when one is pressing."""
    value, words = max((getattr(expression, field), words) for field, words in BEHAVIOR_WORDS.items())
    if value < BEHAVIOR_PRESSING:
        return ""
    return f"{name} wants to {words}."


def _attention_sentence(name: str, state: FullMentalState) -> str:
    """Say where attention drifts, from the valence focus of ``compute_attention_bias``."""
    valence_focus, _ = compute_attention_bias(state)
    if valence_focus <= -ATTENTION_DRIFT_THRESHOLD:
        return f"{name}'s attention drifts toward threat and loss."
    if valence_focus >= ATTENTION_DRIFT_THRESHOLD:
        return f"{name}'s attention drifts toward reward and connection."
    return ""


def _need_sentence(name: str, needs: StateNeeds) -> str:
    """Name the most pressing needs."""
    pressure = []
    for field, word in NEED_WORDS.items():
        value = getattr(needs, field)
        short = value - (1.0 - NEED_PRESSING) if field == "closeness_need" else NEED_PRESSING - value
        if short > 0:
            pressure.append((short, word))
    pressure.sort(reverse=True)
    if not pressure:
        return ""
    return f"{name} longs for {' and '.join(word for _, word in pressure[:FEELING_MAX_NEEDS])}."


def describe_mental_state(state: FullMentalState, *, name: str, partner_name: str = "") -> str:
    """Return the accepted mental state as short story text for prompts.

    The text names the strongest emotions, the mood, the relationship toward
    ``partner_name`` (when the state has one), the strongest urge to act,
    where attention drifts, and the most pressing needs. It uses words, not
    numbers. ``name`` is the companion's name. The expression readouts of
    ``state`` must be set (:func:`~pm.records.mental_state.derive_expression`).
    """
    parts = [
        _emotion_sentence(name, state.state_emotions),
        _mood_sentence(name, state),
        _relationship_sentence(state, partner_name),
        _behavior_sentence(name, state.state_expression_ext),
        _attention_sentence(name, state),
        _need_sentence(name, state.state_needs),
    ]
    return " ".join(part for part in parts if part)


__all__ = [
    "BEHAVIOR_WORDS",
    "EMOTION_WORDS",
    "NEED_WORDS",
    "REGULATION_BLENDED",
    "REGULATION_REACTIVE",
    "REGULATION_REFLECTIVE",
    "appraise_state",
    "describe_mental_state",
    "ema_delta",
    "percept_affect",
    "regulation_mode",
]
