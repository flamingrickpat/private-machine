"""Bounded affect dynamics for the companion.

This module is pure mathematics. It connects four inputs to one new latent
mental state:

- the previous accepted ``FullMentalState``;
- the elapsed time since that state was accepted;
- an accepted appraisal (canonical axes, or explicit emotion and need deltas);
- the stable personality profile.

The dynamics never call a model, touch the Ghost, or read trace files. The
caller owns persistence and commit. The output carries a trace that names every
input, the decay factor, the personality gains, and the expectation error.

Design rules:

- Every value stays inside its declared field bounds. A final clamp runs after
  all arithmetic, so no caller can store an out-of-range vector.
- Decay uses an exponential half-life per group. A mood returns toward its
  baseline while nothing new happens. Repeated events can accumulate before
  decay removes them. Two emotions can coexist.
- Personality changes gain, not the underlying facts. A trait does not invent
  an event.
- A service failure is not a neutral emotion. The caller supplies the failure
  as evidence; this module only maps the accepted appraisal to state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pm_core.model.mental_state_vectors import (
    FullMentalState,
    PersonalityProfile,
    StateCore,
    StateEmotions,
    StateNeeds,
    compute_attention_bias,
    compute_state_delta,
    create_empty_state,
    parse_personality_profile,
)

# Elapsed-time half-lives in seconds. Fast emotions fade within an hour; core
# mood and needs hold longer so a short conversation does not erase a mood.
EMOTION_HALF_LIFE_SECONDS = 2700.0
CORE_HALF_LIFE_SECONDS = 5400.0
NEED_HALF_LIFE_SECONDS = 21600.0

# One EMA gain for the appraisal-derived delta at full personality weight.
BASE_APPRAISAL_GAIN = 0.45
# How strongly an expectation error moves core affect and joy or sadness.
EXPECTATION_VALENCE_GAIN = 0.55
EXPECTATION_EMOTION_GAIN = 0.35
# Magnitude above which an expectation error starts to reduce perceived certainty.
EXPECTATION_SURPRISE_GAIN = 0.40

# Behavior-mapping thresholds. These names have defined effects, not labels.
THREAT_FEAR_THRESHOLD = 0.55
ANGER_REACTIVE_THRESHOLD = 0.60
SADNESS_REFLECTIVE_THRESHOLD = 0.50
WITHDRAWAL_REFLECTIVE_THRESHOLD = 0.55

REGULATION_REACTIVE = "reactive"
REGULATION_REFLECTIVE = "reflective"
REGULATION_BLENDED = "blended"


@dataclass(frozen=True)
class AffectProjection:
    """One projected affect step and its causal trace."""

    state: FullMentalState
    delta: FullMentalState
    trace: dict
    summary: str
    attention_bias: tuple[float, float]
    regulation: str
    behavior: dict = field(default_factory=dict)


def _field_bounds(field_info: Any) -> tuple[float | None, float | None]:
    ge: float | None = None
    le: float | None = None
    for meta in getattr(field_info, "metadata", []):
        if hasattr(meta, "ge"):
            ge = meta.ge
        if hasattr(meta, "le"):
            le = meta.le
    return ge, le


def _clamp_value(value: float, field_info: Any) -> float:
    ge, le = _field_bounds(field_info)
    if ge is not None:
        value = max(float(ge), value)
    if le is not None:
        value = min(float(le), value)
    return value


def _clamp_state(state: FullMentalState) -> FullMentalState:
    """Clamp every declared field back inside its bounds.

    ``init_from_list`` writes directly into ``__dict__`` and skips validators,
    so this guard is required after any direct arithmetic.
    """
    containers = [
        state.appraisal_general,
        state.appraisal_social,
        state.state_neurochemical,
        state.state_core,
        state.state_emotions,
        state.state_cognition,
        state.state_needs,
        state.state_expression,
        state.state_expression_ext,
        state.state_relationship,
    ]
    for container in containers:
        if container is None:
            continue
        for name, field_info in type(container).model_fields.items():
            value = getattr(container, name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                setattr(container, name, _clamp_value(float(value), field_info))
    return state


def _baseline(container) -> dict[str, float]:
    """Return the model default for each numeric field as the decay baseline."""
    baseline: dict[str, float] = {}
    for name, field_info in type(container).model_fields.items():
        default = field_info.default
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            baseline[name] = float(default)
    return baseline


def _decay_group(container, elapsed_seconds: float, half_life: float) -> None:
    if elapsed_seconds <= 0.0 or half_life <= 0.0:
        return
    factor = 0.5 ** (float(elapsed_seconds) / float(half_life))
    for name, base in _baseline(container).items():
        current = float(getattr(container, name))
        setattr(container, name, base + (current - base) * factor)


def decay_toward_baseline(
    state: FullMentalState,
    elapsed_seconds: float,
    *,
    personality: PersonalityProfile | None = None,
) -> FullMentalState:
    """Return a copy of ``state`` with exponential decay applied.

    Emotions decay fastest, core affect more slowly, and needs slowest.
    Neuroticism slows the decay of negative emotions; conscientiousness and
    self-control steady the mood. Decay is a bounded pull toward the model
    baseline, so an event-free interval returns the companion toward rest
    without a cheerful reset.
    """
    result = state.model_copy(deep=True)
    if elapsed_seconds <= 0.0:
        return result
    profile = personality or state.state_personality
    neuroticism = float(profile.neuroticism)
    stability = 1.0 + 0.5 * float(profile.self_control) - 0.4 * neuroticism

    _decay_group(result.state_emotions, elapsed_seconds, EMOTION_HALF_LIFE_SECONDS * stability)
    _decay_group(result.state_core, elapsed_seconds, CORE_HALF_LIFE_SECONDS * (0.8 + 0.4 * stability))
    _decay_group(result.state_needs, elapsed_seconds, NEED_HALF_LIFE_SECONDS)
    _decay_group(result.appraisal_general, elapsed_seconds, CORE_HALF_LIFE_SECONDS)
    _decay_group(result.appraisal_social, elapsed_seconds, CORE_HALF_LIFE_SECONDS)
    _decay_group(result.state_cognition, elapsed_seconds, EMOTION_HALF_LIFE_SECONDS)
    return result


def personality_gains(personality: PersonalityProfile) -> dict[str, float]:
    """Return bounded per-field gain multipliers from the stable traits.

    Each value stays in [0.5, 1.5]. The multipliers shape how strongly an
    appraisal moves a field. The traits do not add energy on their own.
    """
    profile = personality
    gains: dict[str, float] = {}

    def put(name: str, value: float) -> None:
        gains[name] = max(0.5, min(1.5, float(value)))

    # Positive emotions follow extraversion, openness, and the approach drive.
    positive = 1.0 + 0.25 * (profile.extraversion - 0.5) + 0.15 * (profile.openness - 0.5) + 0.20 * (profile.bas_drive - 0.5)
    put("joy", positive)
    put("pride", positive)
    put("curiosity", positive)
    put("playfulness", positive)
    put("amusement", positive)

    # Negative emotions follow neuroticism and the inhibition system, and fall
    # with self-control.
    negative = 1.0 + 0.70 * (profile.neuroticism - 0.5) + 0.30 * (profile.bis_sensitivity - 0.5) - 0.30 * (profile.self_control - 0.5)
    put("sadness", negative)
    put("fear", negative)
    put("shame", negative)
    put("guilt", negative)
    put("disgust", negative)

    # Anger follows bluntness and falls with agreeableness, warmth, and control.
    anger = 1.0 + 0.30 * (profile.bluntness_assertive - 0.5) - 0.30 * (profile.agreeableness - 0.5) - 0.15 * (profile.self_control - 0.5)
    put("anger", anger)

    # Warmth and care follow agreeableness and the expressive warmth trait.
    care = 1.0 + 0.30 * (profile.agreeableness - 0.5) + 0.30 * (profile.warmth_expressive - 0.5)
    put("tenderness", care)
    put("relief", care)

    # Needs follow affiliation and autonomy.
    put("connection", 1.0 + 0.30 * (profile.affiliation_need - 0.5) - 0.20 * (profile.attachment_avoidance - 0.5))
    put("closeness_need", 1.0 + 0.40 * (profile.attachment_anxiety - 0.5) + 0.30 * (profile.affiliation_need - 0.5))
    put("autonomy", 1.0 + 0.20 * (profile.openness - 0.5) + 0.20 * (profile.bas_drive - 0.5))
    return gains


def _apply_explicit_deltas(state: FullMentalState, appraisal: Any) -> dict[str, float]:
    """Apply named emotion and need deltas. Return the fields that changed.

    An explicit delta is a direct instruction from the accepted appraisal. It
    is applied at full weight; personality shapes the derived and decay paths,
    not a value the model already named.
    """
    changed: dict[str, float] = {}
    for name, delta in (getattr(appraisal, "emotion_delta", None) or {}).items():
        if name not in type(state.state_emotions).model_fields:
            continue
        field_info = type(state.state_emotions).model_fields[name]
        value = _clamp_value(float(getattr(state.state_emotions, name)) + float(delta), field_info)
        setattr(state.state_emotions, name, value)
        changed[name] = value
    for name, delta in (getattr(appraisal, "need_delta", None) or {}).items():
        if name not in type(state.state_needs).model_fields:
            continue
        field_info = type(state.state_needs).model_fields[name]
        value = _clamp_value(float(getattr(state.state_needs, name)) + float(delta), field_info)
        setattr(state.state_needs, name, value)
        changed[name] = value
    return changed


def _apply_canonical_axes(
    state: FullMentalState,
    appraisal: Any,
) -> tuple[dict[str, float], dict[str, float]]:
    """Apply canonical appraisal axes and return derived emotion and core values.

    Canonical axes use the ``AppraisalGeneral`` and ``AppraisalSocial`` field
    names. The existing ``compute_state_delta`` maps those axes to core affect
    and discrete emotions. This function only feeds the axes in and reports the
    result. It returns empty mappings when no canonical axis is present, so an
    explicit emotion delta is not wiped by an all-default appraisal.
    """
    general_fields = type(state.appraisal_general).model_fields
    social_fields = type(state.appraisal_social).model_fields
    axes = dict(getattr(appraisal, "appraisal", None) or {})
    applied: dict[str, float] = {}
    for name, value in axes.items():
        if name in general_fields:
            setattr(state.appraisal_general, name, _clamp_value(float(value), general_fields[name]))
            applied[name] = float(value)
        elif name in social_fields:
            setattr(state.appraisal_social, name, _clamp_value(float(value), social_fields[name]))
            applied[name] = float(value)
    if not applied:
        return {}, {}

    derived, _ = compute_state_delta(state)
    emotion_values = {
        name: float(getattr(derived.state_emotions, name))
        for name in StateEmotions.model_fields
    }
    core_values = {
        name: float(getattr(derived.state_core, name))
        for name in StateCore.model_fields
    }
    return emotion_values, core_values


def _blend_derived(
    state: FullMentalState,
    derived_emotions: Mapping[str, float],
    derived_core: Mapping[str, float],
    gain: float,
    gains: Mapping[str, float],
) -> None:
    """Blend the derived instant mapping into the current state as one EMA step.

    The per-field personality gain scales the blend so two trait profiles can
    react differently to the same accepted facts.
    """
    for name, target in derived_emotions.items():
        field_info = type(state.state_emotions).model_fields[name]
        current = float(getattr(state.state_emotions, name))
        per_field = min(1.0, max(0.0, gain * float(gains.get(name, 1.0))))
        setattr(state.state_emotions, name, _clamp_value(current + (target - current) * per_field, field_info))
    for name, target in derived_core.items():
        field_info = type(state.state_core).model_fields[name]
        current = float(getattr(state.state_core, name))
        setattr(state.state_core, name, _clamp_value(current + (target - current) * gain, field_info))


def _apply_expectation_error(state: FullMentalState, error: float, gains: Mapping[str, float] | None = None) -> dict[str, float]:
    """Move core affect and joy or sadness by one expectation error.

    ``error`` is in [-1, 1]. Positive means the outcome was better than
    expected; negative means worse. The magnitude also lowers certainty.
    """
    error = max(-1.0, min(1.0, float(error or 0.0)))
    if abs(error) < 1e-9:
        return {}
    gains = gains or {}
    valence_field = type(state.state_core).model_fields["valence"]
    state.state_core.valence = _clamp_value(
        float(state.state_core.valence) + EXPECTATION_VALENCE_GAIN * error, valence_field,
    )
    target = "joy" if error >= 0 else "sadness"
    target_field = type(state.state_emotions).model_fields[target]
    target_gain = float(gains.get(target, 1.0))
    state.state_emotions.__dict__[target] = _clamp_value(
        float(getattr(state.state_emotions, target)) + EXPECTATION_EMOTION_GAIN * abs(error) * target_gain,
        target_field,
    )
    certainty_field = type(state.appraisal_general).model_fields["certainty"]
    state.appraisal_general.certainty = _clamp_value(
        float(state.appraisal_general.certainty) - EXPECTATION_SURPRISE_GAIN * abs(error), certainty_field,
    )
    return {target: float(getattr(state.state_emotions, target)), "valence": float(state.state_core.valence)}


def summarize_affect(state: FullMentalState) -> str:
    """Return one short factual line about the accepted affect.

    The wording avoids a claim of a sensory experience. It reports the latent
    values that later stages use.
    """
    core = state.state_core
    emotions = state.state_emotions
    needs = state.state_needs
    top = sorted(
        ((name, float(getattr(emotions, name))) for name in StateEmotions.model_fields),
        key=lambda pair: -pair[1],
    )[:3]
    top = [pair for pair in top if pair[1] >= 0.05]
    top_text = ", ".join(f"{name} {value:.2f}" for name, value in top) or "no strong emotion"
    pressing = sorted(
        ((name, float(getattr(needs, name))) for name in StateNeeds.model_fields),
        key=lambda pair: pair[1],
    )[:2]
    need_text = ", ".join(f"{name} {value:.2f}" for name, value in pressing) or "no pressing need"
    return (
        f"core affect valence {core.valence:.2f}, arousal {core.arousal:.2f}, "
        f"dominance {core.dominance:.2f}; strongest emotions: {top_text}; "
        f"pressing needs: {need_text}"
    )


def _behavior_mapping(state: FullMentalState, attention_bias: tuple[float, float]) -> tuple[str, dict]:
    """Map accepted affect to a regulation mode and named behavior pressures."""
    emotions = state.state_emotions
    valence_focus, salience_bias = attention_bias

    # Game danger preempts slow regulation and contemplation. The threat can
    # come from the appraisal axes even before a discrete fear value forms.
    threat = float(state.appraisal_general.bodily_threat)
    aversive = float(state.appraisal_general.aversive_pressure)
    if (threat >= 0.6 or aversive >= 0.75
            or emotions.fear >= THREAT_FEAR_THRESHOLD
            or emotions.anger >= ANGER_REACTIVE_THRESHOLD):
        regulation = REGULATION_REACTIVE
    elif emotions.sadness >= SADNESS_REFLECTIVE_THRESHOLD or state.state_expression_ext.disengage_silence >= WITHDRAWAL_REFLECTIVE_THRESHOLD:
        regulation = REGULATION_REFLECTIVE
    else:
        regulation = REGULATION_BLENDED

    if valence_focus <= -0.2:
        attention_focus = "threat and loss"
    elif valence_focus >= 0.2:
        attention_focus = "reward and connection"
    else:
        attention_focus = "neutral"

    behavior = {
        "regulation": regulation,
        "attention_focus": attention_focus,
        "valence_focus_bias": round(float(valence_focus), 4),
        "salience_bias": round(float(salience_bias), 4),
        "persistence": round(min(1.0, float(emotions.pride) + float(state.state_cognition.willpower) * 0.5 + 0.2), 4),
        "exploration": round(min(1.0, float(emotions.curiosity) + float(state.state_personality.openness) * 0.3), 4),
        "affiliation": round(min(1.0, float(emotions.tenderness) + float(emotions.joy) * 0.3 + float(state.state_needs.closeness_need) * 0.3), 4),
        "withdrawal": round(float(state.state_expression_ext.disengage_silence), 4),
    }
    return regulation, behavior


def project_affect(
    previous_vec: list[float] | None,
    appraisal: Any,
    *,
    elapsed_seconds: float = 0.0,
    personality: Any = None,
    expectation_error: float = 0.0,
) -> AffectProjection:
    """Apply elapsed-time decay, appraisal, and personality to one new state.

    ``previous_vec`` is the latent vector of the last accepted cycle. An empty
    or missing vector starts from an empty state. ``appraisal`` accepts the
    typed :class:`~workflows.main_cognition.schemas.AppraisalResult` shape or
    any object with the same optional attributes. ``personality`` accepts a
    :class:`PersonalityProfile` or a mapping or vector.
    """
    current = FullMentalState.init_from_list(list(previous_vec)) if previous_vec else create_empty_state()
    profile = parse_personality_profile(personality) if personality is not None else current.state_personality
    current.state_personality = profile

    decayed = decay_toward_baseline(current, elapsed_seconds, personality=profile)
    working = decayed.model_copy(deep=True)
    gains = personality_gains(profile)

    derived_emotions, derived_core = _apply_canonical_axes(working, appraisal)
    if derived_emotions or derived_core:
        _blend_derived(working, derived_emotions, derived_core, BASE_APPRAISAL_GAIN, gains)

    changed = _apply_explicit_deltas(working, appraisal)
    expectation_changed = _apply_expectation_error(working, expectation_error, gains)
    working = _clamp_state(working)

    bias = compute_attention_bias(working)
    regulation, behavior = _behavior_mapping(working, bias)
    trace = {
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "emotion_half_life_seconds": EMOTION_HALF_LIFE_SECONDS,
        "need_half_life_seconds": NEED_HALF_LIFE_SECONDS,
        "personality_gains": {name: round(value, 4) for name, value in gains.items()},
        "derived_emotions": {name: round(value, 4) for name, value in derived_emotions.items()},
        "derived_core": {name: round(value, 4) for name, value in derived_core.items()},
        "explicit_emotions": {name: round(value, 4) for name, value in changed.items()},
        "expectation_error": round(float(expectation_error or 0.0), 4),
        "expectation_effect": {name: round(value, 4) for name, value in expectation_changed.items()},
        "regulation": regulation,
        "behavior": behavior,
    }
    delta_vec = [new - old for new, old in zip(working.to_list(), current.to_list())]
    delta = FullMentalState.init_from_list(delta_vec)
    return AffectProjection(
        state=working,
        delta=delta,
        trace=trace,
        summary=summarize_affect(working),
        attention_bias=bias,
        regulation=regulation,
        behavior=behavior,
    )


__all__ = [
    "AffectProjection",
    "EMOTION_HALF_LIFE_SECONDS",
    "NEED_HALF_LIFE_SECONDS",
    "REGULATION_BLENDED",
    "REGULATION_REACTIVE",
    "REGULATION_REFLECTIVE",
    "decay_toward_baseline",
    "personality_gains",
    "project_affect",
    "summarize_affect",
]