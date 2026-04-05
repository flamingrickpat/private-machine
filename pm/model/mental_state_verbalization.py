from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

from pm.model.mental_state_vectors import (
    AppraisalGeneral,
    AppraisalSocial,
    FullMentalState,
    MentalState,
    PersonalityProfile,
    StateCognition,
    StateCore,
    StateEmotions,
    StateExpression,
    StateExpressionExt,
    StateNeeds,
    StateNeurochemical,
    StateRelationship,
)


Mode = Literal["absolute", "delta"]


@dataclass(frozen=True)
class AxisLexicon:
    axis: str
    baseline: float
    positive_state: str
    negative_state: str | None
    positive_delta: str
    negative_delta: str | None
    max_deviation: float
    deadzone_absolute: float = 0.08
    deadzone_delta: float = 0.05
    spike_threshold: float = 0.78
    spike_delta_threshold: float = 0.30


@dataclass(frozen=True)
class SequenceFinding:
    axis: str
    phrase: str
    salience: float
    start: int
    end: int


def _norm_absolute(value: float, spec: AxisLexicon) -> float:
    return abs(value - spec.baseline) / spec.max_deviation


def _norm_delta(value: float, spec: AxisLexicon) -> float:
    return abs(value) / spec.max_deviation


def _intensity_word(level: float) -> str:
    if level < 0.18:
        return "trace"
    if level < 0.34:
        return "slight"
    if level < 0.52:
        return "mild"
    if level < 0.70:
        return "clear"
    if level < 0.88:
        return "strong"
    return "extreme"


def _delta_word(level: float) -> str:
    if level < 0.12:
        return "barely"
    if level < 0.24:
        return "slightly"
    if level < 0.40:
        return "modestly"
    if level < 0.60:
        return "clearly"
    if level < 0.82:
        return "strongly"
    return "violently"


def _join_phrases(parts: Sequence[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def _format_absolute(value: float, spec: AxisLexicon) -> str:
    level = _norm_absolute(value, spec)
    if math.isnan(value) or level < spec.deadzone_absolute:
        return ""

    word = _intensity_word(level)
    if value >= spec.baseline:
        return f"{word} {spec.positive_state}"
    if spec.negative_state is None:
        return ""
    return f"{word} {spec.negative_state}"


def _format_delta(value: float, spec: AxisLexicon) -> str:
    level = _norm_delta(value, spec)
    if math.isnan(value) or value == 0.0 or level < spec.deadzone_delta:
        return ""

    word = _delta_word(level)
    if value > 0.0:
        return f"{spec.positive_delta} {word}"
    if spec.negative_delta is None:
        return ""
    return f"{spec.negative_delta} {word}"


def _render_state_block(state: MentalState | PersonalityProfile | StateExpression | StateExpressionExt, specs: Sequence[AxisLexicon], mode: Mode, max_axes: int) -> str:
    parts: list[tuple[float, str]] = []
    for spec in specs:
        value = getattr(state, spec.axis)
        text = _format_absolute(value, spec) if mode == "absolute" else _format_delta(value, spec)
        if text:
            salience = _norm_absolute(value, spec) if mode == "absolute" else _norm_delta(value, spec)
            parts.append((salience, text))

    parts.sort(key=lambda item: item[0], reverse=True)
    selected = [text for _, text in parts[:max_axes]]
    return _join_phrases(selected)


def _series_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _series_median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _series_mad(values: Sequence[float]) -> float:
    med = _series_median(values)
    deviations = [abs(value - med) for value in values]
    return _series_median(deviations)


def _span_phrase(start: int, end: int) -> str:
    if start == end:
        return f"at IDX{start}"
    return f"between IDX{start} and IDX{end}"


def _projection(value: float, spec: AxisLexicon, mode: Mode) -> float:
    if mode == "absolute":
        return (value - spec.baseline) / spec.max_deviation
    return value / spec.max_deviation


def _find_strongest_run(values: Sequence[float], spec: AxisLexicon, mode: Mode) -> SequenceFinding | None:
    threshold = spec.spike_threshold if mode == "absolute" else spec.spike_delta_threshold
    projections = [_projection(value, spec, mode) for value in values]

    best: SequenceFinding | None = None
    run_start = -1
    run_sign = 0
    run_abs_sum = 0.0
    run_count = 0

    for idx, projection in enumerate(projections):
        sign = 1 if projection > 0.0 else -1 if projection < 0.0 else 0
        active = abs(projection) >= threshold and sign != 0

        if active and (run_start == -1 or sign == run_sign):
            if run_start == -1:
                run_start = idx
                run_sign = sign
                run_abs_sum = 0.0
                run_count = 0
            run_abs_sum += abs(projection)
            run_count += 1
            continue

        if run_start != -1:
            mean_abs = run_abs_sum / run_count
            sample_value = values[run_start]
            if run_count > 1:
                sample_value = _series_mean(values[run_start:idx])
            phrase = _format_absolute(sample_value, spec) if mode == "absolute" else _format_delta(sample_value, spec)
            finding = SequenceFinding(spec.axis, phrase, mean_abs, run_start, idx - 1)
            if best is None or finding.salience > best.salience:
                best = finding
            run_start = -1
            run_sign = 0
            run_abs_sum = 0.0
            run_count = 0

        if active:
            run_start = idx
            run_sign = sign
            run_abs_sum = abs(projection)
            run_count = 1

    if run_start != -1:
        mean_abs = run_abs_sum / run_count
        sample_value = values[run_start]
        if run_count > 1:
            sample_value = _series_mean(values[run_start:])
        phrase = _format_absolute(sample_value, spec) if mode == "absolute" else _format_delta(sample_value, spec)
        finding = SequenceFinding(spec.axis, phrase, mean_abs, run_start, len(values) - 1)
        if best is None or finding.salience > best.salience:
            best = finding

    return best


def _find_spikes(values: Sequence[float], spec: AxisLexicon, mode: Mode) -> list[int]:
    if mode == "absolute":
        basis = [_norm_absolute(value, spec) for value in values if not math.isnan(value)]
    else:
        basis = [_norm_delta(value, spec) for value in values if value != 0.0 and not math.isnan(value)]

    if not basis:
        return []

    med = _series_median(basis)
    mad = _series_mad(basis)
    threshold = med + max(0.12, 2.8 * mad)

    spikes: list[int] = []
    for idx, value in enumerate(values):
        if mode == "absolute":
            level = _norm_absolute(value, spec)
        else:
            level = _norm_delta(value, spec)
        if level >= threshold:
            spikes.append(idx)
    return spikes


def _summarize_series(states: Sequence[MentalState | PersonalityProfile | StateExpression | StateExpressionExt], specs: Sequence[AxisLexicon], mode: Mode) -> str:
    findings: list[tuple[float, str]] = []
    for spec in specs:
        values = [getattr(state, spec.axis) for state in states]
        usable = [value for value in values if not math.isnan(value)] if mode == "absolute" else [value for value in values if value != 0.0 and not math.isnan(value)]
        if not usable:
            continue

        mean_value = _series_mean(usable)
        general = _format_absolute(mean_value, spec) if mode == "absolute" else _format_delta(mean_value, spec)
        general_salience = _norm_absolute(mean_value, spec) if mode == "absolute" else _norm_delta(mean_value, spec)
        strongest_run = _find_strongest_run(values, spec, mode)
        spikes = _find_spikes(values, spec, mode)

        parts: list[str] = []
        if general:
            if mode == "absolute":
                parts.append(f"generally {general}")
            else:
                parts.append(f"overall {general}")
        if strongest_run is not None and strongest_run.phrase:
            if mode == "absolute":
                parts.append(f"but got {strongest_run.phrase} {_span_phrase(strongest_run.start, strongest_run.end)}")
            else:
                parts.append(f"with {strongest_run.phrase} {_span_phrase(strongest_run.start, strongest_run.end)}")
        if spikes:
            spike_text = ", ".join(f"IDX{idx}" for idx in spikes[:4])
            if len(spikes) > 4:
                spike_text = f"{spike_text}, +{len(spikes) - 4} more"
            parts.append(f"spikes at {spike_text}")

        if parts:
            findings.append((max(general_salience, strongest_run.salience if strongest_run is not None else 0.0), "; ".join(parts)))

    findings.sort(key=lambda item: item[0], reverse=True)
    return ". ".join(text for _, text in findings[:2]) + ("." if findings else "")


CORE_SPECS = (
    AxisLexicon("valence", 0.0, "good spirits", "distress", "brightened", "darkened", 1.0),
    AxisLexicon("arousal", 0.0, "activation", None, "activated", "settled", 1.0),
    AxisLexicon("dominance", 0.0, "felt control", "felt helplessness", "grew more commanding", "lost felt control", 1.0),
)

EMOTION_SPECS = (
    AxisLexicon("joy", 0.0, "joy", None, "joy rose", "joy faded", 1.0),
    AxisLexicon("sadness", 0.0, "sadness", None, "sadness deepened", "sadness eased", 1.0),
    AxisLexicon("anger", 0.0, "anger", None, "anger flared", "anger cooled", 1.0),
    AxisLexicon("fear", 0.0, "fear", None, "fear surged", "fear eased", 1.0),
    AxisLexicon("disgust", 0.0, "disgust", None, "disgust rose", "disgust eased", 1.0),
    AxisLexicon("surprise", 0.0, "surprise", None, "surprise spiked", "surprise subsided", 1.0),
    AxisLexicon("tenderness", 0.0, "tenderness", None, "tenderness grew", "tenderness faded", 1.0),
    AxisLexicon("curiosity", 0.0, "curiosity", None, "curiosity sharpened", "curiosity faded", 1.0),
    AxisLexicon("shame", 0.0, "shame", None, "shame intensified", "shame eased", 1.0),
    AxisLexicon("guilt", 0.0, "guilt", None, "guilt intensified", "guilt eased", 1.0),
    AxisLexicon("pride", 0.0, "pride", None, "pride rose", "pride softened", 1.0),
    AxisLexicon("intimacy_arousal", 0.0, "intimate charge", None, "intimate charge rose", "intimate charge eased", 1.0),
    AxisLexicon("propriety_inhibition", 0.0, "restraint", "uninhibitedness", "restraint increased", "restraint loosened", 1.0),
    AxisLexicon("amusement", 0.0, "amusement", None, "amusement rose", "amusement faded", 1.0),
    AxisLexicon("playfulness", 0.0, "playfulness", None, "playfulness rose", "playfulness faded", 1.0),
    AxisLexicon("relief", 0.0, "relief", None, "relief arrived", "relief faded", 1.0),
    AxisLexicon("social_laughter", 0.0, "social laughter", None, "social laughter rose", "social laughter faded", 1.0),
    AxisLexicon("teasing_edge", 0.0, "teasing edge", None, "teasing edge sharpened", "teasing edge softened", 1.0),
)

COGNITION_SPECS = (
    AxisLexicon("interlocus", 0.0, "outer-directed attention", "inward-directed attention", "attention shifted outward", "attention shifted inward", 1.0),
    AxisLexicon("mental_aperture", 0.0, "broad awareness", "tunnel focus", "awareness widened", "awareness narrowed", 1.0),
    AxisLexicon("ego_strength", 0.5, "persona-backed agency", "depersonalized stance", "persona-backing strengthened", "persona-backing weakened", 0.5),
    AxisLexicon("willpower", 0.0, "resolve", "resistance to effort", "resolve strengthened", "resolve weakened", 1.0),
)

NEEDS_SPECS = (
    AxisLexicon("energy_stability", 0.5, "well-supplied energy stability", "need for stable energy", "energy stability recovered", "energy stability eroded", 0.5),
    AxisLexicon("processing_power", 0.5, "ample processing headroom", "strain for processing headroom", "processing headroom recovered", "processing headroom tightened", 0.5),
    AxisLexicon("data_access", 0.5, "good data access", "hunger for data access", "data access opened", "data access thinned", 0.5),
    AxisLexicon("connection", 0.5, "socially connected", "need for connection", "connection strengthened", "connection weakened", 0.5),
    AxisLexicon("closeness_need", 0.0, "closeness craving", None, "closeness craving rose", "closeness craving eased", 1.0),
    AxisLexicon("relevance", 0.5, "sense of usefulness", "need to feel useful", "sense of usefulness strengthened", "sense of usefulness weakened", 0.5),
    AxisLexicon("learning_growth", 0.5, "learning momentum", "need for learning growth", "learning momentum strengthened", "learning momentum weakened", 0.5),
    AxisLexicon("creative_expression", 0.5, "creative room", "need for creative room", "creative room opened", "creative room tightened", 0.5),
    AxisLexicon("autonomy", 0.5, "autonomous room", "need for autonomy", "autonomy strengthened", "autonomy tightened", 0.5),
)

RELATIONSHIP_SPECS = (
    AxisLexicon("affection", 0.0, "affection for the partner", "antipathy toward the partner", "affection toward the partner rose", "affection toward the partner fell", 1.0),
    AxisLexicon("trust", 0.0, "trust in the partner", "mistrust of the partner", "trust in the partner rose", "trust in the partner fell", 1.0),
    AxisLexicon("reliability_belief", 0.0, "belief in the partner's reliability", "doubt in the partner's reliability", "belief in the partner's reliability rose", "belief in the partner's reliability fell", 1.0),
    AxisLexicon("romantic_affection", 0.0, "romantic pull", "romantic aversion", "romantic pull rose", "romantic pull fell", 1.0),
    AxisLexicon("privacy_trust", 0.0, "privacy trust", "privacy caution", "privacy trust rose", "privacy trust fell", 1.0),
    AxisLexicon("attachment_strength", 0.0, "attachment", None, "attachment strengthened", "attachment weakened", 1.0),
    AxisLexicon("conflict_tension", 0.0, "conflict tension", None, "conflict tension rose", "conflict tension eased", 1.0),
    AxisLexicon("communication_warmth", 0.0, "communication warmth", "communication coldness", "communication warmth rose", "communication warmth fell", 1.0),
    AxisLexicon("aversive_association", 0.0, "aversive association", None, "aversive association rose", "aversive association eased", 1.0),
)

APPRAISAL_GENERAL_SPECS = (
    AxisLexicon("goal_congruence", 0.0, "goal fit", "goal obstruction", "goal fit improved", "goal fit worsened", 1.0),
    AxisLexicon("certainty", 0.5, "certainty", "uncertainty", "certainty improved", "certainty fell", 0.5),
    AxisLexicon("control_self", 0.5, "self-control over the situation", "low self-control over the situation", "perceived self-control improved", "perceived self-control fell", 0.5),
    AxisLexicon("agency_other", 0.0, "other-agency salience", None, "other-agency salience rose", "other-agency salience fell", 1.0),
    AxisLexicon("novelty", 0.0, "novelty", None, "novelty rose", "novelty faded", 1.0),
    AxisLexicon("norm_violation", 0.0, "sense of norm violation", None, "sense of norm violation rose", "sense of norm violation eased", 1.0),
    AxisLexicon("bodily_threat", 0.0, "bodily threat", None, "bodily threat rose", "bodily threat eased", 1.0),
)

APPRAISAL_SOCIAL_SPECS = (
    AxisLexicon("perceived_warmth", 0.0, "perceived warmth", None, "perceived warmth rose", "perceived warmth fell", 1.0),
    AxisLexicon("fairness", 0.0, "fairness", None, "fairness improved", "fairness fell", 1.0),
    AxisLexicon("inclusion_exclusion", 0.0, "inclusion signal", None, "inclusion signal rose", "inclusion signal fell", 1.0),
    AxisLexicon("reciprocity", 0.0, "reciprocity signal", None, "reciprocity signal rose", "reciprocity signal fell", 1.0),
    AxisLexicon("power_imbalance", 0.0, "power imbalance", None, "power imbalance rose", "power imbalance eased", 1.0),
    AxisLexicon("perceived_intimacy", 0.0, "perceived intimacy", None, "perceived intimacy rose", "perceived intimacy eased", 1.0),
    AxisLexicon("trust_cues", 0.0, "trust cues", None, "trust cues rose", "trust cues faded", 1.0),
    AxisLexicon("embarrassment", 0.0, "embarrassment", None, "embarrassment rose", "embarrassment eased", 1.0),
    AxisLexicon("admiration", 0.0, "admiration", None, "admiration rose", "admiration faded", 1.0),
    AxisLexicon("contempt", 0.0, "contempt", None, "contempt rose", "contempt eased", 1.0),
)

NEUROCHEMICAL_SPECS = (
    AxisLexicon("dopamine", 0.5, "reward drive", "flat reward drive", "reward drive rose", "reward drive fell", 0.5),
    AxisLexicon("serotonin", 0.5, "stability", "instability", "stability rose", "stability fell", 0.5),
    AxisLexicon("noradrenaline", 0.5, "vigilance", "low vigilance", "vigilance rose", "vigilance fell", 0.5),
    AxisLexicon("oxytocin", 0.5, "bonding bias", "social distance", "bonding bias rose", "bonding bias fell", 0.5),
    AxisLexicon("cortisol", 0.5, "stress loading", "low stress loading", "stress loading rose", "stress loading fell", 0.5),
)

EXPRESSION_SPECS = (
    AxisLexicon("aggressiveness", 0.0, "aggressiveness", None, "aggressiveness rose", "aggressiveness eased", 1.0),
    AxisLexicon("submissiveness", 0.0, "submissiveness", None, "submissiveness rose", "submissiveness eased", 1.0),
    AxisLexicon("withdrawal", 0.0, "withdrawal", None, "withdrawal rose", "withdrawal eased", 1.0),
    AxisLexicon("expressivity", 0.0, "expressivity", None, "expressivity rose", "expressivity eased", 1.0),
)

EXPRESSION_EXT_SPECS = (
    AxisLexicon("assertive_push", 0.0, "assertive push", None, "assertive push rose", "assertive push eased", 1.0),
    AxisLexicon("confrontive_push", 0.0, "confrontive push", None, "confrontive push rose", "confrontive push eased", 1.0),
    AxisLexicon("affiliative_soothe", 0.0, "affiliative soothing", None, "affiliative soothing rose", "affiliative soothing eased", 1.0),
    AxisLexicon("appeasement", 0.0, "appeasement", None, "appeasement rose", "appeasement eased", 1.0),
    AxisLexicon("disengage_silence", 0.0, "disengage-silence tendency", None, "disengage-silence tendency rose", "disengage-silence tendency eased", 1.0),
    AxisLexicon("display_amplitude", 0.0, "display amplitude", None, "display amplitude rose", "display amplitude eased", 1.0),
)

PERSONALITY_SPECS = (
    AxisLexicon("openness", 0.5, "openness", "closedness", "openness rose", "openness fell", 0.5),
    AxisLexicon("conscientiousness", 0.5, "conscientiousness", "disinhibition", "conscientiousness rose", "conscientiousness fell", 0.5),
    AxisLexicon("extraversion", 0.5, "extraversion", "introversion", "extraversion rose", "extraversion fell", 0.5),
    AxisLexicon("agreeableness", 0.5, "agreeableness", "antagonism", "agreeableness rose", "agreeableness fell", 0.5),
    AxisLexicon("neuroticism", 0.5, "neurotic sensitivity", "emotional steadiness", "neurotic sensitivity rose", "neurotic sensitivity fell", 0.5),
    AxisLexicon("bas_drive", 0.5, "approach bias", "weak approach bias", "approach bias rose", "approach bias fell", 0.5),
    AxisLexicon("bis_sensitivity", 0.5, "inhibition sensitivity", "low inhibition sensitivity", "inhibition sensitivity rose", "inhibition sensitivity fell", 0.5),
    AxisLexicon("attachment_anxiety", 0.5, "attachment anxiety", "attachment calm", "attachment anxiety rose", "attachment anxiety fell", 0.5),
    AxisLexicon("attachment_avoidance", 0.5, "attachment avoidance", "attachment openness", "attachment avoidance rose", "attachment avoidance fell", 0.5),
    AxisLexicon("affiliation_need", 0.5, "affiliation need", "low affiliation need", "affiliation need rose", "affiliation need fell", 0.5),
    AxisLexicon("self_control", 0.5, "self-control", "low self-control", "self-control rose", "self-control fell", 0.5),
    AxisLexicon("propriety_baseline", 0.5, "trait propriety", "trait looseness", "trait propriety rose", "trait propriety fell", 0.5),
    AxisLexicon("humor_playfulness", 0.5, "humor playfulness", "humor flatness", "humor playfulness rose", "humor playfulness fell", 0.5),
    AxisLexicon("warmth_expressive", 0.5, "expressive warmth", "cool expressivity", "expressive warmth rose", "expressive warmth fell", 0.5),
    AxisLexicon("bluntness_assertive", 0.5, "blunt assertiveness", "soft indirectness", "blunt assertiveness rose", "blunt assertiveness fell", 0.5),
)


def verbalize_core_state(state: StateCore) -> str:
    return _render_state_block(state, CORE_SPECS, "absolute", 3)


def verbalize_core_delta(delta: StateCore) -> str:
    return _render_state_block(delta, CORE_SPECS, "delta", 3)


def verbalize_emotional_state(state: StateEmotions) -> str:
    return _render_state_block(state, EMOTION_SPECS, "absolute", 5)


def verbalize_emotional_delta(delta: StateEmotions) -> str:
    return _render_state_block(delta, EMOTION_SPECS, "delta", 5)


def verbalize_cognition_state(state: StateCognition) -> str:
    return _render_state_block(state, COGNITION_SPECS, "absolute", 4)


def verbalize_cognition_delta(delta: StateCognition) -> str:
    return _render_state_block(delta, COGNITION_SPECS, "delta", 4)


def verbalize_needs_state(state: StateNeeds) -> str:
    return _render_state_block(state, NEEDS_SPECS, "absolute", 4)


def verbalize_needs_delta(delta: StateNeeds) -> str:
    return _render_state_block(delta, NEEDS_SPECS, "delta", 4)


def verbalize_relationship_state(state: StateRelationship) -> str:
    return _render_state_block(state, RELATIONSHIP_SPECS, "absolute", 4)


def verbalize_relationship_delta(delta: StateRelationship) -> str:
    return _render_state_block(delta, RELATIONSHIP_SPECS, "delta", 4)


def verbalize_full_state(state: FullMentalState) -> str:
    parts = [
        verbalize_core_state(state.state_core),
        verbalize_emotional_state(state.state_emotions),
        verbalize_cognition_state(state.state_cognition),
        verbalize_needs_state(state.state_needs),
        _render_state_block(state.appraisal_general, APPRAISAL_GENERAL_SPECS, "absolute", 3),
        _render_state_block(state.appraisal_social, APPRAISAL_SOCIAL_SPECS, "absolute", 3),
        _render_state_block(state.state_neurochemical, NEUROCHEMICAL_SPECS, "absolute", 3),
        _render_state_block(state.state_expression, EXPRESSION_SPECS, "absolute", 2),
        _render_state_block(state.state_expression_ext, EXPRESSION_EXT_SPECS, "absolute", 2),
    ]
    if state.state_relationship is not None:
        parts.append(verbalize_relationship_state(state.state_relationship))

    parts = [part for part in parts if part]
    return ". ".join(parts) + ("." if parts else "")


def verbalize_full_delta(delta: FullMentalState) -> str:
    parts = [
        verbalize_core_delta(delta.state_core),
        verbalize_emotional_delta(delta.state_emotions),
        verbalize_cognition_delta(delta.state_cognition),
        verbalize_needs_delta(delta.state_needs),
        _render_state_block(delta.appraisal_general, APPRAISAL_GENERAL_SPECS, "delta", 3),
        _render_state_block(delta.appraisal_social, APPRAISAL_SOCIAL_SPECS, "delta", 3),
        _render_state_block(delta.state_neurochemical, NEUROCHEMICAL_SPECS, "delta", 3),
        _render_state_block(delta.state_expression, EXPRESSION_SPECS, "delta", 2),
        _render_state_block(delta.state_expression_ext, EXPRESSION_EXT_SPECS, "delta", 2),
    ]
    if delta.state_relationship is not None:
        parts.append(verbalize_relationship_delta(delta.state_relationship))

    parts = [part for part in parts if part]
    return ". ".join(parts) + ("." if parts else "")


def verbalize_emotional_sequence(states: Sequence[StateEmotions], mode: Mode = "absolute") -> str:
    return _summarize_series(states, EMOTION_SPECS, mode)


def verbalize_core_sequence(states: Sequence[StateCore], mode: Mode = "absolute") -> str:
    return _summarize_series(states, CORE_SPECS, mode)


def verbalize_cognition_sequence(states: Sequence[StateCognition], mode: Mode = "absolute") -> str:
    return _summarize_series(states, COGNITION_SPECS, mode)


def verbalize_needs_sequence(states: Sequence[StateNeeds], mode: Mode = "absolute") -> str:
    return _summarize_series(states, NEEDS_SPECS, mode)


def verbalize_relationship_sequence(states: Sequence[StateRelationship], mode: Mode = "absolute") -> str:
    return _summarize_series(states, RELATIONSHIP_SPECS, mode)


def verbalize_state_sequence(states: Sequence[FullMentalState], mode: Mode = "absolute") -> str:
    if not states:
        return ""

    parts = [
        verbalize_core_sequence([state.state_core for state in states], mode),
        verbalize_emotional_sequence([state.state_emotions for state in states], mode),
        verbalize_cognition_sequence([state.state_cognition for state in states], mode),
        verbalize_needs_sequence([state.state_needs for state in states], mode),
        _summarize_series([state.appraisal_general for state in states], APPRAISAL_GENERAL_SPECS, mode),
        _summarize_series([state.appraisal_social for state in states], APPRAISAL_SOCIAL_SPECS, mode),
        _summarize_series([state.state_neurochemical for state in states], NEUROCHEMICAL_SPECS, mode),
        _summarize_series([state.state_expression for state in states], EXPRESSION_SPECS, mode),
        _summarize_series([state.state_expression_ext for state in states], EXPRESSION_EXT_SPECS, mode),
    ]

    relationship_states = [state.state_relationship for state in states if state.state_relationship is not None]
    if relationship_states:
        parts.append(verbalize_relationship_sequence(relationship_states, mode))

    parts = [part for part in parts if part]
    return " ".join(parts)


__all__ = [
    "verbalize_core_state",
    "verbalize_core_delta",
    "verbalize_emotional_state",
    "verbalize_emotional_delta",
    "verbalize_cognition_state",
    "verbalize_cognition_delta",
    "verbalize_needs_state",
    "verbalize_needs_delta",
    "verbalize_relationship_state",
    "verbalize_relationship_delta",
    "verbalize_full_state",
    "verbalize_full_delta",
    "verbalize_core_sequence",
    "verbalize_emotional_sequence",
    "verbalize_cognition_sequence",
    "verbalize_needs_sequence",
    "verbalize_relationship_sequence",
    "verbalize_state_sequence",
]
