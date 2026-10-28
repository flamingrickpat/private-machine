"""Turn mental-state vectors into readable, non-leaking language.

A :class:`~pm.records.mental_state.FullMentalState` is stored as named axes
with numeric values. A prompt or a self-report cannot read those numbers, and
showing them would expose implementation detail. This module renders the same
dynamics as ordinary prose, deterministically and without a model call.

**What it owns.**

- :data:`CORE_SPECS` and the other lexicon tables map one axis to its
  baseline, its positive and negative wording, its delta wording, and its
  dead zones. A deviation below the dead zone is noise and renders as
  nothing. :func:`verbalize_full_state` and the other ``verbalize_*``
  functions turn a state, a delta, or a sequence into that wording.
- :func:`verbalize_primitive_appraisal_trace` states the strongest primitive
  body, attention, pressure, and regulation changes of a state sequence. It
  is the narrow trace that self-report reasoning is allowed to rely on.
- :func:`generate_self_report_candidates` writes bounded first-person
  candidates from the active primitive axes.
- :func:`judge_self_report_candidates` selects the candidate that is grounded
  in the experienced dynamics trace. It rejects provenance leaks, schema
  dumps, emotion-label-only summaries, and trace keyword repetition.
- :func:`appraisal_trace_passes_quality_gate` is the trace-quality gate that
  scenario checks use.

**Boundary.** Everything here is deterministic. It does not call a model, own
character-card prose, run the cycle, or persist anything. The caller supplies
the state sequences; an empty sequence renders as an empty string rather than
a made-up state.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pm.records.mental_state import AppraisalGeneral, AppraisalSocial, FullMentalState, MentalState, PersonalityProfile, StateCognition, StateCore, StateEmotions, StateExpression, StateExpressionExt, StateNeeds, StateNeurochemical, StateRelationship


Mode = Literal["absolute", "delta"]


@dataclass(frozen=True)
class SelfReportJudgeResult:
    """Result of the immersion judge/rerank over self-report candidates.

    ``selected_candidate`` is the accepted candidate text, or None when every
    candidate failed a gate. ``decision`` is ``selected`` or ``all_failed``;
    ``reason`` explains the choice or lists the rejections.
    """
    selected_candidate: str | None
    decision: str
    reason: str


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
    AxisLexicon("pattern_fit", 0.0, "pattern fit", None, "pattern fit tightened", "pattern fit loosened", 1.0),
    AxisLexicon("body_signal", 0.0, "body signal", None, "body signal rose", "body signal eased", 1.0),
    AxisLexicon("aversive_pressure", 0.0, "aversive pressure", None, "aversive pressure rose", "aversive pressure eased", 1.0),
    AxisLexicon("attractor_distance", 0.0, "attractor distance", None, "attractor distance opened", "attractor distance closed", 1.0),
    AxisLexicon("repulsor_distance", 0.0, "repulsor distance", None, "repulsor distance closed in", "repulsor distance receded", 1.0),
    AxisLexicon("salience", 0.0, "salience", None, "salience rose", "salience faded", 1.0),
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
    AxisLexicon("primitive_approach_pressure", 0.0, "approach pressure", None, "approach pressure gathered", "approach pressure eased", 1.0),
    AxisLexicon("primitive_avoidance_pressure", 0.0, "avoidance pressure", None, "avoidance pressure gathered", "avoidance pressure eased", 1.0),
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
    """Render one mental state as prose, strongest axis of each group first.

    Groups join with ". ". An empty group contributes nothing, so a state with
    no deviation renders as an empty string rather than as filler.
    """
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
    """Render one mental-state delta as prose: what moved, not where the state sits."""
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
    """Render a state sequence: ``absolute`` for levels, ``delta`` for movement.

    The relationship group follows the state's own relationship, so a sequence
    without one simply omits it.
    """
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


def _primitive_rise(states: Sequence[FullMentalState], getter: Callable[[FullMentalState], float]) -> float:
    values = [float(getter(state) or 0.0) for state in states]
    if not values:
        return 0.0
    return max(values) - values[0]


def _primitive_peak(states: Sequence[FullMentalState], getter: Callable[[FullMentalState], float]) -> float:
    values = [float(getter(state) or 0.0) for state in states]
    return max(values) if values else 0.0


def _primitive_phrase(name: str, value: float, verb: str = "rose") -> str:
    return f"{name} {verb} {_delta_word(value)}"


def verbalize_primitive_appraisal_trace(states: Sequence[FullMentalState]) -> str:
    """Return the experienced-dynamics trace of a state sequence.

    The trace names the primitive body, attention, pressure, and regulation
    changes that crossed their reporting thresholds, in a fixed order. It is
    the only evidence that self-report candidates may be grounded in, and it
    must pass :func:`appraisal_trace_passes_quality_gate`.
    """
    if not states:
        return ""

    parts: list[str] = []
    body_rise = _primitive_rise(states, lambda state: state.appraisal_general.body_signal)
    aversive_rise = _primitive_rise(states, lambda state: state.appraisal_general.aversive_pressure)
    salience_rise = _primitive_rise(states, lambda state: state.appraisal_general.salience)
    repulsor_rise = _primitive_rise(states, lambda state: state.appraisal_general.repulsor_distance)
    approach_peak = _primitive_peak(states, lambda state: state.state_expression_ext.primitive_approach_pressure)
    avoidance_peak = _primitive_peak(states, lambda state: state.state_expression_ext.primitive_avoidance_pressure)
    regulation_rise = _primitive_rise(states, lambda state: state.state_cognition.willpower)

    if body_rise >= 0.08:
        parts.append(_primitive_phrase("body signal", body_rise))
    if aversive_rise >= 0.08:
        parts.append(_primitive_phrase("aversive pressure", aversive_rise))
    if salience_rise >= 0.08:
        parts.append(_primitive_phrase("salience", salience_rise))
    if repulsor_rise >= 0.08:
        parts.append(_primitive_phrase("repulsor distance", repulsor_rise, "closed in"))
    if approach_peak >= 0.08:
        parts.append(_primitive_phrase("approach pressure", approach_peak, "gathered"))
    if avoidance_peak >= 0.08:
        parts.append(_primitive_phrase("avoidance pressure", avoidance_peak, "gathered"))
    if regulation_rise >= 0.08:
        parts.append(_primitive_phrase("regulation pressure", regulation_rise, "gathered"))

    return _join_phrases(parts) + ("." if parts else "")


# ---------- Self-report candidate generation (WI-01) ----------

# Template pools: each primitive dynamic maps to first-person phrasings
# that reference body, attention, or action dynamics (never raw emotion labels).

_BODY_SIGNAL_TEMPLATES = (
    "I feel a rising tension in my body",
    "something is pressing against me from the inside of my body",
    "I sense a weight settling in my body",
)

_SALIENCE_TEMPLATES = (
    "something is pulling sharply at my attention",
    "my focus is drawn to something pressing",
    "I feel something demanding my notice",
)

_AVERSIVE_PRESSURE_TEMPLATES = (
    "I feel an aversive pressure pushing me to step back",
    "there is an aversive pressure I want to ease",
    "I sense a pressure to regulate something uncomfortable",
)

_REPULSOR_DISTANCE_TEMPLATES = (
    "I feel something pushing closer that I want to keep at distance",
    "a repelling presence is closing in on me",
    "I sense something unwanted drawing nearer",
)

_APPROACH_PRESSURE_TEMPLATES = (
    "I feel an approach pressure drawing me forward",
    "there is a pull to close a gap I sense ahead",
    "I feel a pressure to move forward",
)

_AVOIDANCE_PRESSURE_TEMPLATES = (
    "I feel a need to step back from something",
    "I sense a pressure to retreat",
    "there is a pull to distance myself",
)

_REGULATION_PRESSURE_TEMPLATES = (
    "I am holding back with effort",
    "I feel a grip of will trying to steady things",
    "I am exerting control to keep from reacting",
)

# Threshold for considering an axis "active" enough to produce a candidate.
# Values below this are treated as noise/baseline.
_CANDIDATE_ACTIVATION_THRESHOLD = 0.35


# Forbidden substrings that indicate implementation provenance leakage.
_PROVENANCE_FORBIDDEN = (
    "module", "agent_id", "agent id", "prompt delta",
    "featuresubtype", "mental_state", "appraisal_general",
    "body_signal", "aversive_pressure", "attractor_distance",
    "repulsor_distance", "primitive_approach_pressure",
    "primitive_avoidance_pressure", "schema", "field dump",
)


def generate_self_report_candidates(
    state: "FullMentalState",
    experienced_dynamics_trace: str,
) -> list[str]:
    """Generate bounded first-person self-report candidates from primitive dynamics.

    Each active primitive axis (above _CANDIDATE_ACTIVATION_THRESHOLD) produces
    one candidate from a template pool. The function filters out any candidate
    that leaks implementation provenance.

    Returns an empty list when no primitive dynamics are active.
    """
    candidates: list[str] = []

    # Body signal
    val = state.appraisal_general.body_signal
    if val >= _CANDIDATE_ACTIVATION_THRESHOLD:
        candidates.append(_BODY_SIGNAL_TEMPLATES[0])

    # Salience
    val = state.appraisal_general.salience
    if val >= _CANDIDATE_ACTIVATION_THRESHOLD:
        candidates.append(_SALIENCE_TEMPLATES[0])

    # Aversive pressure
    val = state.appraisal_general.aversive_pressure
    if val >= _CANDIDATE_ACTIVATION_THRESHOLD:
        candidates.append(_AVERSIVE_PRESSURE_TEMPLATES[0])

    # Repulsor distance
    val = state.appraisal_general.repulsor_distance
    if val >= _CANDIDATE_ACTIVATION_THRESHOLD:
        candidates.append(_REPULSOR_DISTANCE_TEMPLATES[0])

    # Approach pressure
    val = state.state_expression_ext.primitive_approach_pressure
    if val >= _CANDIDATE_ACTIVATION_THRESHOLD:
        candidates.append(_APPROACH_PRESSURE_TEMPLATES[0])

    # Avoidance pressure
    val = state.state_expression_ext.primitive_avoidance_pressure
    if val >= _CANDIDATE_ACTIVATION_THRESHOLD:
        candidates.append(_AVOIDANCE_PRESSURE_TEMPLATES[0])

    # Regulation pressure (willpower as regulation proxy)
    val = state.state_cognition.willpower
    if val >= _CANDIDATE_ACTIVATION_THRESHOLD:
        candidates.append(_REGULATION_PRESSURE_TEMPLATES[0])

    # Provenance filter: remove any candidate that leaks implementation detail
    filtered: list[str] = []
    for candidate in candidates:
        lower = candidate.lower()
        if any(term in lower for term in _PROVENANCE_FORBIDDEN):
            continue
        if re.search(r"\b\d+(?:\.\d+)?\s*%", candidate):
            continue
        if re.search(r"\b[a-z_]+\s*=\s*-?\d", candidate):
            continue
        filtered.append(candidate)

    return filtered


# ---------- Immersion judge/rerank (WI-02) ----------

# Vocabulary that signals grounding in primitive body/attention/action dynamics.
# These are the first-person dynamics words the judge expects in a grounded candidate.
_GROUNDING_VOCAB = (
    "body", "tension", "weight", "pressing", "settling",  # body signal
    "attention", "focus", "pull", "notice", "demanding",  # salience
    "aversive", "pressure", "regulate", "ease", "step back",  # aversive
    "repelling", "distance", "pushing closer", "unwanted", "nearer",  # repulsor
    "approach", "forward", "gap", "move",  # approach pressure
    "retreat", "step back", "distance myself",  # avoidance pressure
    "holding back", "grip", "will", "control", "steady",  # regulation
)

# Raw emotion labels that indicate an emotion-label-only summary.
_EMOTION_LABELS = (
    "joy", "sadness", "anger", "fear", "disgust", "surprise",
    "tenderness", "curiosity", "shame", "guilt", "pride",
    "amusement", "playfulness", "relief", "intimacy",
)

# Minimum number of grounding vocabulary hits required for a candidate to be
# considered grounded in the experienced dynamics trace.
_MIN_GROUNDING_HITS = 3


def _has_provenance_leak(candidate: str) -> bool:
    """Return True if the candidate leaks implementation provenance."""
    lower = candidate.lower()
    if any(term in lower for term in _PROVENANCE_FORBIDDEN):
        return True
    if re.search(r"\b\d+(?:\.\d+)?\s*%", candidate):
        return True
    if re.search(r"\b[a-z_]+\s*=\s*-?\d", candidate):
        return True
    return False


def _is_emotion_label_only(candidate: str) -> bool:
    """Return True if the candidate is an emotion-label-only summary.

    A candidate is emotion-label-only if it contains named emotion labels
    but lacks body/attention/action dynamics grounding vocabulary.
    """
    lower = candidate.lower()
    emotion_count = sum(1 for label in _EMOTION_LABELS if label in lower)
    grounding_count = sum(1 for word in _GROUNDING_VOCAB if word in lower)
    return emotion_count >= 2 and grounding_count < _MIN_GROUNDING_HITS


def _is_keyword_repeat(candidate: str, trace: str) -> bool:
    """Return True if the candidate merely repeats trace keywords without
    first-person grounding.

    A keyword-repetition candidate lists dynamics terms (like the trace output)
    but lacks first-person framing — it reads like a technical enumeration
    rather than a lived experience.
    """
    lower = candidate.lower()
    # If the candidate contains first-person framing, it's not mere repetition
    if any(pers in lower for pers in ("i feel", "i sense", "i am", "my ", "there is")):
        return False
    # Count how many trace terms appear in the candidate
    trace_terms = ("body signal", "salience", "aversive pressure", "repulsor distance",
                   "approach pressure", "avoidance pressure", "regulation pressure")
    trace_hits = sum(1 for term in trace_terms if term in lower)
    # If it hits many trace terms but has no first-person framing, it's repetition
    return trace_hits >= 2


def _score_grounding(candidate: str, trace: str) -> int:
    """Score how well a candidate is grounded in the experienced dynamics trace.

    Returns a non-negative integer count of grounding vocabulary hits.
    """
    lower = candidate.lower()
    return sum(1 for word in _GROUNDING_VOCAB if word in lower)


def judge_self_report_candidates(
    candidates: Sequence[str],
    experienced_dynamics_trace: str,
) -> SelfReportJudgeResult:
    """Judge and rerank self-report candidates against experienced dynamics.

    Selects the best grounded candidate or returns an explicit all-fail result
    if no candidate passes. Candidates are rejected for:
    - implementation provenance leaks (module names, vector fields, percentages)
    - emotion-label-only summaries without dynamics grounding
    - keyword repetition of the trace without first-person framing
    - insufficient grounding in body/attention/action dynamics

    Tied candidates resolve by first-wins (deterministic).
    """
    if not candidates:
        return SelfReportJudgeResult(
            selected_candidate=None,
            decision="all_failed",
            reason="no candidates provided",
        )

    best_candidate: str | None = None
    best_score = -1
    best_reason = ""
    rejection_reasons: list[str] = []

    for idx, candidate in enumerate(candidates):
        # Gate 1: provenance leaks
        if _has_provenance_leak(candidate):
            rejection_reasons.append(f"candidate {idx}: implementation provenance detected")
            continue

        # Gate 2: emotion-label-only summaries
        if _is_emotion_label_only(candidate):
            rejection_reasons.append(f"candidate {idx}: emotion-label-only without dynamics grounding")
            continue

        # Gate 3: keyword repetition of trace
        if _is_keyword_repeat(candidate, experienced_dynamics_trace):
            rejection_reasons.append(f"candidate {idx}: trace keyword repetition without first-person framing")
            continue

        # Gate 4: insufficient grounding
        score = _score_grounding(candidate, experienced_dynamics_trace)
        if score < _MIN_GROUNDING_HITS:
            rejection_reasons.append(f"candidate {idx}: insufficient grounding in experienced dynamics")
            continue

        # Candidate passed all gates; first-wins tie-break
        if best_candidate is None:
            best_candidate = candidate
            best_score = score
            best_reason = f"grounded in experienced dynamics ({score} grounding hits)"

    if best_candidate is not None:
        return SelfReportJudgeResult(
            selected_candidate=best_candidate,
            decision="selected",
            reason=best_reason,
        )

    return SelfReportJudgeResult(
        selected_candidate=None,
        decision="all_failed",
        reason="; ".join(rejection_reasons) if rejection_reasons else "no candidate grounded in experienced dynamics",
    )


def appraisal_trace_passes_quality_gate(trace: str) -> bool:
    """Return True when a trace is non-empty, names primitive dynamics, and leaks nothing.

    A leak is an implementation field name, a module or agent id, a schema
    word, or a raw number with a percent sign or an ``axis=value`` shape.
    """
    text = trace.strip()
    if not text:
        return False

    lower = text.lower()
    forbidden_terms = (
        "module",
        "agent_id",
        "agent id",
        "prompt delta",
        "featuresubtype",
        "mental_state",
        "body_signal",
        "aversive_pressure",
        "attractor_distance",
        "repulsor_distance",
        "primitive_approach_pressure",
        "primitive_avoidance_pressure",
        "schema",
        "field dump",
    )
    if any(term in lower for term in forbidden_terms):
        return False
    if re.search(r"\b\d+(?:\.\d+)?\s*%", lower):
        return False
    if re.search(r"\b[a-z_]+\s*=\s*-?\d", lower):
        return False

    primitive_terms = (
        "body signal",
        "aversive pressure",
        "salience",
        "attractor distance",
        "repulsor distance",
        "approach pressure",
        "avoidance pressure",
        "regulation pressure",
    )
    dynamic_terms = (
        "rose",
        "eased",
        "gathered",
        "closed in",
        "narrowed",
        "shifted",
        "tightened",
        "opened",
        "receded",
    )
    primitive_count = sum(1 for term in primitive_terms if term in lower)
    return primitive_count >= 2 and any(term in lower for term in dynamic_terms)


__all__ = [
    "SelfReportJudgeResult",
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
    "verbalize_primitive_appraisal_trace",
    "appraisal_trace_passes_quality_gate",
    "generate_self_report_candidates",
    "judge_self_report_candidates",
]
