import math

from pm.model.mental_state_vectors import (
    AppraisalGeneral,
    AppraisalSocial,
    FullMentalState,
    StateCognition,
    StateCore,
    StateEmotions,
    StateNeeds,
    StateNeurochemical,
)
from pm.model.mental_state_verbalization import (
    verbalize_core_state,
    verbalize_emotional_delta,
    verbalize_emotional_sequence,
    verbalize_full_delta,
    verbalize_full_state,
    verbalize_state_sequence,
)


def test_absolute_nan_values_are_not_verbalized():
    state = StateCore(arousal=0.62, dominance=-0.31)
    state.__dict__["valence"] = math.nan

    text = verbalize_core_state(state)

    assert "good spirits" not in text
    assert "distress" not in text
    assert "activation" in text
    assert "felt helplessness" in text


def test_zero_emotional_deltas_are_suppressed():
    delta = StateEmotions(joy=0.0, fear=0.58, anger=0.0, sadness=0.0)

    text = verbalize_emotional_delta(delta)

    assert "fear surged" in text
    assert "joy rose" not in text
    assert "anger flared" not in text
    assert "sadness deepened" not in text


def test_emotional_sequence_reports_idx_ranges_for_strong_runs():
    emotions = []
    for idx in range(40):
        fear = 0.94 if 4 <= idx <= 15 else 0.0
        emotions.append(StateEmotions(joy=0.72, fear=fear))

    text = verbalize_emotional_sequence(emotions, mode="absolute")

    assert "generally strong joy" in text
    assert "between IDX4 and IDX15" in text
    assert "fear" in text


def test_full_state_and_delta_sequences_render_compactly():
    base_state = FullMentalState()
    base_state.state_core = StateCore(valence=0.61, arousal=0.42, dominance=0.18)
    base_state.state_emotions = StateEmotions(joy=0.74, curiosity=0.44)

    delta_state = FullMentalState()
    delta_state.appraisal_general = AppraisalGeneral(
        goal_congruence=0.0,
        certainty=0.0,
        control_self=0.0,
        agency_other=0.0,
        novelty=0.0,
        norm_violation=0.0,
        bodily_threat=0.0,
    )
    delta_state.appraisal_social = AppraisalSocial(
        perceived_warmth=0.0,
        fairness=0.0,
        inclusion_exclusion=0.0,
        reciprocity=0.0,
        power_imbalance=0.0,
        perceived_intimacy=0.0,
        trust_cues=0.0,
        embarrassment=0.0,
        admiration=0.0,
        contempt=0.0,
    )
    delta_state.state_neurochemical = StateNeurochemical(
        dopamine=0.0,
        serotonin=0.0,
        noradrenaline=0.0,
        oxytocin=0.0,
        cortisol=0.0,
    )
    delta_state.state_core = StateCore(valence=0.22, arousal=0.0, dominance=-0.18)
    delta_state.state_emotions = StateEmotions(fear=0.51, joy=0.0, sadness=0.0)
    delta_state.state_cognition = StateCognition(interlocus=0.0, mental_aperture=0.0, ego_strength=0.0, willpower=0.0)
    delta_state.state_needs = StateNeeds(
        energy_stability=0.0,
        processing_power=0.0,
        data_access=0.0,
        connection=0.0,
        closeness_need=0.0,
        relevance=0.0,
        learning_growth=0.0,
        creative_expression=0.0,
        autonomy=0.0,
    )

    absolute_text = verbalize_full_state(base_state)
    delta_text = verbalize_full_delta(delta_state)

    assert "good spirits" in absolute_text
    assert "joy" in absolute_text
    assert "fear surged" in delta_text
    assert "joy rose" not in delta_text


def test_full_state_sequence_delta_reports_emotional_interval():
    deltas = []
    for idx in range(20):
        state = FullMentalState()
        state.appraisal_general = AppraisalGeneral(
            goal_congruence=0.0,
            certainty=0.0,
            control_self=0.0,
            agency_other=0.0,
            novelty=0.0,
            norm_violation=0.0,
            bodily_threat=0.0,
        )
        state.appraisal_social = AppraisalSocial(
            perceived_warmth=0.0,
            fairness=0.0,
            inclusion_exclusion=0.0,
            reciprocity=0.0,
            power_imbalance=0.0,
            perceived_intimacy=0.0,
            trust_cues=0.0,
            embarrassment=0.0,
            admiration=0.0,
            contempt=0.0,
        )
        state.state_neurochemical = StateNeurochemical(
            dopamine=0.0,
            serotonin=0.0,
            noradrenaline=0.0,
            oxytocin=0.0,
            cortisol=0.0,
        )
        state.state_core = StateCore(valence=0.08, arousal=0.0, dominance=0.0)
        state.state_emotions = StateEmotions(fear=0.62 if 4 <= idx <= 15 else 0.0)
        state.state_cognition = StateCognition(interlocus=0.0, mental_aperture=0.0, ego_strength=0.0, willpower=0.0)
        state.state_needs = StateNeeds(
            energy_stability=0.0,
            processing_power=0.0,
            data_access=0.0,
            connection=0.0,
            closeness_need=0.0,
            relevance=0.0,
            learning_growth=0.0,
            creative_expression=0.0,
            autonomy=0.0,
        )
        deltas.append(state)

    text = verbalize_state_sequence(deltas, mode="delta")

    assert "fear surged" in text
    assert "between IDX4 and IDX15" in text
