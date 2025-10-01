# tests/test_emotion_metamorphic.py
import math
import datetime as dt
from dataclasses import dataclass
from typing import Optional

import pytest
from hypothesis import given, settings, strategies as st

# ---- adjust imports to your project layout ----
from pm.mental_state_vectors import (
    FullMentalState,
    AppraisalGeneral, AppraisalSocial,
    StateNeurochemical, StateCore, StateEmotions,
    compute_state_delta,
    _clamp01, _clamp11, PersonalityProfile, make_personality_profile,
)

# ---------- helpers ----------
def ms_with(
    goal_congruence=0.0, certainty=0.5, control_self=0.5, agency_other=0.0,
    novelty=0.0, norm_violation=0.0, bodily_threat=0.0,
    perceived_warmth=0.0, fairness=0.0, inclusion_exclusion=0.0, reciprocity=0.0,
    power_imbalance=0.0, perceived_intimacy=0.0, trust_cues=0.0,
    embarrassment=0.0, admiration=0.0, contempt=0.0,
    dop=0.5, ser=0.5, nor=0.5, oxy=0.5, cor=0.5,
    core_valence=0.0, core_arousal=0.5, core_dom=0.0,
):
    ms = FullMentalState()
    ms.appraisal_general = AppraisalGeneral(
        goal_congruence=goal_congruence, certainty=certainty, control_self=control_self,
        agency_other=agency_other, novelty=novelty, norm_violation=norm_violation,
        bodily_threat=bodily_threat
    )
    ms.appraisal_social = AppraisalSocial(
        perceived_warmth=perceived_warmth, fairness=fairness, inclusion_exclusion=inclusion_exclusion,
        reciprocity=reciprocity, power_imbalance=power_imbalance, perceived_intimacy=perceived_intimacy,
        trust_cues=trust_cues, embarrassment=embarrassment, admiration=admiration, contempt=contempt
    )
    ms.state_neurochemical = StateNeurochemical(
        dopamine=dop, serotonin=ser, noradrenaline=nor, oxytocin=oxy, cortisol=cor
    )
    # StateCore here is "current readout" fed into compute_state_delta for gains
    ms.state_core = StateCore(valence=core_valence, arousal=core_arousal, dominance=core_dom)
    return ms

def emotions_after(ms: FullMentalState):
    res, _delta = compute_state_delta(ms, relationship_entity_id=None)
    return res.state_emotions, res.state_core

# ============ Hypothesis strategies ============
f01 = st.floats(min_value=0.0, max_value=1.0)
fm11 = st.floats(min_value=-1.0, max_value=1.0)

# Wider but safe jitter around a base value in [0,1]
def jitter01(x, eps=0.15):
    lo = float(max(0.0, x - eps))
    hi = float(min(1.0, x + eps))
    return st.floats(min_value=lo, max_value=hi)

def jitter11(x, eps=0.15):
    lo = float(max(-1.0, x - eps))
    hi = float(min(1.0, x + eps))
    return st.floats(min_value=lo, max_value=hi)

# ============ Tests ============

@settings(deadline=None, max_examples=300)
@given(
    goal=st.floats(min_value=-1.0, max_value=1.0),
    ctrl=f01, cert=f01,
    base=st.floats(min_value=-1.0, max_value=1.0),
    step=st.floats(min_value=0.05, max_value=0.25),
)
def test_joy_monotone_in_goal_congruence(goal, ctrl, cert, base, step):
    """Increasing goal_congruence should not decrease joy (ceteris paribus)."""
    ms1 = ms_with(goal_congruence=goal, control_self=ctrl, certainty=cert, core_valence=base, core_arousal=0.6)
    e1, _ = emotions_after(ms1)
    ms2 = ms_with(goal_congruence=max(-1.0, min(1.0, goal + step)), control_self=ctrl, certainty=cert, core_valence=base, core_arousal=0.6)
    e2, _ = emotions_after(ms2)
    assert e2.joy + 1e-6 >= e1.joy  # non-decreasing

@settings(deadline=None, max_examples=300)
@given(
    threat=f01, ctrl=f01, cert=f01,
    dom=fm11, base=fm11,
    step=st.floats(min_value=0.05, max_value=0.25),
)
def test_fear_increases_with_threat_or_low_control(threat, ctrl, cert, dom, base, step):
    """Higher bodily_threat or lower control_self should not reduce fear."""
    # threat up
    ms1 = ms_with(bodily_threat=threat, control_self=ctrl, certainty=cert, core_valence=base, core_dom=-abs(dom), core_arousal=0.8)
    e1, _ = emotions_after(ms1)
    ms2 = ms_with(bodily_threat=min(1.0, threat + step), control_self=ctrl, certainty=cert, core_valence=base, core_dom=-abs(dom), core_arousal=0.8)
    e2, _ = emotions_after(ms2)
    assert e2.fear + 1e-6 >= e1.fear

    # control down
    ms3 = ms_with(bodily_threat=threat, control_self=ctrl, certainty=cert, core_valence=base, core_dom=-abs(dom), core_arousal=0.8)
    e3, _ = emotions_after(ms3)
    ms4 = ms_with(bodily_threat=threat, control_self=max(0.0, ctrl - step), certainty=cert, core_valence=base, core_dom=-abs(dom), core_arousal=0.8)
    e4, _ = emotions_after(ms4)
    assert e4.fear + 1e-6 >= e3.fear

@settings(deadline=None, max_examples=300)
@given(
    nv=f01, agency=f01, aro=f01,
    dom=fm11, base=fm11,
    step=st.floats(min_value=0.05, max_value=0.25),
)
def test_anger_rises_with_norm_violation_agency_other_under_high_arousal(nv, agency, aro, dom, base, step):
    """Anger should not decrease when norm_violation & agency_other increase with high arousal and positive dominance."""
    aro = max(aro, 0.6)
    dom = abs(dom)  # positive dominance
    ms1 = ms_with(norm_violation=nv, agency_other=agency, core_arousal=aro, core_dom=dom, core_valence=min(0.0, base))
    e1, _ = emotions_after(ms1)
    ms2 = ms_with(norm_violation=min(1.0, nv + step), agency_other=min(1.0, agency + step), core_arousal=aro, core_dom=dom, core_valence=min(0.0, base))
    e2, _ = emotions_after(ms2)
    assert e2.anger + 1e-6 >= e1.anger

@settings(deadline=None, max_examples=300)
@given(
    warmth=f01, oxy=f01, base=fm11, step=st.floats(min_value=0.05, max_value=0.25)
)
def test_tenderness_monotone_in_warmth_and_oxytocin(warmth, oxy, base, step):
    """Tenderness should not decrease when warmth and oxytocin increase (with positive/neutral valence, low arousal)."""
    val = max(0.0, base)
    ms1 = ms_with(perceived_warmth=warmth, core_valence=val, core_arousal=0.3,
                  dop=0.5, ser=0.6, oxy=oxy)
    e1, _ = emotions_after(ms1)
    ms2 = ms_with(perceived_warmth=min(1.0, warmth + step), core_valence=val, core_arousal=0.3,
                  dop=0.5, ser=0.6, oxy=min(1.0, oxy + step))
    e2, _ = emotions_after(ms2)
    assert e2.tenderness + 1e-6 >= e1.tenderness

@settings(deadline=None, max_examples=300)
@given(
    embar=f01, nv=f01, dom=fm11, base=fm11,
    step=st.floats(min_value=0.05, max_value=0.25)
)
def test_shame_increases_with_embarrassment_low_dominance(embar, nv, dom, base, step):
    """Shame ↑ with embarrassment & norm_violation when dominance is low/negative."""
    ms1 = ms_with(embarrassment=embar, norm_violation=nv, core_valence=min(0.0, base), core_dom=-abs(dom), core_arousal=0.4)
    e1, _ = emotions_after(ms1)
    ms2 = ms_with(embarrassment=min(1.0, embar + step), norm_violation=min(1.0, nv + step),
                  core_valence=min(0.0, base), core_dom=-abs(dom), core_arousal=0.4)
    e2, _ = emotions_after(ms2)
    assert e2.shame + 1e-6 >= e1.shame

@settings(deadline=None, max_examples=300)
@given(
    nv=f01, fairness=f01, ser=f01, base=fm11,
    step=st.floats(min_value=0.05, max_value=0.25)
)
def test_guilt_increases_with_norm_violation_and_fairness(nv, fairness, ser, base, step):
    """Guilt should not decrease when norm_violation and fairness increase."""
    ms1 = ms_with(norm_violation=nv, fairness=fairness, core_valence=min(0.0, base), core_arousal=0.3, ser=ser)
    e1, _ = emotions_after(ms1)
    ms2 = ms_with(norm_violation=min(1.0, nv + step), fairness=min(1.0, fairness + step),
                  core_valence=min(0.0, base), core_arousal=0.3, ser=ser)
    e2, _ = emotions_after(ms2)
    assert e2.guilt + 1e-6 >= e1.guilt

@settings(deadline=None, max_examples=300)
@given(
    nv=f01, threat=f01, cert=f01, ctrl=f01, dop=f01,
    step=st.floats(min_value=0.05, max_value=0.25)
)
def test_amusement_benign_violation_gate(nv, threat, cert, ctrl, dop, step):
    """If threat low & resolution high: more norm_violation shouldn't reduce amusement;
       if threat high: more norm_violation shouldn't increase amusement."""
    # benign regime
    threat_b = min(threat, 0.2)
    cert_b = max(cert, 0.6)
    ctrl_b = max(ctrl, 0.6)

    ms1 = ms_with(norm_violation=nv, bodily_threat=threat_b, certainty=cert_b, control_self=ctrl_b, dop=dop)
    e1, _ = emotions_after(ms1)
    ms2 = ms_with(norm_violation=min(1.0, nv + step), bodily_threat=threat_b, certainty=cert_b, control_self=ctrl_b, dop=dop)
    e2, _ = emotions_after(ms2)
    assert e2.amusement + 1e-6 >= e1.amusement

    # non-benign regime (high threat)
    threat_h = max(threat, 0.8)
    ms3 = ms_with(norm_violation=nv, bodily_threat=threat_h, certainty=cert_b, control_self=ctrl_b, dop=dop)
    f1, _ = emotions_after(ms3)
    ms4 = ms_with(norm_violation=min(1.0, nv + step), bodily_threat=threat_h, certainty=cert_b, control_self=ctrl_b, dop=dop)
    f2, _ = emotions_after(ms4)
    assert f2.amusement <= f1.amusement + 1e-6  # should not increase when threat is high

@settings(deadline=None, max_examples=200)
@given(
    x=f01, nv=f01, ag=f01, aro=f01, base=fm11
)
def test_anger_fear_dominance_symmetry(x, nv, ag, aro, base):
    """With same appraisals, positive dominance should favor anger over fear; negative dominance favors fear."""
    aro = max(aro, 0.6)
    # negative valence baseline to allow anger/fear activation
    val = min(0.0, base)

    ms_pos = ms_with(
        norm_violation=nv, agency_other=ag, core_arousal=aro, core_dom=abs(x), core_valence=val,
        bodily_threat=x*0.0 + 0.4, control_self=0.5, certainty=0.5
    )
    e_pos, _ = emotions_after(ms_pos)

    ms_neg = ms_with(
        norm_violation=nv, agency_other=ag, core_arousal=aro, core_dom=-abs(x), core_valence=val,
        bodily_threat=0.4, control_self=0.5, certainty=0.5
    )
    e_neg, _ = emotions_after(ms_neg)

    assert e_pos.anger + 1e-6 >= e_neg.anger - 1e-6
    assert e_neg.fear  + 1e-6 >= e_pos.fear  - 1e-6

@settings(deadline=None, max_examples=200)
@given(
    # small perturbations to check smoothness (no wild jumps)
    goal=jitter11(0.0, 0.2), nv=jitter01(0.2, 0.2), threat=jitter01(0.2, 0.2),
    ctrl=jitter01(0.5, 0.2), warmth=jitter01(0.5, 0.2), dom=jitter11(0.0, 0.2),
    eps=st.floats(min_value=1e-3, max_value=5e-2)
)
def test_outputs_bounded_and_smooth(goal, nv, threat, ctrl, warmth, dom, eps):
    """All outputs bounded; small input nudges → small output changes."""
    ms1 = ms_with(goal_congruence=goal, norm_violation=nv, bodily_threat=threat,
                  control_self=ctrl, perceived_warmth=warmth, core_dom=dom, core_arousal=0.6)
    e1, c1 = emotions_after(ms1)
    ms2 = ms_with(goal_congruence=_clamp11(goal + eps), norm_violation=_clamp01(nv + eps),
                  bodily_threat=_clamp01(threat + eps), control_self=_clamp01(ctrl + eps),
                  perceived_warmth=_clamp01(warmth + eps), core_dom=_clamp11(dom + eps), core_arousal=0.6)
    e2, c2 = emotions_after(ms2)

    # bounds
    for val in [e1.joy, e1.sadness, e1.anger, e1.fear, e1.disgust, e1.surprise,
                e2.joy, e2.sadness, e2.anger, e2.fear, e2.disgust, e2.surprise]:
        assert 0.0 - 1e-9 <= val <= 1.0 + 1e-9
    for v in [c1.valence, c2.valence, c1.dominance, c2.dominance]:
        assert -1.0 - 1e-9 <= v <= 1.0 + 1e-9
    for a in [c1.arousal, c2.arousal]:
        assert 0.0 - 1e-9 <= a <= 1.0 + 1e-9

    # smoothness (Lipschitz-ish): small input => output change limited
    def dist(a, b): return abs(a - b)
    assert dist(e1.joy, e2.joy)       <= 0.6
    assert dist(e1.anger, e2.anger)   <= 0.6
    assert dist(e1.fear, e2.fear)     <= 0.6
    assert dist(e1.tenderness, e2.tenderness) <= 0.6

# ----- helpers -----
def ms_base(
    *,
    goal=0.0, certainty=0.7, control=0.6, agency=0.4,
    novelty=0.2, nv=0.1, threat=0.1,
    warmth=0.6, fairness=0.6, trustc=0.6, contempt=0.0, power_gap=0.0, intimacy_cue=0.5,
    dop=0.5, ser=0.6, nor=0.5, oxy=0.6, cor=0.4,
    val=0.0, aro=0.6, dom=0.2,
    pers: PersonalityProfile = None,
    prior_intimacy=0.0,
):
    ms = FullMentalState()
    ms.appraisal_general = AppraisalGeneral(
        goal_congruence=goal, certainty=certainty, control_self=control,
        agency_other=agency, novelty=novelty, norm_violation=nv, bodily_threat=threat
    )
    ms.appraisal_social = AppraisalSocial(
        perceived_warmth=warmth, fairness=fairness, inclusion_exclusion=0.5, reciprocity=0.5,
        power_imbalance=power_gap, perceived_intimacy=intimacy_cue,
        trust_cues=trustc, embarrassment=0.0, admiration=0.0, contempt=contempt
    )
    ms.state_neurochemical = StateNeurochemical(dopamine=dop, serotonin=ser, noradrenaline=nor, oxytocin=oxy, cortisol=cor)
    ms.state_core = StateCore(valence=val, arousal=aro, dominance=dom)
    # IMPORTANT: romantic_affection uses *current_ms.state_emotions.intimacy_arousal*
    ms.state_emotions = StateEmotions(intimacy_arousal=_clamp01(prior_intimacy))
    ms.state_personality = pers or make_personality_profile()
    return ms

def get_rel(state):
    # Handle current typo: code sets state_result.state_relationship (plural) dynamically.
    # If it exists, use it; else use state_relationship.
    return getattr(state, "state_relationship", None) or getattr(state, "state_relationship", None)

# ============== TESTS ==============

def test_assertive_vs_confrontive_by_relationship_and_personality():
    """
    High trust/warmth + assertive trait → assertive_push > confrontive_push.
    Low trust + high tension → confrontive_push rises and can dominate.
    """
    # Persona: high extraversion/bluntness (assertive), low BIS
    pers = make_personality_profile(big5=(0.6,0.6,0.8,0.7,0.4), blunt=0.8, bas_drive=0.7, bis=0.3)

    # Friendly context (high trust/warmth), mild violation
    ms1 = ms_base(
        goal=0.3, nv=0.05, threat=0.0, warmth=0.8, trustc=0.85, fairness=0.8, contempt=0.0,
        val=0.1, aro=0.7, dom=0.3, pers=pers
    )
    s1, _ = compute_state_delta(ms1, relationship_entity_id=1)
    expr1 = s1.state_expression_ext
    assert expr1.assertive_push >= expr1.confrontive_push - 1e-6

    # Adversarial context: trust low, tension high (violation + contempt)
    ms2 = ms_base(
        goal=-0.2, nv=0.8, threat=0.1, warmth=0.2, trustc=0.1, fairness=0.2, contempt=0.8,
        val=-0.2, aro=0.8, dom=0.4, pers=pers
    )
    s2, _ = compute_state_delta(ms2, relationship_entity_id=1)
    expr2 = s2.state_expression_ext

    # Confrontive push should increase, assertive push should fall
    assert expr2.confrontive_push >= expr1.confrontive_push - 1e-6
    assert expr2.assertive_push <= expr1.assertive_push + 1e-6


def test_appeasement_scales_with_anxiety_and_tension_low_trust():
    """
    Appeasement increases with attachment anxiety (personality) and tension/low trust (relationship).
    """
    # Same situation, change personality anxiety
    pers_low = make_personality_profile(attach_anx=0.2, affiliation=0.5)
    pers_high = make_personality_profile(attach_anx=0.8, affiliation=0.8)

    # Low trust, high violation ⇒ tension
    ms_low = ms_base(pers=pers_low, nv=0.7, warmth=0.2, trustc=0.1, fairness=0.2, contempt=0.6, val=-0.2, aro=0.7, dom=-0.1)
    s_low, _ = compute_state_delta(ms_low, relationship_entity_id=42)
    expr_low = s_low.state_expression_ext

    ms_high = ms_base(pers=pers_high, nv=0.7, warmth=0.2, trustc=0.1, fairness=0.2, contempt=0.6, val=-0.2, aro=0.7, dom=-0.1)
    s_high, _ = compute_state_delta(ms_high, relationship_entity_id=42)
    expr_high = s_high.state_expression_ext

    assert expr_high.appeasement >= expr_low.appeasement - 1e-6


def test_disengage_silence_increases_with_avoidance_low_trust_aversion():
    """
    Disengage/silence rises with avoidant style, low trust, and aversive association.
    """
    # Set strong avoidant bias
    p1 = make_personality_profile(attach_avoid=0.8, bas_drive=0.3, bis=0.7)
    # Low trust + high aversion: use low trust_cues/fairness & high contempt to build aversion
    ms = ms_base(pers=p1, nv=0.6, warmth=0.2, trustc=0.05, fairness=0.1, contempt=0.9, val=-0.1, aro=0.4, dom=-0.2)
    s, _ = compute_state_delta(ms, relationship_entity_id=7)
    expr = s.state_expression_ext
    assert expr.disengage_silence >= 0.5  # should be pretty strong in this setup


def test_display_amplitude_tracks_arousal_dopamine_and_playfulness():
    """
    Higher arousal/dopamine and playful mood should increase display_amplitude.
    """
    # Baseline
    ms0 = ms_base(val=0.0, aro=0.3, dop=0.4)
    s0, _ = compute_state_delta(ms0, relationship_entity_id=1)
    amp0 = s0.state_expression_ext.display_amplitude

    # Higher arousal + dopamine + novelty → more playfulness → more amplitude
    ms1 = ms_base(val=0.1, aro=0.8, dop=0.8, novelty=0.8, nv=0.1)
    s1, _ = compute_state_delta(ms1, relationship_entity_id=1)
    amp1 = s1.state_expression_ext.display_amplitude

    assert amp1 >= amp0 - 1e-6


# ---------- Relationship emotions ----------

def test_romantic_affection_monotone_in_intimacy_when_affection_positive():
    """
    romantic_affection = affection * (0.5 + 0.5 * prior_intimacy)
    So for positive affection, increasing prior_intimacy increases romantic_affection.
    """
    # Positive affection setup: warm, fair context
    ms_lo = ms_base(warmth=0.9, fairness=0.9, trustc=0.9, nv=0.0, val=0.2, aro=0.4, prior_intimacy=0.1)
    s_lo, _ = compute_state_delta(ms_lo, relationship_entity_id=99)
    rel_lo = get_rel(s_lo)
    assert rel_lo is not None
    ra_lo = rel_lo.romantic_affection

    ms_hi = ms_base(warmth=0.9, fairness=0.9, trustc=0.9, nv=0.0, val=0.2, aro=0.4, prior_intimacy=0.8)
    s_hi, _ = compute_state_delta(ms_hi, relationship_entity_id=99)
    rel_hi = get_rel(s_hi)
    ra_hi = rel_hi.romantic_affection

    assert ra_hi >= ra_lo - 1e-6


def test_trust_increases_with_trust_cues_fairness_and_drops_with_violation():
    """
    trust = f(trust_cues, fairness) - k*norm_violation  (with oxytocin gain)
    """
    # High trust cues/fairness, low violation
    ms_good = ms_base(trustc=0.9, fairness=0.9, nv=0.0, val=0.1, aro=0.5)
    s_good, _ = compute_state_delta(ms_good, relationship_entity_id=5)
    rel_good = get_rel(s_good)
    t_good = rel_good.trust

    # Lower trust cues/fairness, higher violation
    ms_bad = ms_base(trustc=0.2, fairness=0.2, nv=0.8, val=-0.1, aro=0.6)
    s_bad, _ = compute_state_delta(ms_bad, relationship_entity_id=5)
    rel_bad = get_rel(s_bad)
    t_bad = rel_bad.trust

    assert t_good >= t_bad - 1e-6


def test_conflict_tension_mixes_hostile_affect_and_relief_lowers_it():
    """
    conflict_tension ~ 0.6*anger + 0.5*disgust + 0.4*contempt - 0.5*relief - 0.3*social_laughter
    """
    # Build a hostile setup: violation + contempt, low resolution
    ms_hostile = ms_base(nv=0.9, contempt=0.9, trustc=0.1, fairness=0.1, threat=0.1, certainty=0.3, control=0.3, val=-0.2, aro=0.8)
    s_h, _ = compute_state_delta(ms_hostile, relationship_entity_id=2)
    rel_h = get_rel(s_h)
    ct_h = rel_h.conflict_tension

    # Same but add resolution cues: higher certainty/control (→ relief up)
    ms_relief = ms_base(nv=0.9, contempt=0.9, trustc=0.1, fairness=0.1, threat=0.0, certainty=0.9, control=0.9, val=-0.1, aro=0.7)
    s_r, _ = compute_state_delta(ms_relief, relationship_entity_id=2)
    rel_r = get_rel(s_r)
    ct_r = rel_r.conflict_tension

    assert ct_h >= ct_r - 1e-6


def test_communication_warmth_rises_with_tenderness_joy_and_falls_with_anger_disgust():
    """
    communication_warmth = (0.6*tenderness + 0.4*joy) - 0.4*anger - 0.3*disgust (clamped)
    """
    # Warm case: goal-congruent, warm social cues → tenderness/joy up
    ms_warm = ms_base(goal=0.6, warmth=0.9, trustc=0.8, fairness=0.8, nv=0.0, threat=0.0, val=0.3, aro=0.4)
    s_w, _ = compute_state_delta(ms_warm, relationship_entity_id=3)
    rel_w = get_rel(s_w)
    cw_w = rel_w.communication_warmth

    # Cold/hostile case: violation & disgust
    ms_cold = ms_base(goal=-0.3, warmth=0.2, trustc=0.2, fairness=0.2, nv=0.9, threat=0.2, val=-0.3, aro=0.8)
    s_c, _ = compute_state_delta(ms_cold, relationship_entity_id=3)
    rel_c = get_rel(s_c)
    cw_c = rel_c.communication_warmth

    assert cw_w >= cw_c - 1e-6
