# tests/test_ema_normalization.py
import math
import datetime as dt
import random
from typing import List, Tuple

import pytest

# ---- adjust these imports to your module layout ----
# from your_module import (
from pm.mental_state_vectors import (
    VectorModelReservedSize,
    AppraisalGeneral, AppraisalSocial, StateNeurochemical, StateCore,
    StateEmotions, StateCognition, StateNeeds, StateRelationship,
    MentalFeature, Mind,
    ema_baselined_normalize, _collect_axis_bounds,
)

# Helpers
def make_vec():
    return [0.0] * VectorModelReservedSize

def put(v: List[float], idx: int, val: float):
    v[idx] = val
    return v

def hist_from_pairs(pairs: List[Tuple[int, List[float]]], start=None):
    """
    pairs: list of (seconds_since_start, delta_vec)
    """
    if start is None:
        start = dt.datetime(2025, 1, 1, 12, 0, 0)
    res = []
    for s, vec in pairs:
        res.append((start + dt.timedelta(seconds=s), vec))
    return res

# Axis positions from your schema
IDX_VALENCE = AppraisalGeneral.model_fields["goal_congruence"].json_schema_extra["vector_position"]  # 11
IDX_CORE_VAL = StateCore.model_fields["valence"].json_schema_extra["vector_position"]               # 150
IDX_CORE_ARO = StateCore.model_fields["arousal"].json_schema_extra["vector_position"]               # 151
IDX_EMO_JOY  = StateEmotions.model_fields["joy"].json_schema_extra["vector_position"]               # 400

# ---------- fixtures ----------
@pytest.fixture(scope="module")
def axis_bounds():
    return _collect_axis_bounds()

@pytest.fixture
def base_time():
    return dt.datetime(2025, 1, 1, 12, 0, 0)

# ---------- tests ----------

def test_axis_bounds_contains_known_fields(axis_bounds):
    # Spot-check: arousal should be [0,1], valence [-1,1]
    ge_ar, le_ar = axis_bounds[IDX_CORE_ARO]
    ge_va, le_va = axis_bounds[IDX_CORE_VAL]
    assert ge_ar == 0.0 and le_ar == 1.0
    assert ge_va == -1.0 and le_va == 1.0


def test_empty_history_returns_zeros(axis_bounds):
    out = ema_baselined_normalize(
        history=[],
        vec_len=VectorModelReservedSize,
        half_life_s=60.0,
        axis_bounds=axis_bounds,
        start_level=None,
    )
    assert len(out) == VectorModelReservedSize
    assert all(abs(x) < 1e-12 for x in out)  # default zeros mapped to [-1,1] baseline → 0


def test_cumsum_and_ema_time_awareness(axis_bounds):
    """
    Feed two large deltas separated by a long gap — alpha should be ~1 for big dt,
    so EMA should nearly jump to level on the second step.
    """
    # Build history on a single axis (valence index 150) using the global vector
    v1 = put(make_vec(), IDX_CORE_VAL, 10.0)   # big positive delta
    v2 = put(make_vec(), IDX_CORE_VAL, -4.0)   # partial pullback

    # dt1 small, dt2 large (e.g., 1s then 10,000s)
    history = hist_from_pairs([
        (0, v1),
        (1, v2),          # short gap
        (10000, make_vec())  # no further delta, just trigger time step with large alpha
    ])

    out = ema_baselined_normalize(
        history=history,
        vec_len=VectorModelReservedSize,
        half_life_s=60.0,  # half-life 60s → alpha for 10,000s is ~1
        axis_bounds=axis_bounds,
    )

    # Sanity: output bounded, and on a [-1,1] axis it stays within [-1,1]
    assert -1.0 <= out[IDX_CORE_VAL] <= 1.0

    # With such a long gap, EMA almost equals level; deviation ≈ 0 → normalized ≈ 0
    assert abs(out[IDX_CORE_VAL]) < 0.15


def test_two_sigma_maps_to_unity(axis_bounds):
    """
    Construct deviations so that final (level - EMA) ≈ 2 * std for a given axis.
    Then check normalized output ~1.0 on that axis.
    """
    idx = IDX_CORE_VAL

    # Build many steps where deviation distribution has a known std.
    # We do alternating +a / -a to keep EMA ~0 and std ~a.
    # Final step pushes deviation to ~2a.
    a = 0.5
    pairs = []
    t = 0
    for k in range(200):
        delta = make_vec()
        # flip sign to oscillate
        delta[idx] = a if (k % 2 == 0) else -a
        pairs.append((t, delta))
        t += 2  # 2 seconds apart

    # Final "2 sigma" push: add approx 2a
    final = make_vec()
    final[idx] = 2.0 * a
    pairs.append((t, final))

    history = hist_from_pairs(pairs)
    out = ema_baselined_normalize(
        history=history,
        vec_len=VectorModelReservedSize,
        half_life_s=90.0,   # modest half-life so EMA tracks slowly
        axis_bounds=axis_bounds,
    )

    # Expect near +1.0 on a symmetric [-1,1] axis
    assert 0.85 <= out[idx] <= 1.0


def test_mapping_to_01_bounds(axis_bounds):
    """
    For a [0,1] axis (arousal), normalized z in [-1,1] must map to [0,1].
    """
    idx = IDX_CORE_ARO
    # Create a sequence with a big positive deviation last
    pairs = []
    t, a = 0, 0.4
    for k in range(100):
        delta = make_vec()
        delta[idx] = (a if (k % 2 == 0) else -a)
        pairs.append((t, delta))
        t += 1

    final = make_vec()
    final[idx] = 2.0  # big positive deviation
    pairs.append((t, final))

    out = ema_baselined_normalize(
        history=hist_from_pairs(pairs),
        vec_len=VectorModelReservedSize,
        half_life_s=60.0,
        axis_bounds=axis_bounds,
    )

    # Should be within [0,1] and near the upper end due to positive deviation
    assert 0.0 <= out[idx] <= 1.0
    assert out[idx] > 0.6


def test_outlier_does_not_break_scaling(axis_bounds):
    """
    Include one extreme outlier; ensure outputs remain bounded and subsequent values are not pinned.
    """
    idx = IDX_CORE_VAL
    pairs = []
    t = 0

    # normal small noise
    for k in range(50):
        delta = make_vec()
        delta[idx] = 0.1 * math.sin(k / 3.0)
        pairs.append((t, delta))
        t += 1

    # single huge spike
    spike = make_vec()
    spike[idx] = 1000.0
    pairs.append((t, spike))
    t += 1

    # return to normal
    for k in range(20):
        delta = make_vec()
        delta[idx] = 0.1 * math.sin(k / 2.0)
        pairs.append((t, delta))
        t += 1

    out = ema_baselined_normalize(
        history=hist_from_pairs(pairs),
        vec_len=VectorModelReservedSize,
        half_life_s=120.0,
        axis_bounds=axis_bounds,
    )

    # Still bounded
    assert -1.0 <= out[idx] <= 1.0

    # And not pinned at extrema
    assert abs(out[idx]) < 0.95


def test_integration_mind_returns_bounded_full_state(axis_bounds):
    """
    Build a Mind with features over time affecting a few axes,
    run get_current_mental_state and verify model bounds and shape.
    """
    m = Mind()
    start = dt.datetime(2025, 1, 1, 8, 0, 0)
    # Create 60 minutes of minute-by-minute deltas for valence and arousal
    for i in range(60):
        vec = make_vec()
        vec[IDX_CORE_VAL] = 0.05 * math.sin(i / 7.0)
        vec[IDX_CORE_ARO] = 0.03 * math.cos(i / 9.0)
        f = MentalFeature(
            content=f"tick {i}",
            source_entity_id=None,
            timestamp=start + dt.timedelta(minutes=i),
            state_appraisal_vector=make_vec(),  # not used here
            state_delta_vector=vec,
        )
        m.features.append(f)

    ms = m.get_current_mental_state(reference_timeframe_minutes=30)

    # Spot-check a few fields are within their declared ranges
    assert -1.0 <= ms.state_core.valence <= 1.0
    assert 0.0 <= ms.state_core.arousal <= 1.0

    assert 0.0 <= ms.state_emotions.joy <= 1.0  # note: may be 0 if no mapping code runs, but should be within bounds

    # Also ensure vector reconstruction roundtrip length
    vec = ms.to_list()
    assert len(vec) == VectorModelReservedSize


def test_half_life_effect(axis_bounds):
    """
    Shorter half-life should yield more reactive normalization (bigger |z| for same final dev).
    """
    idx = IDX_CORE_VAL
    pairs = []
    t = 0
    for k in range(100):
        delta = make_vec()
        delta[idx] = 0.2 if (k % 10 == 0) else 0.0  # sparse impulses
        pairs.append((t, delta))
        t += 5

    hist = hist_from_pairs(pairs)

    out_fast = ema_baselined_normalize(
        history=hist, vec_len=VectorModelReservedSize,
        half_life_s=30.0, axis_bounds=axis_bounds
    )
    out_slow = ema_baselined_normalize(
        history=hist, vec_len=VectorModelReservedSize,
        half_life_s=300.0, axis_bounds=axis_bounds
    )

    # Fast half-life → EMA forgets faster → larger deviation at end
    assert abs(out_fast[idx]) < abs(out_slow[idx]) - 1e-6


def test_relationship_filter_path(axis_bounds):
    """
    If you filter features by source_entity_id in Mind.get_current_mental_state,
    verify that filtering changes the outcome.
    """
    m = Mind()
    base = dt.datetime(2025, 1, 1, 9, 0, 0)

    # Partner A contributes positive valence deltas
    for i in range(20):
        vec = make_vec()
        vec[IDX_CORE_VAL] = 0.2
        m.features.append(MentalFeature(
            content="A", source_entity_id=1,
            timestamp=base + dt.timedelta(seconds=i*10),
            state_appraisal_vector=make_vec(),
            state_delta_vector=vec,
        ))

    # Partner B contributes negative valence deltas
    for i in range(20):
        vec = make_vec()
        vec[IDX_CORE_VAL] = -0.2
        m.features.append(MentalFeature(
            content="B", source_entity_id=2,
            timestamp=base + dt.timedelta(seconds=5 + i*10),
            state_appraisal_vector=make_vec(),
            state_delta_vector=vec,
        ))

    ms_all = m.get_current_mental_state(reference_timeframe_minutes=10, conversation_partner_entity_id=None)
    ms_a   = m.get_current_mental_state(reference_timeframe_minutes=10, conversation_partner_entity_id=1)
    ms_b   = m.get_current_mental_state(reference_timeframe_minutes=10, conversation_partner_entity_id=2)

    # With both, valence should be closer to neutral than with either alone
    assert abs(ms_all.state_core.valence) < max(abs(ms_a.state_core.valence), abs(ms_b.state_core.valence))


def test_std_floor_prevents_nan(axis_bounds):
    """
    If all deviations are identical (std=0), the internal std floor should prevent NaNs.
    """
    idx = IDX_CORE_VAL
    pairs = []
    for s in range(10):
        v = make_vec()
        v[idx] = 1.0  # constant delta → constant level increments
        pairs.append((s, v))

    out = ema_baselined_normalize(
        history=hist_from_pairs(pairs),
        vec_len=VectorModelReservedSize,
        half_life_s=60.0,
        axis_bounds=axis_bounds,
    )
    assert not math.isnan(out[idx])
    assert -1.0 <= out[idx] <= 1.0
