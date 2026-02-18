import datetime as dt
from typing import List, Tuple

import pytest

from pm.data_structures import Feature, FeatureType
from pm.mental_state_vectors import (
    AppraisalGeneral,
    StateCore,
    VectorModelReservedSize,
    _collect_axis_bounds,
    _features_to_history,
    compute_partner_expectation_profile,
    create_empty_state,
    ema_baselined_normalize,
)


def _make_vec() -> List[float]:
    return [0.0] * VectorModelReservedSize


def _put(v: List[float], idx: int, val: float) -> List[float]:
    v[idx] = val
    return v


def _hist_from_pairs(pairs: List[Tuple[int, List[float]]], start=None):
    if start is None:
        start = dt.datetime(2025, 1, 1, 12, 0, 0)
    return [(start + dt.timedelta(seconds=s), vec) for s, vec in pairs]


IDX_APPRAISAL_GOAL = AppraisalGeneral.model_fields["goal_congruence"].json_schema_extra["vector_position"]
IDX_CORE_VAL = StateCore.model_fields["valence"].json_schema_extra["vector_position"]
IDX_CORE_ARO = StateCore.model_fields["arousal"].json_schema_extra["vector_position"]


@pytest.fixture(scope="module")
def axis_bounds():
    return _collect_axis_bounds()


def test_axis_bounds_contains_known_fields(axis_bounds):
    ge_ar, le_ar = axis_bounds[IDX_CORE_ARO]
    ge_va, le_va = axis_bounds[IDX_CORE_VAL]
    assert ge_ar == 0.0 and le_ar == 1.0
    assert ge_va == -1.0 and le_va == 1.0


def test_ema_baselined_normalize_basic_shape(axis_bounds):
    v1 = _put(_make_vec(), IDX_CORE_VAL, 0.8)
    v2 = _put(_make_vec(), IDX_CORE_VAL, -0.2)
    out = ema_baselined_normalize(
        history=_hist_from_pairs([(0, v1), (10, v2)]),
        vec_len=VectorModelReservedSize,
        half_life_s=60.0,
        axis_bounds=axis_bounds,
    )
    assert len(out) == VectorModelReservedSize
    assert -1.0 <= out[IDX_CORE_VAL] <= 1.0
    assert -1.0 <= out[IDX_APPRAISAL_GOAL] <= 1.0


def test_features_to_history_supports_feature_schema():
    base = dt.datetime(2025, 1, 1, 9, 0, 0)
    delta = _put(_make_vec(), IDX_CORE_VAL, 0.3)
    f = Feature(
        content="test delta",
        feature_type=FeatureType.Dialogue,
        source="User",
        timestamp_creation=base,
        mental_state_delta=delta,
    )
    history = _features_to_history([f])
    assert len(history) == 1
    ts, vec = history[0]
    assert ts == base
    assert vec[IDX_CORE_VAL] == 0.3


def test_compute_partner_expectation_profile_directionality():
    base = dt.datetime(2025, 1, 1, 10, 0, 0)
    pos = Feature(
        content="good",
        feature_type=FeatureType.Dialogue,
        source="User",
        source_entity_id=1,
        timestamp_creation=base,
        affective_valence=0.8,
    )
    neg = Feature(
        content="bad",
        feature_type=FeatureType.Dialogue,
        source="User",
        source_entity_id=1,
        timestamp_creation=base + dt.timedelta(minutes=5),
        affective_valence=-0.6,
    )
    prof = compute_partner_expectation_profile([pos, neg], conversation_partner_entity_id=1)
    assert "baseline_quality" in prof
    assert "recent_quality" in prof
    assert -1.0 <= float(prof["baseline_quality"]) <= 1.0
    assert -1.0 <= float(prof["recent_quality"]) <= 1.0
    assert 0.0 <= float(prof["disappointment_pressure"]) <= 1.0


def test_create_empty_state_roundtrip_vector_size():
    ms = create_empty_state()
    vec = ms.to_list()
    assert len(vec) == VectorModelReservedSize
