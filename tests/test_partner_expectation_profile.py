import datetime as dt
from types import SimpleNamespace

from pm.codelets.codelet_definitions import CodeletFamily
from pm.ghosts.procedures.codelets import _compute_state_boosts
from pm.ghosts.procedures.learning import _score_outcome
from pm.mental_state_vectors import (
    FullMentalState,
    StateRelationship,
    VectorModelReservedSize,
    compute_partner_expectation_profile,
)


def _feature(
    ts: dt.datetime,
    text: str,
    src_id: int = 2,
    *,
    warmth_delta: float = 0.0,
    trust_delta: float = 0.0,
    conflict_delta: float = 0.0,
):
    delta = [0.0] * VectorModelReservedSize
    delta[StateRelationship.model_fields["communication_warmth"].json_schema_extra["vector_position"]] = warmth_delta
    delta[StateRelationship.model_fields["trust"].json_schema_extra["vector_position"]] = trust_delta
    delta[StateRelationship.model_fields["conflict_tension"].json_schema_extra["vector_position"]] = conflict_delta
    return SimpleNamespace(
        timestamp_creation=ts,
        source_entity_id=src_id,
        content=text,
        affective_valence=0.0,
        mental_state_delta=delta,
        metadata={},
        feature_type=SimpleNamespace(name="Dialogue"),
    )


def test_partner_profile_raises_baseline_and_tracks_disappointment():
    base = dt.datetime(2026, 1, 1, 12, 0, 0)
    feats = []
    for i in range(10):
        feats.append(
            _feature(
                base + dt.timedelta(minutes=i),
                "neutral content",
                2,
                warmth_delta=0.30,
                trust_delta=0.20,
                conflict_delta=-0.10,
            )
        )
    feats.append(
        _feature(
            base + dt.timedelta(minutes=11),
            "neutral content",
            2,
            warmth_delta=-0.05,
            trust_delta=-0.10,
            conflict_delta=0.35,
        )
    )

    profile = compute_partner_expectation_profile(feats, conversation_partner_entity_id=2, reference_timeframe_minutes=60)

    assert profile["baseline_quality"] > 0.2
    assert profile["recent_quality"] < profile["baseline_quality"]
    assert profile["disappointment_pressure"] > 0.05


def test_codelet_boosts_use_partner_expectation_pressure():
    ms = FullMentalState()
    ghost = SimpleNamespace(
        current_state=SimpleNamespace(latent_mental_state=ms),
        partner_expectation_profile={"disappointment_pressure": 0.8, "positive_surprise_bias": 0.0},
    )

    boosts = _compute_state_boosts(ghost)
    assert boosts[CodeletFamily.RegulationCoping] > 1.0
    assert boosts[CodeletFamily.Appraisals] > 1.0
    assert boosts[CodeletFamily.Narratives] < 1.0


def test_expectation_scoring_gets_stricter_with_pressure():
    expected = "user confirms concise implementation progress"
    reality = "user confirms progress"

    loose = _score_outcome(expected, reality, strictness=0.0)
    strict = _score_outcome(expected, reality, strictness=1.0)

    assert float(strict["reward"]) <= float(loose["reward"])
