import pytest

from test.test_final_prompt_snapshots import FINAL_PROMPT_SCENARIOS, _build_final_prompt_bundle


@pytest.mark.parametrize(("snapshot_name", "seed", "query_topic", "scale"), FINAL_PROMPT_SCENARIOS)
def test_regression_lane_metadata_never_selects_more_than_candidates(snapshot_name, seed, query_topic, scale) -> None:
    bundle = _build_final_prompt_bundle(snapshot_name, seed, query_topic, scale)

    assert bundle["lane_selection_overflow_count"] == 0


@pytest.mark.parametrize(("snapshot_name", "seed", "query_topic", "scale"), FINAL_PROMPT_SCENARIOS)
def test_regression_large_histories_should_use_most_of_available_budget(snapshot_name, seed, query_topic, scale) -> None:
    bundle = _build_final_prompt_bundle(snapshot_name, seed, query_topic, scale)

    assert bundle["planner_used_tokens"] >= int(bundle["planner_token_budget"] * 0.75)


@pytest.mark.parametrize(("snapshot_name", "seed", "query_topic", "scale"), FINAL_PROMPT_SCENARIOS)
def test_regression_preserve_history_should_have_real_historical_candidates(snapshot_name, seed, query_topic, scale) -> None:
    bundle = _build_final_prompt_bundle(snapshot_name, seed, query_topic, scale)
    historical_meta = bundle["planner_output"].metadata["lanes"]["historical"]

    assert historical_meta["candidate_count"] > 0
    assert historical_meta["token_budget_used"] > 128
