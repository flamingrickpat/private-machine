import pytest

from pm.subsystems.context.common_context import build_context_story_simple, plan_context_story_simple
from pm.utils.token_utils import get_token_count
from test.infrastructure import GenerationScale, TestInfrastructureConfig, populate_test_ghost


LONG_STORY_SCALE = GenerationScale(days=24, scenes_per_day=4, topical_clusters_per_topic=4, temporal_cluster_levels=(6, 5, 4, 3))
SHORT_STORY_SCALE = GenerationScale(days=4, scenes_per_day=2, topical_clusters_per_topic=1, temporal_cluster_levels=(6, 5))


@pytest.mark.parametrize(
    ("topic_name", "scale"),
    [
        ("cluster_a", LONG_STORY_SCALE),
        ("cluster_b", LONG_STORY_SCALE),
        ("cluster_a", SHORT_STORY_SCALE),
        ("cluster_b", SHORT_STORY_SCALE),
    ],
)
def test_build_context_story_simple(topic_name, scale):
    corpus = populate_test_ghost(
        TestInfrastructureConfig(
            seed=11,
            topics=("cluster_a", "cluster_b"),
            scale=scale,
        )
    )

    planner_output, settings, static_prefix = plan_context_story_simple(corpus.ghost, corpus.query_embeddings[topic_name], total_token_budget=4000)
    story_context = build_context_story_simple(corpus.ghost, corpus.query_embeddings[topic_name])
    other_topic = "cluster_b" if topic_name == "cluster_a" else "cluster_a"
    lane_metadata = planner_output.metadata["lanes"]
    used_tokens = sum(meta["token_budget_used"] for meta in lane_metadata.values())

    assert settings.token_length < 4000
    assert planner_output.metadata["total_token_budget"] == settings.token_length
    assert story_context.startswith(static_prefix)
    assert topic_name in story_context

    recent_lane = planner_output.lane_data["recent"]
    assert recent_lane
    assert any(getattr(item, "type", None).value == "VirtualKnoxel" for item in recent_lane)

    historical_lane = planner_output.lane_data["historical"]
    assert historical_lane
    assert any(topic_name in getattr(item, "content", "") for item in historical_lane)

    narratives = planner_output.lane_data["narratives_ai"] + planner_output.lane_data["narratives_user"]
    assert narratives
    assert any(topic_name in item.content for item in narratives)

    if scale is LONG_STORY_SCALE:
        assert used_tokens >= settings.token_length - 20
        assert get_token_count(story_context) <= 4100
        assert story_context.count(topic_name) > story_context.count(other_topic)
        assert any(meta["token_budget_current"] != meta["token_budget_base"] for meta in lane_metadata.values())
    else:
        assert used_tokens < settings.token_length
        assert get_token_count(story_context) < 2000
        assert any(topic_name in getattr(item, "content", "") for item in recent_lane + historical_lane)
