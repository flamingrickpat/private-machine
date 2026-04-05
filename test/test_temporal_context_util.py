from __future__ import annotations

from functools import lru_cache

import pytest

from pm.model.knoxel_common import MemoryClusterKnoxel
from pm.model.knoxel_feature import Feature
from pm.subsystems.context.temporal_context_util import build_temporal_context
from pm.utils.token_utils import get_token_count
from test.infrastructure import GenerationScale, TestInfrastructureConfig, populate_test_ghost
from test.test_infrastructure import MULTI_YEAR_SCALE


CONTEXT_BUDGETS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]


@lru_cache(maxsize=None)
def _build_scenario(name: str):
    if name == "tiny_no_memory":
        corpus = populate_test_ghost(
            TestInfrastructureConfig(
                seed=11,
                scale=GenerationScale(days=2, scenes_per_day=1, topical_clusters_per_topic=1, temporal_cluster_levels=(6,)),
                topics=("music", "travel"),
            )
        )
        memory_ids = {memory.id for memory in corpus.ghost.all_episodic_memories}
        corpus.ghost.all_episodic_memories = []
        corpus.ghost.all_knoxels = {
            knoxel_id: knoxel for knoxel_id, knoxel in corpus.ghost.all_knoxels.items() if knoxel_id not in memory_ids
        }
        return corpus
    if name == "medium":
        return populate_test_ghost(
            TestInfrastructureConfig(
                seed=17,
                scale=GenerationScale(days=12, scenes_per_day=4, topical_clusters_per_topic=3, temporal_cluster_levels=(6, 5, 4)),
                topics=("music", "gardening", "robotics", "travel", "trust_repair"),
            )
        )
    if name == "ultra_months":
        return populate_test_ghost(
            TestInfrastructureConfig(
                seed=31,
                scale=GenerationScale(days=120, scenes_per_day=4, topical_clusters_per_topic=6, temporal_cluster_levels=(6, 5, 4, 3, 2, 1)),
                topics=("music", "travel", "trust_repair", "boundary_fear", "robotics", "gardening"),
            )
        )
    if name == "multi_year":
        return populate_test_ghost(
            TestInfrastructureConfig(
                seed=41,
                scale=MULTI_YEAR_SCALE,
                topics=("music", "travel", "trust_repair", "boundary_fear", "robotics", "gardening"),
            )
        )
    raise AssertionError(name)


@pytest.mark.parametrize("scenario_name", ["tiny_no_memory", "medium", "ultra_months", "multi_year"])
@pytest.mark.parametrize("max_tokens", CONTEXT_BUDGETS)
def test_temporal_context_util_handles_all_budget_tiers(scenario_name: str, max_tokens: int) -> None:
    corpus = _build_scenario(scenario_name)
    if scenario_name == "multi_year" and max_tokens in (4096, 8192):
        with pytest.raises(Exception, match="no exact history cover fits inside max_tokens"):
            build_temporal_context(
                corpus.ghost,
                corpus.query_embeddings["music"],
                ratio=0.3,
                max_tokens=max_tokens,
                require_temporal_history=True,
                require_contiguous_temporal_span=True,
                run_final_checks=True,
            )
        return

    items = build_temporal_context(
        corpus.ghost,
        corpus.query_embeddings["music"],
        ratio=0.3,
        max_tokens=max_tokens,
        require_temporal_history=True,
        require_contiguous_temporal_span=True,
        run_final_checks=True,
    )

    total_tokens = sum(get_token_count(item.get_story_element()) for item in items)
    raw_total_tokens = sum(
        get_token_count(feature.get_story_element())
        for feature in corpus.ghost.all_features
        if feature.causal
    )

    assert items
    assert total_tokens <= max_tokens

    if scenario_name == "tiny_no_memory":
        assert all(isinstance(item, Feature) for item in items)
        return

    if raw_total_tokens <= max_tokens:
        assert all(isinstance(item, Feature) for item in items)
    else:
        assert any(isinstance(item, MemoryClusterKnoxel) for item in items)


@pytest.mark.parametrize(
    ("scenario_name", "small_budget", "large_budget"),
    [
        ("medium", 4096, 524288),
        ("ultra_months", 4096, 524288),
        ("multi_year", 32768, 524288),
    ],
)
def test_temporal_context_util_uses_more_raw_data_as_budget_grows(
    scenario_name: str,
    small_budget: int,
    large_budget: int,
) -> None:
    corpus = _build_scenario(scenario_name)
    small_items = build_temporal_context(
        corpus.ghost,
        corpus.query_embeddings["music"],
        ratio=0.3,
        max_tokens=small_budget,
        run_final_checks=True,
    )
    large_items = build_temporal_context(
        corpus.ghost,
        corpus.query_embeddings["music"],
        ratio=0.3,
        max_tokens=large_budget,
        run_final_checks=True,
    )

    small_raw_tokens = sum(get_token_count(item.get_story_element()) for item in small_items if isinstance(item, Feature))
    large_raw_tokens = sum(get_token_count(item.get_story_element()) for item in large_items if isinstance(item, Feature))
    small_summary_count = sum(1 for item in small_items if isinstance(item, MemoryClusterKnoxel))
    large_summary_count = sum(1 for item in large_items if isinstance(item, MemoryClusterKnoxel))

    assert large_raw_tokens >= small_raw_tokens
    assert large_summary_count <= small_summary_count
