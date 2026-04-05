from datetime import datetime

import pytest

from pm.subsystems.context.common_context import plan_context_story_simple
from pm.subsystems.memory.memory_consolidation import MemoryConsolidationConfig
from pm.subsystems.memory.memory_consolidation_test_mode import SyntheticMemoryConsolidator
from test.infrastructure import GenerationScale, TestInfrastructureConfig, build_query_embedding, populate_test_ghost_raw_only


def _build_temporal_split_bundle(days: int) -> dict:
    scale = GenerationScale(
        days=days,
        scenes_per_day=4,
        topical_clusters_per_topic=5,
        temporal_cluster_levels=(6, 5, 4, 3, 2, 1),
    )
    config = TestInfrastructureConfig(
        seed=701,
        start_time_utc=datetime(2024, 1, 1, 9, 0),
        scale=scale,
        topics=("music", "travel", "trust_repair", "boundary_fear", "robotics", "gardening"),
    )
    corpus = populate_test_ghost_raw_only(config)
    ghost = corpus.ghost

    consolidator = SyntheticMemoryConsolidator(
        ghost,
        MemoryConsolidationConfig(
            min_event_memory_consolidation_threshold=6,
            min_events_for_processing=3,
            min_split_up_cluster_size=4,
            hierchical_summary_token_limit=300,
            enable_cause_effect_extraction=True,
            duplicate_string_similarity_threshold=0.96,
            duplicate_embedding_similarity_threshold=0.995,
            duplicate_overlap_ratio_threshold=0.9,
        ),
        seed=1701,
        embedding_dim=24,
    )
    consolidator.consolidate_memory_if_needed()

    planner_output, settings, _ = plan_context_story_simple(ghost, build_query_embedding("trust_repair", config), 16000)
    recent_meta = planner_output.metadata["lanes"]["recent"]
    historical_meta = planner_output.metadata["lanes"]["historical"]
    planner_used_tokens = sum(meta["token_budget_used"] for meta in planner_output.metadata["lanes"].values())
    return {
        "planner_output": planner_output,
        "settings": settings,
        "recent_meta": recent_meta,
        "historical_meta": historical_meta,
        "recent_items": planner_output.lane_data["recent"],
        "historical_items": planner_output.lane_data["historical"],
        "planner_used_tokens": planner_used_tokens,
    }


def test_temporal_split_small_history_allows_recent_to_consume_everything() -> None:
    bundle = _build_temporal_split_bundle(days=12)

    assert bundle["recent_items"]
    assert not bundle["historical_items"]
    assert bundle["recent_meta"]["token_budget_used"] > 0
    assert bundle["historical_meta"]["token_budget_used"] == 0
    assert bundle["recent_meta"]["first_timestamp_utc"] is not None
    assert bundle["historical_meta"]["first_timestamp_utc"] is None


def test_temporal_split_medium_history_partitions_recent_and_historical() -> None:
    bundle = _build_temporal_split_bundle(days=30)

    assert bundle["recent_items"]
    assert bundle["historical_items"]
    assert bundle["recent_meta"]["candidate_count"] > 0
    assert bundle["historical_meta"]["candidate_count"] > 0
    assert bundle["recent_meta"]["token_budget_used"] > 1024
    assert bundle["historical_meta"]["token_budget_used"] > 1024
    assert bundle["historical_meta"]["last_timestamp_utc"] < bundle["recent_meta"]["first_timestamp_utc"]
    assert bundle["planner_used_tokens"] >= int(bundle["settings"].token_length * 0.75)

