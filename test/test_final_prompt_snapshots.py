import os
from collections import Counter
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pytest

from pm.model.knoxel_feature import Feature
from pm.model.knoxel_list import KnoxelList
from pm.subsystems.context.common_context import plan_context_story_simple
from pm.subsystems.context.render_presets import context_render_story_messages_assistant_for_dialoge
from pm.subsystems.memory.memory_consolidation import MemoryConsolidationConfig
from pm.subsystems.memory.memory_consolidation_test_mode import SyntheticMemoryConsolidator
from pm.utils.token_utils import get_token_count
from test.infrastructure import GenerationScale, TestInfrastructureConfig, build_query_embedding, populate_test_ghost_raw_only


SNAPSHOT_DIR = Path(__file__).with_name("test_data")
UPDATE_SNAPSHOTS = os.environ.get("UPDATE_FINAL_PROMPT_SNAPSHOTS") == "1"
TARGET_TOKEN_BUDGET = 16000


FINAL_PROMPT_SCENARIOS = (
    ("final_prompt_test_1.txt", 701, "trust_repair", GenerationScale(days=70, scenes_per_day=4, topical_clusters_per_topic=5, temporal_cluster_levels=(6, 5, 4, 3, 2, 1))),
    ("final_prompt_test_2.txt", 811, "travel", GenerationScale(days=120, scenes_per_day=5, topical_clusters_per_topic=6, temporal_cluster_levels=(6, 5, 4, 3, 2, 1))),
    ("final_prompt_test_3.txt", 907, "boundary_fear", GenerationScale(days=365, scenes_per_day=6, topical_clusters_per_topic=8, temporal_cluster_levels=(6, 5, 4, 3, 2, 1))),
)


@pytest.mark.parametrize(("snapshot_name", "seed", "query_topic", "scale"), FINAL_PROMPT_SCENARIOS)
def test_final_prompt_snapshot(snapshot_name: str, seed: int, query_topic: str, scale: GenerationScale) -> None:
    bundle = _build_final_prompt_bundle(snapshot_name, seed, query_topic, scale)
    _assert_snapshot_text(snapshot_name, bundle["snapshot"])


@pytest.mark.parametrize(("snapshot_name", "seed", "query_topic", "scale"), FINAL_PROMPT_SCENARIOS)
def test_final_prompt_quality_checks(snapshot_name: str, seed: int, query_topic: str, scale: GenerationScale) -> None:
    bundle = _build_final_prompt_bundle(snapshot_name, seed, query_topic, scale)

    assert bundle["raw_token_count"] >= 20000
    assert bundle["target_token_budget"] == TARGET_TOKEN_BUDGET
    assert bundle["recent_items"]
    assert bundle["historical_items"]
    assert bundle["recent_max_end"] >= bundle["global_max_end"] - timedelta(hours=12)
    assert bundle["historical_max_end"] <= bundle["recent_min_begin"]
    assert bundle["adjacent_same_role_pairs"] <= 1
    assert bundle["assistant_turn_count"] >= 2
    assert bundle["user_turn_count"] >= 2
    assert abs(bundle["assistant_turn_count"] - bundle["user_turn_count"]) <= 2
    assert bundle["rendered_token_count"] >= 5000
    assert bundle["rendered_token_count"] <= TARGET_TOKEN_BUDGET + 3500
    assert bundle["planner_used_tokens"] <= bundle["planner_token_budget"] + 64


@lru_cache(maxsize=None)
def _build_final_prompt_bundle(snapshot_name: str, seed: int, query_topic: str, scale: GenerationScale) -> dict:
    config = TestInfrastructureConfig(
        seed=seed,
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
        seed=seed + 1000,
        embedding_dim=24,
    )
    consolidator.consolidate_memory_if_needed()

    query_embedding = build_query_embedding(query_topic, config)
    planner_output, settings, _ = plan_context_story_simple(ghost, query_embedding, TARGET_TOKEN_BUDGET)
    turns = context_render_story_messages_assistant_for_dialoge(ghost, planner_output)

    causal_features = sorted([feature for feature in ghost.all_features if feature.causal], key=lambda item: (item.timestamp_world_begin, item.id))
    raw_story = KnoxelList(causal_features).get_story()
    raw_token_count = get_token_count(raw_story)
    rendered_text = _render_turns_as_text(turns)
    rendered_token_count = get_token_count(rendered_text)

    recent_items = [item for item in planner_output.lane_data["recent"] if hasattr(item, "timestamp_world_begin")]
    historical_items = [item for item in planner_output.lane_data["historical"] if hasattr(item, "timestamp_world_begin")]

    recent_real_features = [item for item in planner_output.lane_data["recent"] if isinstance(item, Feature)]
    recent_max_end = max(item.timestamp_world_end for item in recent_items)
    recent_min_begin = min(item.timestamp_world_begin for item in recent_items)
    historical_max_end = max(item.timestamp_world_end for item in historical_items)
    global_max_end = max(item.timestamp_world_end for item in causal_features)

    role_counts = Counter(role for role, _ in turns)
    adjacent_same_role_pairs = sum(1 for (left_role, _), (right_role, _) in zip(turns, turns[1:]) if left_role == right_role)
    duplicate_nontrivial_line_count = _count_duplicate_nontrivial_lines(rendered_text)
    role_source_mismatch_count = _count_role_source_mismatches(turns)
    lane_selection_overflow_count = sum(
        1
        for meta in planner_output.metadata["lanes"].values()
        if len(meta["selected_source_ids"]) > meta["candidate_count"]
    )

    planner_used_tokens = sum(meta["token_budget_used"] for meta in planner_output.metadata["lanes"].values())
    snapshot = _render_snapshot(
        snapshot_name=snapshot_name,
        query_topic=query_topic,
        raw_token_count=raw_token_count,
        rendered_token_count=rendered_token_count,
        planner_token_budget=settings.token_length,
        planner_used_tokens=planner_used_tokens,
        turns=turns,
        planner_output=planner_output,
        duplicate_nontrivial_line_count=duplicate_nontrivial_line_count,
        adjacent_same_role_pairs=adjacent_same_role_pairs,
        lane_selection_overflow_count=lane_selection_overflow_count,
        role_source_mismatch_count=role_source_mismatch_count,
    )

    return {
        "snapshot": snapshot,
        "planner_output": planner_output,
        "turns": turns,
        "raw_token_count": raw_token_count,
        "rendered_token_count": rendered_token_count,
        "target_token_budget": TARGET_TOKEN_BUDGET,
        "planner_token_budget": settings.token_length,
        "planner_used_tokens": planner_used_tokens,
        "recent_items": recent_items,
        "historical_items": historical_items,
        "recent_real_features": recent_real_features,
        "recent_max_end": recent_max_end,
        "recent_min_begin": recent_min_begin,
        "historical_max_end": historical_max_end,
        "global_max_end": global_max_end,
        "assistant_turn_count": role_counts.get("assistant", 0),
        "user_turn_count": role_counts.get("user", 0),
        "adjacent_same_role_pairs": adjacent_same_role_pairs,
        "duplicate_nontrivial_line_count": duplicate_nontrivial_line_count,
        "lane_selection_overflow_count": lane_selection_overflow_count,
        "role_source_mismatch_count": role_source_mismatch_count,
    }


def _render_snapshot(
    *,
    snapshot_name: str,
    query_topic: str,
    raw_token_count: int,
    rendered_token_count: int,
    planner_token_budget: int,
    planner_used_tokens: int,
    turns,
    planner_output,
    duplicate_nontrivial_line_count: int,
    adjacent_same_role_pairs: int,
    lane_selection_overflow_count: int,
    role_source_mismatch_count: int,
) -> str:
    lane_metadata = planner_output.metadata["lanes"]
    lane_lines = [
        (
            f"LANE {lane_name}: candidates={meta['candidate_count']} "
            f"selected={len(meta['selected_source_ids'])} used={meta['token_budget_used']} "
            f"base={meta['token_budget_base']} current={meta['token_budget_current']}"
        )
        for lane_name, meta in lane_metadata.items()
    ]

    turn_lines = []
    for index, (role, content) in enumerate(turns):
        turn_lines.append(f"TURN {index} ROLE={role}")
        turn_lines.append(content)
        turn_lines.append("")

    return "\n".join(
        [
            f"SNAPSHOT: {snapshot_name}",
            f"QUERY_TOPIC: {query_topic}",
            f"RAW_TOKEN_COUNT: {raw_token_count}",
            f"TARGET_TOKEN_BUDGET: {TARGET_TOKEN_BUDGET}",
            f"PLANNER_TOKEN_BUDGET: {planner_token_budget}",
            f"PLANNER_USED_TOKENS: {planner_used_tokens}",
            f"RENDERED_TOKEN_COUNT: {rendered_token_count}",
            f"TURN_COUNT: {len(turns)}",
            f"DUPLICATE_NONTRIVIAL_LINE_COUNT: {duplicate_nontrivial_line_count}",
            f"ADJACENT_SAME_ROLE_PAIRS: {adjacent_same_role_pairs}",
            f"LANE_SELECTION_OVERFLOW_COUNT: {lane_selection_overflow_count}",
            f"ROLE_SOURCE_MISMATCH_COUNT: {role_source_mismatch_count}",
            "",
            "=== LANE METADATA ===",
            *lane_lines,
            "",
            "=== FINAL PROMPT ===",
            *turn_lines,
        ]
    )


def _render_turns_as_text(turns) -> str:
    return "\n\n".join(f"{role.upper()}:\n{content}" for role, content in turns)


def _count_duplicate_nontrivial_lines(text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 48]
    counts = Counter(lines)
    return sum(count - 1 for count in counts.values() if count > 1)


def _count_role_source_mismatches(turns) -> int:
    mismatches = 0
    for role, content in turns:
        if role == "assistant" and "Rick says:" in content:
            mismatches += 1
        if role == "user" and "Mira says:" in content:
            mismatches += 1
    return mismatches


def _assert_snapshot_text(snapshot_name: str, text: str) -> None:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    snapshot_path = SNAPSHOT_DIR / snapshot_name
    if UPDATE_SNAPSHOTS or not snapshot_path.exists():
        snapshot_path.write_text(text, encoding="utf-8")
    expected = snapshot_path.read_text(encoding="utf-8")
    assert text == expected
