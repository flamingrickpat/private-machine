import os
from datetime import datetime
from pathlib import Path

import pytest

from pm.model.knoxel_common import CauseEffectKnoxel, DeclarativeFactKnoxel, MemoryClusterKnoxel
from pm.model.knoxel_enums import ClusterType
from pm.model.knoxel_list import KnoxelList
from pm.subsystems.memory.memory_consolidation import MemoryConsolidationConfig
from pm.subsystems.memory.memory_consolidation_test_mode import SyntheticMemoryConsolidator
from test.infrastructure import (
    GenerationScale,
    TestInfrastructureConfig,
    populate_test_ghost_raw_only,
    validate_generated_ghost_strict,
)


SNAPSHOT_DIR = Path(__file__).with_name("test_data")
UPDATE_SNAPSHOTS = os.environ.get("UPDATE_PROMPT_SNAPSHOTS") == "1"


def test_raw_only_infrastructure_starts_without_derived_memory() -> None:
    corpus = populate_test_ghost_raw_only(
        TestInfrastructureConfig(
            seed=101,
            start_time_utc=datetime(2024, 1, 1, 9, 0),
            scale=GenerationScale(days=8, scenes_per_day=3, topical_clusters_per_topic=2, temporal_cluster_levels=(6, 5, 4)),
            topics=("music", "travel", "trust_repair", "robotics"),
        )
    )
    ghost = corpus.ghost

    assert ghost.all_features
    assert not ghost.all_episodic_memories
    assert not ghost.all_declarative_facts
    assert not ghost.all_cause_effects
    assert not any(isinstance(knoxel, MemoryClusterKnoxel) for knoxel in ghost.all_knoxels.values())
    assert not any(isinstance(knoxel, DeclarativeFactKnoxel) for knoxel in ghost.all_knoxels.values())
    assert not any(isinstance(knoxel, CauseEffectKnoxel) for knoxel in ghost.all_knoxels.values())


@pytest.mark.parametrize(
    ("snapshot_name", "seed", "scale"),
    [
        ("prompt_test_1.txt", 103, GenerationScale(days=8, scenes_per_day=4, topical_clusters_per_topic=2, temporal_cluster_levels=(6, 5, 4))),
        ("prompt_test_2.txt", 211, GenerationScale(days=12, scenes_per_day=4, topical_clusters_per_topic=3, temporal_cluster_levels=(6, 5, 4, 3))),
        ("prompt_test_3.txt", 307, GenerationScale(days=18, scenes_per_day=5, topical_clusters_per_topic=4, temporal_cluster_levels=(6, 5, 4, 3))),
    ],
)
def test_synthetic_consolidator_prompt_snapshot(snapshot_name: str, seed: int, scale: GenerationScale) -> None:
    ghost = _build_consolidated_ghost(seed=seed, scale=scale)
    snapshot = _render_prompt_snapshot(ghost, snapshot_name)
    _assert_snapshot_text(snapshot_name, snapshot)


def test_synthetic_consolidator_generates_memories_from_raw_only_corpus() -> None:
    ghost = _build_consolidated_ghost(
        seed=401,
        scale=GenerationScale(days=20, scenes_per_day=4, topical_clusters_per_topic=3, temporal_cluster_levels=(6, 5, 4, 3)),
    )

    topical = [memory for memory in ghost.all_episodic_memories if memory.cluster_type == ClusterType.Topical]
    temporal = [memory for memory in ghost.all_episodic_memories if memory.cluster_type == ClusterType.Temporal]

    assert topical
    assert temporal
    assert any(memory.level == 100 for memory in topical)
    assert any(memory.level == 6 for memory in temporal)
    assert any(memory.level <= 4 for memory in temporal)
    assert all(memory.embedding for memory in temporal)
    assert all(fact.source_cluster_id in {memory.id for memory in topical} for fact in ghost.all_declarative_facts)
    assert all(item.source_cluster_id in {memory.id for memory in topical} for item in ghost.all_cause_effects)

    validate_generated_ghost_strict(ghost)


def _build_consolidated_ghost(*, seed: int, scale: GenerationScale):
    corpus = populate_test_ghost_raw_only(
        TestInfrastructureConfig(
            seed=seed,
            start_time_utc=datetime(2024, 1, 1, 9, 0),
            scale=scale,
            topics=("music", "travel", "trust_repair", "boundary_fear", "robotics", "gardening"),
        )
    )
    ghost = corpus.ghost
    unclustered_before = [feature for feature in ghost.all_features if feature.causal]

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

    assert len(unclustered_before) >= 8
    assert ghost.all_episodic_memories
    assert ghost.all_declarative_facts
    assert ghost.all_cause_effects
    return ghost


def _render_prompt_snapshot(ghost, snapshot_name: str) -> str:
    causal_features = sorted([feature for feature in ghost.all_features if feature.causal], key=lambda item: (item.timestamp_world_begin, item.id))
    topical = sorted(
        [memory for memory in ghost.all_episodic_memories if memory.cluster_type == ClusterType.Topical],
        key=lambda item: (item.timestamp_world_begin, item.id),
    )
    temporal = sorted(
        [memory for memory in ghost.all_episodic_memories if memory.cluster_type == ClusterType.Temporal],
        key=lambda item: (item.level, item.timestamp_world_begin, item.id),
    )
    facts = sorted(ghost.all_declarative_facts, key=lambda item: (item.timestamp_world_begin, item.id))
    cause_effects = sorted(ghost.all_cause_effects, key=lambda item: (item.timestamp_world_begin, item.id))

    raw_prompt = KnoxelList(causal_features[:10]).get_story()
    topical_prompt = "\n\n".join(_describe_memory(memory) for memory in topical[:3])
    temporal_prompt = "\n\n".join(_describe_memory(memory) for memory in temporal[:6])
    facts_prompt = "\n".join(_describe_fact(fact) for fact in facts[:6])
    cause_effect_prompt = "\n".join(_describe_cause_effect(item) for item in cause_effects[:4])

    return "\n".join(
        [
            f"SNAPSHOT: {snapshot_name}",
            f"FEATURE_COUNT: {len(causal_features)}",
            f"TOPICAL_COUNT: {len(topical)}",
            f"TEMPORAL_COUNT: {len(temporal)}",
            f"FACT_COUNT: {len(facts)}",
            f"CAUSE_EFFECT_COUNT: {len(cause_effects)}",
            "",
            "=== RAW STORY PROMPT ===",
            raw_prompt,
            "",
            "=== TOPICAL MEMORIES ===",
            topical_prompt,
            "",
            "=== TEMPORAL MEMORIES ===",
            temporal_prompt,
            "",
            "=== FACTS ===",
            facts_prompt,
            "",
            "=== CAUSE EFFECT ===",
            cause_effect_prompt,
            "",
        ]
    )


def _describe_memory(memory: MemoryClusterKnoxel) -> str:
    return "\n".join(
        [
            f"MEMORY id={memory.id} type={memory.cluster_type.value} level={memory.level} begin={memory.timestamp_world_begin} end={memory.timestamp_world_end}",
            f"EVENT_IDS={memory.included_event_ids or '-'}",
            f"CLUSTER_IDS={memory.included_cluster_ids or '-'}",
            f"CONTENT={memory.content}",
            f"STORY={memory.get_story_element()}",
        ]
    )


def _describe_fact(fact: DeclarativeFactKnoxel) -> str:
    return (
        f"FACT id={fact.id} source={fact.source_cluster_id} begin={fact.timestamp_world_begin} end={fact.timestamp_world_end} "
        f"categories={','.join(fact.category)} content={fact.content}"
    )


def _describe_cause_effect(item: CauseEffectKnoxel) -> str:
    return (
        f"CAUSE_EFFECT id={item.id} source={item.source_cluster_id} begin={item.timestamp_world_begin} end={item.timestamp_world_end} "
        f"cause={item.cause} effect={item.effect}"
    )


def _assert_snapshot_text(snapshot_name: str, text: str) -> None:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    snapshot_path = SNAPSHOT_DIR / snapshot_name
    if UPDATE_SNAPSHOTS or not snapshot_path.exists():
        snapshot_path.write_text(text, encoding="utf-8")
    expected = snapshot_path.read_text(encoding="utf-8")
    assert text == expected
