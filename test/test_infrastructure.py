from __future__ import annotations

from collections import defaultdict

from pm.model.knoxel_common import MemoryClusterKnoxel
from pm.model.knoxel_enums import ClusterType, FeatureType
from test.infrastructure import (
    LARGE_SCALE,
    MEDIUM_SCALE,
    GenerationScale,
    TestInfrastructureConfig,
    populate_test_ghost,
    validate_generated_ghost_strict,
)

MULTI_YEAR_SCALE = GenerationScale(
    days=365 * 3,
    scenes_per_day=2,
    topical_clusters_per_topic=12,
    temporal_cluster_levels=(6, 5, 4, 3, 2, 1),
)


def test_tiny_corpus_can_simulate_pre_memory_state() -> None:
    corpus = populate_test_ghost(
        TestInfrastructureConfig(
            seed=11,
            scale=GenerationScale(days=2, scenes_per_day=1, topical_clusters_per_topic=1, temporal_cluster_levels=(6,)),
            topics=("music", "travel"),
        )
    )
    ghost = corpus.ghost

    assert ghost.all_features
    assert any(feature.causal for feature in ghost.all_features)
    assert any(feature.feature_type == FeatureType.Dialogue for feature in ghost.all_features)

    memory_ids = {memory.id for memory in ghost.all_episodic_memories}
    ghost.all_episodic_memories = []
    ghost.all_knoxels = {knoxel_id: knoxel for knoxel_id, knoxel in ghost.all_knoxels.items() if knoxel_id not in memory_ids}

    assert ghost.all_features
    assert not ghost.all_episodic_memories
    assert not any(isinstance(knoxel, MemoryClusterKnoxel) for knoxel in ghost.all_knoxels.values())


def test_medium_corpus_contains_raw_data_and_temporal_history() -> None:
    corpus = populate_test_ghost(
        TestInfrastructureConfig(
            seed=17,
            scale=MEDIUM_SCALE,
            topics=("music", "gardening", "robotics", "travel", "trust_repair"),
        )
    )
    ghost = corpus.ghost

    validate_generated_ghost_strict(ghost)

    causal_features = [feature for feature in ghost.all_features if feature.causal]
    topical_clusters = [memory for memory in ghost.all_episodic_memories if memory.cluster_type == ClusterType.Topical]
    temporal_clusters = [memory for memory in ghost.all_episodic_memories if memory.cluster_type == ClusterType.Temporal]
    temporal_levels = {memory.level for memory in temporal_clusters}

    assert len(causal_features) >= 40
    assert any(feature.feature_type == FeatureType.Dialogue for feature in causal_features)
    assert any(feature.feature_type == FeatureType.Thought for feature in causal_features)
    assert topical_clusters
    assert temporal_clusters
    assert temporal_levels == set(MEDIUM_SCALE.temporal_cluster_levels)

    _assert_temporal_hierarchy_is_exact(ghost)


def test_large_corpus_has_deep_temporal_summary_coverage() -> None:
    corpus = populate_test_ghost(
        TestInfrastructureConfig(
            seed=23,
            scale=LARGE_SCALE,
            topics=("music", "gardening", "robotics", "travel", "trust_repair", "boundary_fear"),
        )
    )
    ghost = corpus.ghost

    validate_generated_ghost_strict(ghost)

    causal_features = [feature for feature in ghost.all_features if feature.causal]
    temporal_clusters = [memory for memory in ghost.all_episodic_memories if memory.cluster_type == ClusterType.Temporal]
    clusters_by_level = defaultdict(list)
    for cluster in temporal_clusters:
        clusters_by_level[cluster.level].append(cluster)

    assert len(causal_features) > 200
    assert set(clusters_by_level) == set(LARGE_SCALE.temporal_cluster_levels)
    assert len(clusters_by_level[3]) >= 2
    assert len(clusters_by_level[4]) >= len(clusters_by_level[3])
    assert len(clusters_by_level[5]) >= len(clusters_by_level[4])
    assert len(clusters_by_level[6]) >= len(clusters_by_level[5])

    _assert_temporal_hierarchy_is_exact(ghost)
    _assert_temporal_levels_are_chronological(clusters_by_level)


def test_ultra_long_corpus_builds_multi_month_hierarchy() -> None:
    corpus = populate_test_ghost(
        TestInfrastructureConfig(
            seed=31,
            scale=GenerationScale(days=120, scenes_per_day=4, topical_clusters_per_topic=6, temporal_cluster_levels=(6, 5, 4, 3, 2, 1)),
            topics=("music", "travel", "trust_repair", "boundary_fear", "robotics", "gardening"),
        )
    )
    ghost = corpus.ghost

    validate_generated_ghost_strict(ghost)

    causal_features = [feature for feature in ghost.all_features if feature.causal]
    temporal_clusters = [memory for memory in ghost.all_episodic_memories if memory.cluster_type == ClusterType.Temporal]
    clusters_by_level = defaultdict(list)
    for cluster in temporal_clusters:
        clusters_by_level[cluster.level].append(cluster)

    assert len(causal_features) >= 120 * 8
    assert set(clusters_by_level) == {1, 2, 3, 4, 5, 6}
    assert clusters_by_level[1]
    assert clusters_by_level[2]
    assert len(clusters_by_level[1]) >= 3
    assert all(cluster.included_cluster_ids for cluster in clusters_by_level[2])
    assert all(cluster.included_event_ids or cluster.included_cluster_ids for cluster in clusters_by_level[1])
    assert any(cluster.included_cluster_ids for cluster in temporal_clusters)

    _assert_temporal_hierarchy_is_exact(ghost)
    _assert_temporal_levels_are_chronological(clusters_by_level)


def test_multi_year_corpus_builds_multi_year_hierarchy() -> None:
    corpus = populate_test_ghost(
        TestInfrastructureConfig(
            seed=41,
            scale=MULTI_YEAR_SCALE,
            topics=("music", "travel", "trust_repair", "boundary_fear", "robotics", "gardening"),
        )
    )
    ghost = corpus.ghost

    validate_generated_ghost_strict(ghost)

    causal_features = [feature for feature in ghost.all_features if feature.causal]
    temporal_clusters = [memory for memory in ghost.all_episodic_memories if memory.cluster_type == ClusterType.Temporal]
    clusters_by_level = defaultdict(list)
    for cluster in temporal_clusters:
        clusters_by_level[cluster.level].append(cluster)

    assert len(causal_features) >= MULTI_YEAR_SCALE.days * 4
    assert set(clusters_by_level) == {1, 2, 3, 4, 5, 6}
    assert len(clusters_by_level[1]) >= 30
    assert len(clusters_by_level[2]) >= 60
    assert len(clusters_by_level[6]) >= MULTI_YEAR_SCALE.days
    assert any(cluster.included_cluster_ids for cluster in clusters_by_level[2])
    assert any(cluster.included_cluster_ids for cluster in clusters_by_level[4])

    _assert_temporal_hierarchy_is_exact(ghost)
    _assert_temporal_levels_are_chronological(clusters_by_level)


def _assert_temporal_hierarchy_is_exact(ghost) -> None:
    feature_by_id = {feature.id: feature for feature in ghost.all_features}
    memory_by_id = {memory.id: memory for memory in ghost.all_episodic_memories}

    for memory in ghost.all_episodic_memories:
        if memory.cluster_type == ClusterType.Topical:
            event_ids = [int(value) for value in memory.included_event_ids.split(",") if value]
            covered = [feature_by_id[event_id] for event_id in event_ids]
            assert covered
            assert memory.level == 100
            assert memory.timestamp_world_begin == min(item.timestamp_world_begin for item in covered)
            assert memory.timestamp_world_end == max(item.timestamp_world_end for item in covered)
            continue

        assert memory.cluster_type == ClusterType.Temporal
        assert memory.level in {1, 2, 3, 4, 5, 6}
        assert memory.min_event_id <= memory.max_event_id
        assert memory.min_tick_id <= memory.max_tick_id

        if memory.included_cluster_ids:
            child_ids = [int(value) for value in memory.included_cluster_ids.split(",") if value]
            children = [memory_by_id[child_id] for child_id in child_ids]
            assert children
            assert all(child.level > memory.level for child in children)
            assert memory.timestamp_world_begin == min(child.timestamp_world_begin for child in children)
            assert memory.timestamp_world_end == max(child.timestamp_world_end for child in children)
        else:
            event_ids = [int(value) for value in memory.included_event_ids.split(",") if value]
            covered = [feature_by_id[event_id] for event_id in event_ids]
            assert covered
            assert memory.timestamp_world_begin == min(item.timestamp_world_begin for item in covered)
            assert memory.timestamp_world_end == max(item.timestamp_world_end for item in covered)


def _assert_temporal_levels_are_chronological(clusters_by_level) -> None:
    for clusters in clusters_by_level.values():
        ordered = sorted(clusters, key=lambda item: (item.timestamp_world_begin, item.id))
        for previous, current in zip(ordered, ordered[1:]):
            assert previous.timestamp_world_begin <= current.timestamp_world_begin
            assert previous.timestamp_world_end <= current.timestamp_world_begin
