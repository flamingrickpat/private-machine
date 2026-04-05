from datetime import datetime, timedelta, timezone

from pm.model.knoxel_enums import FeatureType, InterlocusType
from pm.model.knoxel_feature import Feature
from pm.subsystems.context.context_planner_base import (
    ContextDiversitySettings,
    ContextLaneSettings,
    ContextPlanner,
    ContextPlannerSettings,
    ContextPostProcessSettings,
    ContextPreFilterRules,
    ContextSampleSettings,
    ContextScoreSettings,
    SortingStrategy,
    TemporalSamplingMode,
)
from test.infrastructure import populate_test_ghost


def test_single_lane_returns_top_embedding_matches_within_budget():
    corpus = populate_test_ghost()
    planner = ContextPlanner()
    query_embedding = corpus.query_embeddings["music"]

    settings = ContextPlannerSettings(
        token_length=70,
        ordered_lanes=[
            ContextLaneSettings(
                name="all_knoxels",
                token_budget_pct=1.0,
                sorting_strategy=SortingStrategy.ScoreDesc,
                scoring=ContextScoreSettings(weighted_embeddings=[(1.0, query_embedding)]),
            )
        ],
        lane_priority=[],
        pre_filter=ContextPreFilterRules(),
        scoring=ContextScoreSettings(weighted_embeddings=[(1.0, query_embedding)]),
        diversity=ContextDiversitySettings(),
        sampler=ContextSampleSettings(),
        post_process=ContextPostProcessSettings(),
    )

    knoxels = list(corpus.ghost.all_knoxels.values())
    planner_output = planner.plan(knoxels=knoxels, settings=settings)
    selected = planner_output.lane_data["all_knoxels"]
    lane_metadata = planner_output.metadata["lanes"]["all_knoxels"]

    assert selected
    assert all(hasattr(item, "embedding") for item in selected)
    assert lane_metadata["token_budget_used"] <= settings.token_length

    scored_candidates = []
    for knoxel in knoxels:
        similarity = planner._cosine_similarity(knoxel.embedding, query_embedding)
        scored_candidates.append((knoxel, similarity, planner._estimate_knoxel_tokens(knoxel)))
    scored_candidates.sort(key=lambda item: (-item[1], item[0].id))

    expected_ids = []
    expected_scores = []
    tokens_used = 0
    for knoxel, similarity, token_cost in scored_candidates:
        if expected_ids and tokens_used + token_cost > settings.token_length:
            continue
        expected_ids.append(knoxel.id)
        expected_scores.append(similarity)
        tokens_used += token_cost
        if tokens_used >= settings.token_length:
            break

    assert [item.id for item in selected] == expected_ids
    assert [entry["id"] for entry in lane_metadata["selected_scores"]] == expected_ids
    assert [round(entry["score"], 8) for entry in lane_metadata["selected_scores"]] == [round(score, 8) for score in expected_scores]
    assert [entry["score"] for entry in lane_metadata["selected_scores"]] == sorted(
        [entry["score"] for entry in lane_metadata["selected_scores"]],
        reverse=True,
    )


def test_recent_window_returns_contiguous_most_recent_suffix():
    planner = ContextPlanner()
    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    recent_features = []
    for tick_id in range(1, 13):
        recent_features.append(
            Feature(
                id=tick_id,
                content=f"dialogue tick {tick_id}",
                feature_type=FeatureType.Dialogue,
                interlocus=InterlocusType.Public,
                causal=True,
                source="Rick" if tick_id % 2 else "Mira",
                embedding=[1.0 if tick_id in {3, 7} else 0.0],
                tick_id=tick_id,
                timestamp_world_begin=base_time + timedelta(minutes=tick_id),
                timestamp_world_end=base_time + timedelta(minutes=tick_id, seconds=30),
            )
        )

    item_tokens = planner._estimate_knoxel_tokens(recent_features[-1])
    budget_for_five_items = item_tokens * 5

    settings = ContextPlannerSettings(
        token_length=budget_for_five_items,
        ordered_lanes=[
            ContextLaneSettings(
                name="recent",
                token_budget_pct=1.0,
                sorting_strategy=SortingStrategy.TimeAsc,
                pre_filter=ContextPreFilterRules(allowed_types=[FeatureType.Dialogue], causal_only=True),
                scoring=ContextScoreSettings(weighted_embeddings=[(1.0, [1.0])]),
                sampler=ContextSampleSettings(
                    sample_temporal_mode=TemporalSamplingMode.RecentWindow,
                    sample_require_contiguous_temporal_span=True,
                ),
            )
        ],
        lane_priority=[],
        pre_filter=ContextPreFilterRules(),
        scoring=ContextScoreSettings(),
        diversity=ContextDiversitySettings(),
        sampler=ContextSampleSettings(),
        post_process=ContextPostProcessSettings(),
    )

    planner_output = planner.plan(knoxels=recent_features, settings=settings)
    selected = planner_output.lane_data["recent"]

    assert [item.tick_id for item in selected] == [8, 9, 10, 11, 12]
    assert [item.id for item in selected] == [feature.id for feature in recent_features[-5:]]
