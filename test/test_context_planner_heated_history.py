from collections import Counter

from pm.data.prompts.character_card_addendum import architecture_description_story_detailed
from pm.model.knoxel_common import CauseEffectKnoxel, DeclarativeFactKnoxel, MemoryClusterKnoxel, Narrative
from pm.model.knoxel_enums import ClusterType, FeatureType
from pm.model.knoxel_feature import Feature
from pm.subsystems.context.common_context import render_story_context_string
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
    VirtualKnoxel,
)
from pm.utils.token_utils import get_token_count
from test.infrastructure import LARGE_SCALE, TestInfrastructureConfig, populate_test_ghost


def test_heated_situation_keywords_emotion_history_and_stats():
    config = TestInfrastructureConfig(
        seed=23,
        scale=LARGE_SCALE,
        topics=("heated_conflict", "quiet_routine"),
        character_card_story=(
            "Mira is a fictional research companion with a long memory, strong narrative continuity, "
            "and a habit of replaying emotionally intense conversations until they make sense."
        ),
        available_tools="calendar.lookup, notes.search, memory.timeline",
    )
    corpus = populate_test_ghost(config)
    planner = ContextPlanner()

    companion_name = corpus.ghost.ghost_config.companion_name
    user_name = corpus.ghost.ghost_config.user_name
    static_lane_text = architecture_description_story_detailed.format(
        companion_name=companion_name,
        user_name=user_name,
        available_tools=config.available_tools,
    )
    static_prefix = static_lane_text + "\n" + config.character_card_story

    heated_embedding = corpus.query_embeddings["heated_conflict"]
    distress_delta = corpus.mental_queries["distress"]
    weighted_keywords = ["heated", "argument", "betrayal", "shouting", "fear", "apology"]

    settings = ContextPlannerSettings(
        token_length=4000,
        ordered_lanes=[
            ContextLaneSettings(
                name="static",
                token_budget_pct=0.0,
                sorting_strategy=SortingStrategy.TimeAsc,
                overwrite_data=[static_lane_text, config.character_card_story],
            ),
            ContextLaneSettings(
                name="narratives",
                token_budget_pct=0.10,
                sorting_strategy=SortingStrategy.ScoreDesc,
                pre_filter=ContextPreFilterRules(allowed_types=[Narrative]),
                scoring=ContextScoreSettings(
                    weighted_embeddings=[(1.0, heated_embedding)],
                    weighted_key_phrases=[(2.0, keyword) for keyword in weighted_keywords],
                ),
            ),
            ContextLaneSettings(
                name="facts_rules",
                token_budget_pct=0.20,
                sorting_strategy=SortingStrategy.ScoreDesc,
                pre_filter=ContextPreFilterRules(allowed_types=[DeclarativeFactKnoxel, CauseEffectKnoxel]),
                scoring=ContextScoreSettings(
                    weighted_embeddings=[(1.0, heated_embedding)],
                    weighted_key_phrases=[(2.0, keyword) for keyword in weighted_keywords],
                ),
            ),
            ContextLaneSettings(
                name="historical",
                token_budget_pct=0.40,
                sorting_strategy=SortingStrategy.TimeAsc,
                pre_filter=ContextPreFilterRules(
                    allowed_types=[MemoryClusterKnoxel, FeatureType.Dialogue, FeatureType.Thought, FeatureType.StoryWildcard],
                    end_before_next_lane=True,
                ),
                scoring=ContextScoreSettings(
                    weighted_embeddings=[(1.0, heated_embedding)],
                    weighted_mental_delta=[(2.2, distress_delta)],
                    weighted_key_phrases=[(2.4, keyword) for keyword in weighted_keywords],
                ),
                sampler=ContextSampleSettings(
                    sample_temporal_mode=TemporalSamplingMode.PreserveHistory,
                    sample_require_temporal_history=True,
                    sample_knoxel_type_distribution=[
                        (0.35, ClusterType.Topical),
                        (0.40, ClusterType.Temporal),
                        (0.25, FeatureType.Dialogue),
                    ],
                    sample_resolve_cluster_to_children_pct=0.35,
                    sample_resolve_cluster_children_types=[FeatureType.Dialogue],
                ),
            ),
            ContextLaneSettings(
                name="recent",
                token_budget_pct=0.30,
                sorting_strategy=SortingStrategy.TimeAsc,
                pre_filter=ContextPreFilterRules(
                    allowed_types=[FeatureType.Dialogue, FeatureType.Thought, FeatureType.StoryWildcard, FeatureType.ExternalThought],
                    causal_only=True,
                    begin_after_previous_lane=True,
                ),
                scoring=ContextScoreSettings(
                    weighted_embeddings=[(1.0, heated_embedding)],
                    weighted_mental_delta=[(2.0, distress_delta)],
                    weighted_key_phrases=[(2.5, keyword) for keyword in weighted_keywords],
                    temporal_decay_per_day=0.75,
                    temporal_decay_exempt_types=[FeatureType.Dialogue, FeatureType.Thought, FeatureType.ExternalThought],
                ),
                sampler=ContextSampleSettings(
                    sample_temporal_mode=TemporalSamplingMode.RecentWindow,
                    sample_knoxel_type_distribution=[
                        (0.45, FeatureType.Dialogue),
                        (0.25, FeatureType.Thought),
                        (0.30, FeatureType.StoryWildcard),
                    ],
                    sample_fill_shortfall_from_type_distribution=[(FeatureType.Thought, FeatureType.Dialogue)],
                ),
                post_process=ContextPostProcessSettings(
                    additional_data_insert_timestamp_headers=True,
                    additional_data_timestamp_header_mode="day_change",
                ),
            ),
        ],
        lane_priority=[],
        pre_filter=ContextPreFilterRules(allow_virtual_knoxels=True),
        scoring=ContextScoreSettings(weighted_embeddings=[(1.0, heated_embedding)]),
        diversity=ContextDiversitySettings(),
        sampler=ContextSampleSettings(),
        post_process=ContextPostProcessSettings(),
    )
    settings.lane_priority = [
        next(lane for lane in settings.ordered_lanes if lane.name == "recent"),
        next(lane for lane in settings.ordered_lanes if lane.name == "historical"),
        next(lane for lane in settings.ordered_lanes if lane.name == "facts_rules"),
        next(lane for lane in settings.ordered_lanes if lane.name == "narratives"),
    ]

    planner_output = planner.plan(list(corpus.ghost.all_knoxels.values()), settings)
    rendered = render_story_context_string(planner_output, static_prefix)

    lane_metadata = planner_output.metadata["lanes"]
    total_knoxels = len(corpus.ghost.all_knoxels)
    print(f"total_knoxels={total_knoxels}")
    for lane_name, meta in lane_metadata.items():
        discarded = meta["candidate_count"] - len(meta["selected_source_ids"])
        print(
            f"lane={lane_name} candidate_count={meta['candidate_count']} selected={len(meta['selected_source_ids'])} "
            f"discarded={discarded} budget_used={meta['token_budget_used']} budget_current={meta['token_budget_current']}"
        )

    historical_lane = planner_output.lane_data["historical"]
    recent_lane = planner_output.lane_data["recent"]
    historical_real_items = [item for item in historical_lane if not isinstance(item, VirtualKnoxel)]
    recent_virtuals = [item for item in recent_lane if isinstance(item, VirtualKnoxel)]
    historical_timestamps = [item.timestamp_world_begin for item in historical_real_items]
    assert historical_timestamps == sorted(historical_timestamps)

    temporal_levels = Counter(item.level for item in historical_real_items if isinstance(item, MemoryClusterKnoxel) and item.cluster_type == ClusterType.Temporal)
    feature_types = Counter(str(getattr(item, "feature_type", "")) for item in historical_real_items if isinstance(item, Feature))
    total_historical_items = max(1, len(historical_real_items))
    print(f"historical_feature_types={dict(feature_types)}")
    print(f"historical_temporal_levels={dict(temporal_levels)}")
    print(
        "historical_feature_type_ratios="
        + str({key: round(value / total_historical_items, 3) for key, value in feature_types.items()})
    )
    temporal_total = max(1, sum(temporal_levels.values()))
    print(
        "historical_summary_level_ratios="
        + str({key: round(value / temporal_total, 3) for key, value in temporal_levels.items()})
    )

    temporal_dates = sorted(
        {
            item.timestamp_world_begin.date()
            for item in historical_real_items
            if isinstance(item, MemoryClusterKnoxel) and item.cluster_type == ClusterType.Temporal
        }
    )
    max_temporal_gap_days = 0
    for previous, current in zip(temporal_dates, temporal_dates[1:]):
        max_temporal_gap_days = max(max_temporal_gap_days, (current - previous).days)
    gapless_temporal_history = max_temporal_gap_days <= 2
    print(f"gapless_temporal_history={gapless_temporal_history} max_temporal_gap_days={max_temporal_gap_days}")
    print(f"recent_timestamp_virtuals={len(recent_virtuals)}")
    print(f"rendered_token_count={get_token_count(rendered)}")

    assert total_knoxels > 300
    assert gapless_temporal_history
    assert len(recent_virtuals) >= 2
    assert lane_metadata["historical"]["candidate_count"] > len(lane_metadata["historical"]["selected_source_ids"])
    assert lane_metadata["recent"]["candidate_count"] > len(lane_metadata["recent"]["selected_source_ids"])

    assert "private-machine" in rendered
    assert "LIDA-inspired cognitive architecture" in rendered
    assert "long memory" in rendered
    assert "heated argument" in rendered
    assert "betrayal" in rendered
    assert "shouting" in rendered
    assert "fear" in rendered
    assert "apology" in rendered

    historical_text = "\n".join(item.content for item in historical_real_items)
    recent_text = "\n".join(item.content for item in recent_lane if not isinstance(item, VirtualKnoxel))
    facts_rules_text = "\n".join(item.content for item in planner_output.lane_data["facts_rules"])
    assert "heated argument" in historical_text or "shouting" in historical_text
    assert "fear" in historical_text or "betrayal" in historical_text
    assert "apology" in recent_text or "heated argument" in recent_text
    assert "betrayal" in recent_text or "shouting" in recent_text
    assert "heated_conflict" in facts_rules_text
    assert "ClusterType.Topical" not in rendered

    first_recent_virtual_index = next(index for index, item in enumerate(recent_lane) if isinstance(item, VirtualKnoxel))
    assert first_recent_virtual_index == 0
    assert any("[20" in item.content for item in recent_virtuals)
