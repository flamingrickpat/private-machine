from typing import List, Tuple

from pm.ghost.ghost_base import BaseGhost
from pm.model.knoxel_common import CauseEffectKnoxel, DeclarativeFactKnoxel, MemoryClusterKnoxel, Narrative
from pm.model.knoxel_core import KnoxelType
from pm.model.knoxel_enums import ClusterType, FeatureType
from pm.subsystems.context.context_planner_base import (
    ContextDiversitySettings,
    ContextLaneSettings,
    ContextPlanner,
    ContextPlannerOutput,
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


def build_context_story_simple(ghost: BaseGhost, embedding: List[float]) -> str:
    planner_output, _, static_prefix = plan_context_story_simple(ghost, embedding)
    return render_story_context_string(planner_output, static_prefix)


def plan_context_story_simple(
    ghost: BaseGhost,
    embedding: List[float],
    total_token_budget: int = 4000,
) -> Tuple[ContextPlannerOutput, ContextPlannerSettings, str]:
    static_prefix = ghost.get_simple_character_story_block()
    dynamic_story_budget = max(1, total_token_budget - get_token_count(static_prefix))
    planner = ContextPlanner()

    companion_name = ghost.get_companion_name()
    user_name = ghost.get_user_name()

    settings = ContextPlannerSettings(
        token_length=dynamic_story_budget,
        lanes=[
            ContextLaneSettings(
                name="static",
                token_budget_pct=0.0,
                sorting_strategy=SortingStrategy.TimeAsc,
                overwrite_data=[static_prefix],
            ),
            ContextLaneSettings(
                name="narratives_ai",
                token_budget_pct=0.1125,
                sorting_strategy=SortingStrategy.ScoreDesc,
                pre_filter=ContextPreFilterRules(allowed_types=[Narrative]),
                scoring=ContextScoreSettings(
                    weighted_embeddings=[(1.0, embedding)],
                    lambdas_ban_knoxel=[lambda k: k.target_name not in ["{companion_name}", companion_name]],
                ),
            ),
            ContextLaneSettings(
                name="narratives_user",
                token_budget_pct=0.0375,
                sorting_strategy=SortingStrategy.ScoreDesc,
                pre_filter=ContextPreFilterRules(allowed_types=[Narrative]),
                scoring=ContextScoreSettings(
                    weighted_embeddings=[(1.0, embedding)],
                    lambdas_ban_knoxel=[lambda k: k.target_name not in ["{user_name}", user_name]],
                ),
            ),
            ContextLaneSettings(
                name="facts",
                token_budget_pct=0.075,
                sorting_strategy=SortingStrategy.ScoreDesc,
                pre_filter=ContextPreFilterRules(allowed_types=[DeclarativeFactKnoxel]),
                scoring=ContextScoreSettings(
                    weighted_embeddings=[(1.0, embedding)],
                    lambdas_weight_knoxel=[lambda k: float(k.importance)],
                ),
            ),
            ContextLaneSettings(
                name="rules",
                token_budget_pct=0.075,
                sorting_strategy=SortingStrategy.ScoreDesc,
                pre_filter=ContextPreFilterRules(allowed_types=[CauseEffectKnoxel]),
                scoring=ContextScoreSettings(weighted_embeddings=[(1.0, embedding)]),
            ),
            ContextLaneSettings(
                name="historical",
                token_budget_pct=0.35,
                sorting_strategy=SortingStrategy.TimeAsc,
                pre_filter=ContextPreFilterRules(
                    allowed_types=[ClusterType.Topical, ClusterType.Temporal],
                    end_before_next_lane=True,
                ),
                scoring=ContextScoreSettings(weighted_embeddings=[(1.0, embedding)]),
                sampler=ContextSampleSettings(
                    sample_temporal_mode=TemporalSamplingMode.PreserveHistory,
                    sample_require_temporal_history=True,
                    sample_resolve_cluster_to_children_pct=0, #.40,
                ),
            ),
            ContextLaneSettings(
                name="recent",
                token_budget_pct=0.35,
                sorting_strategy=SortingStrategy.TimeAsc,
                pre_filter=ContextPreFilterRules(
                    allowed_types=[KnoxelType.Feature],
                    causal_only=True,
                    begin_after_previous_lane=True,
                ),
                scoring=ContextScoreSettings(
                    weighted_embeddings=[(1.0, embedding)],
                    temporal_decay_per_day=0.75,
                    temporal_decay_reference="candidate_pool_most_recent",
                    temporal_decay_exempt_types=[FeatureType.Dialogue, FeatureType.Thought, FeatureType.ExternalThought],
                ),
                sampler=ContextSampleSettings(
                    sample_temporal_mode=TemporalSamplingMode.RecentWindow,
                    sample_require_contiguous_temporal_span=True,
                    sample_knoxel_type_distribution=[
                        (0.40, FeatureType.Dialogue),
                        (0.20, FeatureType.Thought)
                    ],
                    sample_fill_shortfall_from_type_distribution=[
                        (FeatureType.Thought, FeatureType.Dialogue),
                    ],
                    sample_max_type_to_reference_ratio=[
                        (1.0, KnoxelType.Feature, [FeatureType.Dialogue, FeatureType.Thought, FeatureType.ExternalThought]),
                    ],
                ),
                post_process=ContextPostProcessSettings(
                    additional_data_insert_timestamp_headers=True,
                    additional_data_timestamp_header_mode="day_change",
                ),
            ),
        ],
        lane_priority=[],
        pre_filter=ContextPreFilterRules(allow_virtual_knoxels=True),
        scoring=ContextScoreSettings(weighted_embeddings=[(1.0, embedding)]),
        diversity=ContextDiversitySettings(),
        sampler=ContextSampleSettings(),
        post_process=ContextPostProcessSettings(),
    )

    settings.lane_priority = [
        next(lane for lane in settings.lanes if lane.name == "recent"),
        next(lane for lane in settings.lanes if lane.name == "historical"),
        next(lane for lane in settings.lanes if lane.name == "facts"),
        next(lane for lane in settings.lanes if lane.name == "rules"),
        next(lane for lane in settings.lanes if lane.name == "narratives_ai"),
        next(lane for lane in settings.lanes if lane.name == "narratives_user"),
    ]
    settings.ordered_lanes = settings.lane_priority

    planner_output = planner.plan(ghost.get_all_knoxels(), settings)
    return planner_output, settings, static_prefix



def plan_context_story_debug(
    ghost: BaseGhost,
    embedding: List[float],
    total_token_budget: int = 4000,
) -> Tuple[ContextPlannerOutput, ContextPlannerSettings, str]:
    static_prefix = ghost.get_simple_character_story_block()
    dynamic_story_budget = max(1, total_token_budget - get_token_count(static_prefix))
    planner = ContextPlanner()

    companion_name = ghost.get_companion_name()
    user_name = ghost.get_user_name()

    settings = ContextPlannerSettings(
        token_length=dynamic_story_budget,
        ordered_lanes=[
            ContextLaneSettings(
                name="narratives_ai",
                token_budget_pct=0.1125,
                sorting_strategy=SortingStrategy.ScoreDesc,
                pre_filter=ContextPreFilterRules(allowed_types=[Narrative]),
                scoring=ContextScoreSettings(
                    weighted_embeddings=[(1.0, embedding)],
                    lambdas_ban_knoxel=[lambda k: k.target_name != companion_name],
                ),
            ),
            ContextLaneSettings(
                name="historical",
                token_budget_pct=0.8,
                sorting_strategy=SortingStrategy.TimeAsc,
                pre_filter=ContextPreFilterRules(
                    allowed_types=[ClusterType.Topical, ClusterType.Temporal],
                    end_before_next_lane=True,
                ),
                scoring=ContextScoreSettings(weighted_embeddings=[(1.0, embedding)]),
                sampler=ContextSampleSettings(
                    sample_temporal_mode=TemporalSamplingMode.PreserveHistory,
                    sample_require_temporal_history=True,
                    sample_knoxel_type_distribution=[
                        (0.40, ClusterType.Topical),
                        (0.60, ClusterType.Temporal),
                    ],
                    sample_resolve_cluster_to_children_pct=0.40,
                    sample_resolve_cluster_children_types=[FeatureType.Dialogue],
                ),
            ),
        ],
        lane_priority=[],
        pre_filter=ContextPreFilterRules(allow_virtual_knoxels=True),
        scoring=ContextScoreSettings(weighted_embeddings=[(1.0, embedding)]),
        diversity=ContextDiversitySettings(),
        sampler=ContextSampleSettings(),
        post_process=ContextPostProcessSettings(),
    )

    settings.lane_priority = [
        next(lane for lane in settings.ordered_lanes if lane.name == "historical"),
        next(lane for lane in settings.ordered_lanes if lane.name == "narratives_ai"),
    ]

    planner_output = planner.plan(ghost.get_all_knoxels(), settings)
    return planner_output, settings, static_prefix


def render_story_context_string(planner_output: ContextPlannerOutput, static_prefix: str = "") -> str:
    buffer: List[str] = [static_prefix.strip()] if static_prefix else []
    for lane_name, items in planner_output.lane_data.items():
        if lane_name == "static":
            continue
        for item in items:
            if isinstance(item, VirtualKnoxel):
                buffer.append(item.content)
            else:
                buffer.append(item.get_story_element())
    return "\n".join(part for part in buffer if part)
