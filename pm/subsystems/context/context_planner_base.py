import datetime
import logging
import math
import random
from enum import StrEnum
from types import SimpleNamespace
from typing import Type, List, Literal, Union, Dict, Tuple, Callable, Optional, Any

from pydantic import BaseModel, Field, ConfigDict

from pm.model.knoxel_core import KnoxelBase, KnoxelType
from pm.model.knoxel_enums import KnoxelSubtypeBase
from pm.model.knoxel_list import KnoxelList
from pm.model.mental_state_vectors import FullMentalState
from pm.model.knoxel_common import MemoryClusterKnoxel
from pm.model.knoxel_feature import Feature
from pm.model.knoxel_enums import ClusterType
from pm.subsystems.context.temporal_context_util import build_temporal_context
from pm.utils.emb_utils import cosine_sim
from pm.utils.token_utils import get_token_count

logger = logging.getLogger(__name__)


class DiversityStrategy(StrEnum):
    """
    Strategy for reducing near-duplicates and preserving topic breadth.

    Notes
    -----
    Disabled:
        No diversity stage is applied. Results are purely score/sort/sampling driven.

    CosineMMR:
        Maximal Marginal Relevance style selection. Prefer high-score items while
        penalizing candidates that are too similar to already selected items.

    BestPerCluster:
        Select the best representative per temporal or semantic cluster.
    """
    Disabled = "disabled"
    CosineMMR = "cosine_mmr"
    BestPerCluster = "best_per_cluster"


class TemporalSamplingMode(StrEnum):
    """
    High-level temporal sampling mode for a lane.
    """
    ScoreOnly = "score_only"
    PreserveHistory = "preserve_history"
    RecentWindow = "recent_window"


class SortingStrategy(StrEnum):
    """
    Final ordering policy after scoring and lane assignment.

    This ordering is applied after filtering, scoring, diversity, and sampling.
    """
    ScoreAsc = "score_asc"
    ScoreDesc = "score_desc"
    TimeAsc = "time_asc"
    TimeDesc = "time_desc"
    Random = "random"


class VirtualKnoxel(BaseModel):
    """
    Planner-generated synthetic context item.

    Intended uses:
    - static overwrite strings converted into planner output
    - time skip markers
    - system lifecycle markers
    - lane diagnostics or explicit context notes

    The planner treats these like ordinary output items, but they are never part of the
    input candidate pool unless explicitly created by the planner.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_turn: Literal["user", "assistant"]
    causal_timestamp_utc: datetime.datetime
    content: str

    @property
    def type(self) -> KnoxelType:
        return KnoxelType.VirtualKnoxel


class ContextPreFilterRules(BaseModel):
    """
    Hard filtering and explicit forcing rules applied before scoring.

    This is the stage that decides what a lane is even allowed to inspect.
    Use this for broad inclusion/exclusion constraints, not nuanced ranking.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    allowed_types: List[Union[Type[KnoxelBase], KnoxelType, KnoxelSubtypeBase]] = Field(
        default_factory=list,
        description="If non-empty, only these knoxel classes/types/subtypes are eligible."
    )
    banned_types: List[Union[Type[KnoxelBase], KnoxelType, KnoxelSubtypeBase]] = Field(
        default_factory=list,
        description="Explicitly excluded knoxel classes/types/subtypes."
    )
    temporal_scope_start: Optional[datetime.datetime] = Field(
        default=None,
        description="Optional inclusive temporal range start."
    )
    temporal_scope_end: Optional[datetime.datetime] = Field(
        default=None,
        description="Optional inclusive temporal range end."
    )
    causal_only: Optional[bool] = Field(
        default=None,
        description="If true keep only causal items; if false keep only non-causal items; if None allow both."
    )
    allow_virtual_knoxels: bool = Field(
        default=True,
        description="Allow planner-generated virtual blocks in output and post-processing."
    )

    begin_after_previous_lane: bool = Field(
        default=False,
        description="Use the last causal timestamp from the previous resolved lane as dynamic lower bound."
    )
    end_before_next_lane: bool = Field(
        default=False,
        description="Use the first causal timestamp from the next resolved lane as dynamic upper bound."
    )


class ContextScoreSettings(BaseModel):
    """
    Settings used to assign a relevance score to each candidate knoxel.

    A lane may use the global scoring settings or override them with a lane-specific
    variant. Keep this stage broad and compositional. The actual implementation can
    choose any internal normalization strategy.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    weighted_embeddings: List[Tuple[float, List[float]]] = Field(
        default_factory=list,
        description="Embeddings and their relative influence on retrieval/ranking."
    )
    weighted_embeddings_strings: List[Tuple[float, str]] = Field(
        default_factory=list,
        description="Natural language queries that should be embedded lazily."
    )
    weighted_mental_state: List[Tuple[float, Union[FullMentalState, List[float]]]] = Field(
        default_factory=list,
        description="Target mental states for similarity search. Uninteresting dimensions may be masked with NaN."
    )
    weighted_mental_delta: List[Tuple[float, Union[FullMentalState, List[float]]]] = Field(
        default_factory=list,
        description="Target mental deltas for similarity search. Uninteresting dimensions may be masked with NaN."
    )
    weighted_key_phrases: List[Tuple[float, str]] = Field(
        default_factory=list,
        description="Keywords or phrases and how strongly they should influence lexical matching."
    )

    required_source_ids: List[int] = Field(
        default_factory=list,
        description="Force these specific knoxel ids into candidate set or final output."
    )
    banned_source_ids: List[int] = Field(
        default_factory=list,
        description="Explicitly ban these source knoxel ids from the result."
    )

    lambdas_ban_knoxel: List[Callable[[KnoxelBase], bool]] = Field(
        default_factory=list,
        description="Predicates that completely ban a knoxel from consideration."
    )
    lambdas_force_knoxel: List[Callable[[KnoxelBase], bool]] = Field(
        default_factory=list,
        description="Predicates that force a knoxel into the lane result if possible."
    )
    lambdas_weight_knoxel: List[Callable[[KnoxelBase], float]] = Field(
        default_factory=list,
        description="Custom weighting functions applied before final sort/sampling."
    )
    focus_instructions: List[Tuple[float, str]] = Field(
        default_factory=list,
        description="Free-form planner hints for advanced or agentic retrieval."
    )

    temporal_decay_per_day: Optional[float] = Field(
        default=None,
        description="If set, multiply the score of non-dialogue items by this factor for each day away from the reference timestamp."
    )
    temporal_decay_reference: Literal["now", "candidate_pool_most_recent", "lane_most_recent_selected"] = Field(
        default="candidate_pool_most_recent",
        description="Reference timestamp used for temporal score decay."
    )
    temporal_decay_exempt_types: List[Union[Type[KnoxelBase], KnoxelType, KnoxelSubtypeBase]] = Field(
        default_factory=list,
        description="Types/subtypes exempt from temporal score decay. Dialogue and inner thought usually belong here."
    )

class ContextDiversitySettings(BaseModel):
    """
    De-duplication and diversity preservation.

    Notes
    -----
    This stage usually should not affect purely chronological lanes where full temporal
    continuity matters more than novelty.
    """
    strategy: DiversityStrategy = Field(
        default=DiversityStrategy.Disabled,
        description="Diversity strategy."
    )
    lambda_diversity: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Tradeoff between raw relevance and novelty when using MMR-like methods."
    )


class ContextSampleSettings(BaseModel):
    """
    Sampling policy used after scoring/diversity to fit the lane within its token budget.

    Notes
    -----
    The planner is expected to support both deterministic selection and stochastic
    temperature-based sampling.
    """
    sample_temperature: float = Field(
        default=0.0,
        description="How much the score affects selection. 0 = deterministic, higher = more exploratory."
    )
    sample_require_temporal_history: bool = Field(
        default=False,
        description="For historical lanes, attempt to preserve timeline continuity rather than only top scores."
    )
    sample_knoxel_type_distribution: List[Tuple[float, Union[Type[KnoxelBase], KnoxelType, KnoxelSubtypeBase]]] = Field(
        default_factory=list,
        description="Target distribution of output by knoxel class/type/subtype."
    )
    sample_enfore_knoxel_type_distribution: bool = Field(
        default=False,
        description="If true, fail to satisfy the lane fully rather than violating the requested type distribution."
    )
    sample_type_distribution_unit: Literal["tokens", "items"] = Field(
        default="tokens",
        description="Whether type distributions should be enforced approximately by token share or item count."
    )
    sample_temporal_mode: TemporalSamplingMode = Field(
        default=TemporalSamplingMode.ScoreOnly,
        description="High-level temporal sampling policy for the lane."
    )
    sample_require_contiguous_temporal_span: bool = Field(
        default=False,
        description="If true, enforce a contiguous temporal window once the lane anchor or first selected timestamp is known."
    )
    sample_resolve_cluster_to_children_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="For cluster/summary lanes, resolve this fraction of selected cluster budget down to original child knoxels."
    )
    sample_resolve_cluster_children_types: List[Union[Type[KnoxelBase], KnoxelType, KnoxelSubtypeBase]] = Field(
        default_factory=list,
        description="When resolving clusters to children, restrict expansion to these underlying types/subtypes."
    )
    sample_fill_shortfall_from_type_distribution: List[
        Tuple[
            Union[Type[KnoxelBase], KnoxelType, KnoxelSubtypeBase],
            Union[Type[KnoxelBase], KnoxelType, KnoxelSubtypeBase],
        ]
    ] = Field(
        default_factory=list,
        description="Fallback mapping used when one requested type is missing. Example: fill missing inner-thought budget with dialogue."
    )
    sample_max_type_to_reference_ratio: List[
        Tuple[
            float,
            Union[Type[KnoxelBase], KnoxelType, KnoxelSubtypeBase],
            List[Union[Type[KnoxelBase], KnoxelType, KnoxelSubtypeBase]],
        ]
    ] = Field(
        default_factory=list,
        description="Upper bound for a target type relative to one or more reference types. Example: mixed features <= 1.0 * (dialogue + inner thought)."
    )
    sample_expand_relevant_raw_to_topic_blocks: bool = Field(
        default=False,
        description="If true, selected raw features are expanded to their full topical cluster block for better generation coherence."
    )
    sample_min_raw_feature_block_count: int = Field(
        default=0,
        ge=0,
        description="If a selected topical raw block is smaller than this many features, pull adjacent topical clusters until the minimum is reached."
    )
    sample_min_raw_feature_block_tokens: int = Field(
        default=0,
        ge=0,
        description="If a selected topical raw block is smaller than this many tokens, pull adjacent topical clusters until the minimum is reached."
    )
    sample_generation_budget_slack_tokens: int = Field(
        default=0,
        ge=0,
        description="Optional extra lane-local token slack used for generation-quality block expansion."
    )

class ContextPostProcessSettings(BaseModel):
    """
    Planner-side augmentation applied after a lane has selected its concrete items.
    """
    additional_data_insert_timeskip_info: bool = Field(
        default=False,
        description="If there is a large temporal gap inside a lane, inject a synthetic time-skip note."
    )
    additional_data_insert_system_events: bool = Field(
        default=False,
        description="Inject synthetic notes for system lifecycle events if available."
    )
    additional_data_insert_timestamp_headers: bool = Field(
        default=False,
        description="Insert synthetic timestamp markers into the lane output."
    )
    additional_data_timestamp_header_mode: Literal["every_item", "day_change", "gap_only"] = Field(
        default="day_change",
        description="How often timestamp headers should be inserted when enabled."
    )


class ContextLaneSettings(BaseModel):
    """
    Configuration for a single lane in the context plan.

    A lane is one ordered slice of the final context, such as:
    - static character / architecture text
    - semi-static narratives and facts
    - historical summaries
    - recent raw features
    - codelet outputs
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="Unique lane name.")
    token_budget_pct: float = Field(
        description="Fraction of the total token budget allocated to this lane before redistribution.",
        ge=0.0,
        le=1.0,
    )
    sorting_strategy: SortingStrategy = Field(
        default=SortingStrategy.ScoreDesc,
        description="Final ordering policy within this lane."
    )
    overwrite_data: Optional[List[str]] = Field(
        default=None,
        description="Static lane content. If set, no knoxel retrieval is needed for this lane."
    )

    pre_filter: Optional[ContextPreFilterRules] = Field(
        default=None,
        description="Lane-specific pre-filter settings. If None, global settings are used."
    )
    scoring: Optional[ContextScoreSettings] = Field(
        default=None,
        description="Lane-specific scoring settings. If None, global settings are used."
    )
    diversity: Optional[ContextDiversitySettings] = Field(
        default=None,
        description="Lane-specific diversity settings. If None, global settings are used."
    )
    sampler: Optional[ContextSampleSettings] = Field(
        default=None,
        description="Lane-specific sampling settings. If None, global settings are used."
    )
    post_process: Optional[ContextPostProcessSettings] = Field(
        default=None,
        description="Lane-specific post-processing settings. If None, global settings are used."
    )
    token_budget_absolute: Optional[int] = Field(
        default=None,
        description="If set, use a fixed absolute token budget for this lane before percentage-based redistribution."
    )
    exclude_from_token_budget: bool = Field(
        default=False,
        description="If true, append this lane outside the dynamic token allocation pool. Useful for static overwrite text."
    )


class ContextPlannerSettings(BaseModel):
    """
    Top-level configuration object for a full context planning run.

    Design goals
    ------------
    - minimal simple case
    - lane-specific overrides
    - multi-pass redistribution if some lanes underfill
    - deterministic structure, even if individual stages use sampling
    """
    token_length: int = Field(description="Maximum total tokens allowed for the full context plan.")
    lanes: List[ContextLaneSettings] = Field(default_factory=list)
    ordered_lanes: List[ContextLaneSettings] = Field(
        default_factory=list,
        description="Processing order of lanes."
    )
    lane_priority: List[ContextLaneSettings] = Field(
        default_factory=list,
        description="Redistribution priority when leftover budget must be reassigned."
    )

    pre_filter: ContextPreFilterRules = Field(description="Global default pre-filter rules.")
    scoring: ContextScoreSettings = Field(description="Global default scoring settings.")
    diversity: ContextDiversitySettings = Field(description="Global default diversity settings.")
    sampler: ContextSampleSettings = Field(description="Global default sample settings.")
    post_process: ContextPostProcessSettings = Field(description="Global default post-process settings.")


class ContextPlannerOutput(BaseModel):
    """
    Final planner output.

    `lane_data` contains only the chosen items in their final lane order.
    `metadata` is intentionally free-form so the planner can emit debug, token usage,
    redistribution history, score summaries, or trace artifacts without changing the interface.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    lane_data: Dict[str, List[Union[KnoxelBase, VirtualKnoxel]]]
    metadata: Dict[str, Any]


class ContextLaneResolvedSettings(BaseModel):
    """
    Fully resolved lane configuration after inheriting global defaults.

    This internal helper object makes the runtime code easier to read and avoids repeated
    fallback checks throughout the planning loop.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    token_budget_pct: float
    sorting_strategy: SortingStrategy
    overwrite_data: Optional[List[str]]

    pre_filter: ContextPreFilterRules
    scoring: ContextScoreSettings
    diversity: ContextDiversitySettings
    sampler: ContextSampleSettings
    post_process: ContextPostProcessSettings


class ContextLaneRuntimeState(BaseModel):
    """
    Mutable per-lane runtime state used during planning.

    This object tracks:
    - resolved settings
    - assigned token budget
    - selected data
    - token usage
    - unresolved deficit/surplus
    - temporal boundaries discovered during selection
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    resolved: ContextLaneResolvedSettings

    token_budget_base: int = 0
    token_budget_current: int = 0
    token_budget_used: int = 0
    token_budget_missing: int = 0
    token_budget_surplus: int = 0

    selected_items: List[Union[KnoxelBase, VirtualKnoxel]] = Field(default_factory=list)
    selected_source_ids: List[int] = Field(default_factory=list)

    first_timestamp_utc: Optional[datetime.datetime] = None
    last_timestamp_utc: Optional[datetime.datetime] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)


class ContextPlanner:
    """
    Multi-pass context planner for knoxel-based memory systems.

    Overview
    --------
    The planner operates lane-by-lane. Each lane:
    1. resolves its effective settings,
    2. receives an initial token budget,
    3. selects context items,
    4. records token usage and temporal boundaries,
    5. optionally participates in later refinement passes.

    The full planning run then:
    - resolves dynamic temporal bounds between neighboring lanes,
    - reruns affected lanes,
    - redistributes unused tokens to underfilled high-priority lanes,
    - applies planner-level post-processing,
    - packages output and metadata.

    Important
    ---------
    This class intentionally contains:
    - a real orchestration method (`plan`)
    - named helper methods with concrete signatures
    - no actual retrieval/scoring implementation

    Once the helper methods are implemented, `plan()` should run end-to-end without
    requiring manual loop code elsewhere.
    """

    def plan(
        self,
        knoxels: List[KnoxelBase],
        settings: ContextPlannerSettings,
    ) -> ContextPlannerOutput:
        """
        Plan context across all configured lanes.

        Parameters
        ----------
        knoxels:
            Flat input pool of candidate knoxels. The planner is responsible for
            filtering, scoring, and allocating them across lanes.

        settings:
            Global planner settings plus ordered lane definitions.

        Returns
        -------
        ContextPlannerOutput
            Final lane data and free-form metadata.

        High-level algorithm
        --------------------
        1. Validate settings and normalize obvious inconsistencies.
        2. Build per-lane runtime state with inherited defaults.
        3. Compute initial token budgets from total token length and lane percentages.
        4. First pass: process lanes in order with currently known bounds.
        5. Resolve dynamic temporal constraints between neighboring lanes.
        6. Re-run lanes affected by dynamic temporal boundaries.
        7. Multi-pass redistribution: move unused budget to priority lanes that underfilled.
        8. Apply planner-wide post-processing and finalize output.
        """
        if not settings.ordered_lanes and settings.lanes:
            settings.ordered_lanes = list(settings.lanes)
        if not settings.lane_priority:
            settings.lane_priority = list(settings.ordered_lanes)

        self._validate_settings(settings)
        self._plan_knoxels_current = list(knoxels)

        lane_states = self._build_lane_runtime_states(settings)
        self._assign_initial_token_budgets(knoxels, lane_states, settings.token_length)

        self._run_lane_pass(
            knoxels=knoxels,
            lane_states=lane_states,
            settings=settings,
            lane_names=None,
            pass_name="main",
        )

        temporal_rerun_iteration = 1
        while True:
            affected_lane_names = self._resolve_neighbor_temporal_constraints(lane_states)
            if not affected_lane_names:
                break

            previous_temporal_signature = self._get_temporal_lane_signature(lane_states)
            self._clear_lane_outputs(lane_states, affected_lane_names)
            self._run_lane_pass(
                knoxels=knoxels,
                lane_states=lane_states,
                settings=settings,
                lane_names=affected_lane_names,
                pass_name=f"temporal_rerun_{temporal_rerun_iteration}",
            )
            current_temporal_signature = self._get_temporal_lane_signature(lane_states)
            if current_temporal_signature == previous_temporal_signature:
                break
            temporal_rerun_iteration += 1

        redistribution_round = 1
        while self._should_run_budget_redistribution(lane_states):
            if not self._redistribute_unused_budget(
                lane_states=lane_states,
                lane_priority=settings.lane_priority,
                total_token_limit=settings.token_length,
                redistribution_round=redistribution_round,
            ):
                break

            changed_lane_names = self._get_lanes_with_changed_budgets(lane_states)
            if not changed_lane_names:
                break

            self._clear_lane_outputs(lane_states, changed_lane_names)
            self._run_lane_pass(
                knoxels=knoxels,
                lane_states=lane_states,
                settings=settings,
                lane_names=changed_lane_names,
                pass_name=f"redistribution_{redistribution_round}",
            )

            temporal_rerun_iteration = 1
            while True:
                affected_lane_names = self._resolve_neighbor_temporal_constraints(lane_states)
                if not affected_lane_names:
                    break

                previous_temporal_signature = self._get_temporal_lane_signature(lane_states)
                self._clear_lane_outputs(lane_states, affected_lane_names)
                self._run_lane_pass(
                    knoxels=knoxels,
                    lane_states=lane_states,
                    settings=settings,
                    lane_names=affected_lane_names,
                    pass_name=f"redistribution_temporal_{redistribution_round}_{temporal_rerun_iteration}",
                )
                current_temporal_signature = self._get_temporal_lane_signature(lane_states)
                if current_temporal_signature == previous_temporal_signature:
                    break
                temporal_rerun_iteration += 1
            redistribution_round += 1

        self._run_global_post_process(
            lane_states=lane_states,
            settings=settings,
        )

        lane_data = self._materialize_lane_data(lane_states)
        metadata = self._build_output_metadata(
            lane_states=lane_states,
            settings=settings,
        )
        return ContextPlannerOutput(lane_data=lane_data, metadata=metadata)

    def _validate_settings(self, settings: ContextPlannerSettings) -> None:
        """
        Validate planner settings before any runtime state is created.

        Expected checks
        ---------------
        - lane names are unique
        - ordered_lanes is not empty
        - lane percentages are sane
        - lane_priority refers only to configured lanes
        - global token budget is positive
        - contradictory temporal flags are tolerated only if implementation supports them
        """
        if settings.token_length <= 0:
            raise ValueError("Context planner token_length must be positive.")
        if not settings.ordered_lanes:
            raise ValueError("Context planner requires at least one lane.")

        lane_names = [lane.name for lane in settings.ordered_lanes]
        if len(lane_names) != len(set(lane_names)):
            raise ValueError("Context planner lane names must be unique.")

        known_lane_names = set(lane_names)
        for lane in settings.lane_priority:
            if lane.name not in known_lane_names:
                raise ValueError(f"lane_priority contains unknown lane: {lane.name}")

    def _build_lane_runtime_states(
        self,
        settings: ContextPlannerSettings,
    ) -> List[ContextLaneRuntimeState]:
        """
        Create per-lane runtime state and resolve lane-local overrides against global defaults.

        Returns
        -------
        List[ContextLaneRuntimeState]
            Runtime lane states in `ordered_lanes` order.
        """
        return [
            ContextLaneRuntimeState(
                name=lane.name,
                resolved=self._resolve_lane_settings(lane, settings),
            )
            for lane in settings.ordered_lanes
        ]

    def _resolve_lane_settings(
        self,
        lane: ContextLaneSettings,
        settings: ContextPlannerSettings,
    ) -> ContextLaneResolvedSettings:
        """
        Merge a lane's optional overrides with planner-wide defaults.
        """
        return ContextLaneResolvedSettings(
            name=lane.name,
            token_budget_pct=lane.token_budget_pct,
            sorting_strategy=lane.sorting_strategy,
            overwrite_data=lane.overwrite_data,
            pre_filter=(lane.pre_filter or settings.pre_filter).model_copy(deep=True),
            scoring=(lane.scoring or settings.scoring).model_copy(deep=True),
            diversity=(lane.diversity or settings.diversity).model_copy(deep=True),
            sampler=(lane.sampler or settings.sampler).model_copy(deep=True),
            post_process=(lane.post_process or settings.post_process).model_copy(deep=True),
        )

    def _assign_initial_token_budgets(
        self,
        knoxels: List[KnoxelBase],
        lane_states: List[ContextLaneRuntimeState],
        total_token_budget: int,
    ) -> None:
        """
        Compute the base token budget per lane from the total token budget.

        Notes
        -----
        Implementation should handle rounding carefully so the sum of lane budgets
        does not exceed `total_token_budget`.
        """
        remaining_budget = total_token_budget
        for index, lane_state in enumerate(lane_states):
            if index == len(lane_states) - 1:
                base_budget = max(0, remaining_budget)
            else:
                base_budget = min(int(total_token_budget * lane_state.resolved.token_budget_pct), remaining_budget)
                if lane_state.resolved.sampler.sample_temporal_mode != TemporalSamplingMode.PreserveHistory:
                    effective_prefilter = self._resolve_effective_prefilter(
                        lane_state=lane_state,
                        previous_lane_state=None,
                        next_lane_state=None,
                    )
                    candidates = self._build_lane_candidates(knoxels, effective_prefilter)
                    max_possible_tokens = sum([self._estimate_knoxel_tokens(k) for k in candidates])
                    base_budget = min(base_budget, max_possible_tokens)

            lane_state.token_budget_base = base_budget
            lane_state.token_budget_current = base_budget
            remaining_budget -= base_budget

    def _run_lane_pass(
        self,
        knoxels: List[KnoxelBase],
        lane_states: List[ContextLaneRuntimeState],
        settings: ContextPlannerSettings,
        lane_names: Optional[List[str]],
        pass_name: str,
    ) -> None:
        """
        Process one full pass over either all lanes or a named subset.

        Parameters
        ----------
        lane_names:
            If None, process all lanes. Otherwise only process the named lanes.
        """
        selected_lane_names = set(lane_names) if lane_names is not None else None
        for index, lane_state in enumerate(lane_states):
            if selected_lane_names is not None and lane_state.name not in selected_lane_names:
                continue
            self._process_single_lane(
                all_knoxels=knoxels,
                lane_state=lane_state,
                previous_lane_state=lane_states[index - 1] if index > 0 else None,
                next_lane_state=lane_states[index + 1] if index + 1 < len(lane_states) else None,
                pass_name=pass_name,
            )

    def _process_single_lane(
        self,
        all_knoxels: List[KnoxelBase],
        lane_state: ContextLaneRuntimeState,
        previous_lane_state: Optional[ContextLaneRuntimeState],
        next_lane_state: Optional[ContextLaneRuntimeState],
        pass_name: str,
    ) -> None:
        """
        Process one lane from start to finish.

        Recommended flow
        ----------------
        1. If `overwrite_data` is present, convert it directly into VirtualKnoxels.
        2. Resolve effective temporal bounds for this lane.
        3. Build candidate pool through pre-filtering.
        4. Score candidates.
        5. Sort candidates.
        6. Apply diversity stage.
        7. Sample/select to lane token budget.
        8. Run lane-level post-processing.
        9. Measure token usage and update timestamps/metadata.
        """
        if lane_state.resolved.overwrite_data is not None:
            lane_items: List[Union[KnoxelBase, VirtualKnoxel]] = self._build_virtual_knoxels_from_overwrite_data(lane_state)
        else:
            if (
                lane_state.resolved.pre_filter.end_before_next_lane
                and next_lane_state is not None
                and next_lane_state.first_timestamp_utc is None
                and not (previous_lane_state is not None and previous_lane_state.first_timestamp_utc is not None)
            ):
                lane_state.metadata["candidate_count"] = 0
                lane_state.metadata["selected_count"] = 0
                lane_state.metadata["selected_scores"] = []
                lane_state.metadata["deferred_for_temporal_bound"] = True
                lane_items = []
            else:
                lane_state.metadata["deferred_for_temporal_bound"] = False
                effective_prefilter = self._resolve_effective_prefilter(
                    lane_state=lane_state,
                    previous_lane_state=previous_lane_state,
                    next_lane_state=next_lane_state,
                )
                lane_state.metadata["effective_prefilter"] = effective_prefilter
                candidates = self._build_lane_candidates(all_knoxels, effective_prefilter)
                scored_candidates = self._score_lane_candidates(candidates, lane_state.resolved.scoring)
                sorted_candidates = self._sort_scored_candidates(scored_candidates, lane_state.resolved.sorting_strategy)
                diversified_candidates = self._apply_diversity_stage(
                    scored_candidates=sorted_candidates,
                    diversity=lane_state.resolved.diversity,
                    sorting_strategy=lane_state.resolved.sorting_strategy,
                )
                lane_items = self._sample_to_lane_budget(diversified_candidates, lane_state)

        lane_items = self._run_lane_post_process(lane_items, lane_state)
        lane_items = self._clip_output_items_to_budget(lane_items, self._effective_lane_token_budget(lane_state))
        token_used, first_timestamp, last_timestamp = self._measure_lane_output(lane_items, lane_state)
        self._update_lane_runtime_metrics(
            lane_state=lane_state,
            lane_items=lane_items,
            token_used=token_used,
            first_timestamp=first_timestamp,
            last_timestamp=last_timestamp,
            pass_name=pass_name,
        )

    def _resolve_effective_prefilter(
        self,
        lane_state: ContextLaneRuntimeState,
        previous_lane_state: Optional[ContextLaneRuntimeState],
        next_lane_state: Optional[ContextLaneRuntimeState],
    ) -> ContextPreFilterRules:
        """
        Build the effective pre-filter for one lane, including dynamic temporal bounds.

        Dynamic temporal logic
        ----------------------
        - If `begin_after_previous_lane` is enabled and the previous lane has a known
          last timestamp, use that as lower bound.
        - If `end_before_next_lane` is enabled and the next lane has a known first
          timestamp, use that as upper bound.
        """
        boundary_epsilon = datetime.timedelta(microseconds=1)
        effective_prefilter = lane_state.resolved.pre_filter.model_copy(deep=True)
        if effective_prefilter.begin_after_previous_lane and previous_lane_state and previous_lane_state.last_timestamp_utc:
            effective_prefilter.temporal_scope_start = previous_lane_state.last_timestamp_utc + boundary_epsilon
        if effective_prefilter.end_before_next_lane:
            if (
                previous_lane_state
                and previous_lane_state.resolved.sampler.sample_temporal_mode == TemporalSamplingMode.RecentWindow
                and previous_lane_state.first_timestamp_utc
            ):
                effective_prefilter.temporal_scope_end = previous_lane_state.first_timestamp_utc - boundary_epsilon
            elif next_lane_state and next_lane_state.first_timestamp_utc:
                effective_prefilter.temporal_scope_end = next_lane_state.first_timestamp_utc - boundary_epsilon
            elif previous_lane_state and previous_lane_state.first_timestamp_utc:
                effective_prefilter.temporal_scope_end = previous_lane_state.first_timestamp_utc - boundary_epsilon
        return effective_prefilter

    def _build_lane_candidates(
        self,
        knoxels: List[KnoxelBase],
        effective_prefilter: ContextPreFilterRules,
    ) -> List[KnoxelBase]:
        """
        Build the candidate pool for a lane after hard pre-filtering.
        """
        return [knoxel for knoxel in knoxels if self._knoxel_matches_prefilter(knoxel, effective_prefilter)]

    def _score_lane_candidates(
        self,
        candidates: List[KnoxelBase],
        scoring: ContextScoreSettings,
    ) -> List[Tuple[KnoxelBase, float, Dict[str, float]]]:
        """
        Score candidate knoxels.

        Returns
        -------
        List[Tuple[KnoxelBase, float, Dict[str, float]]]
            Each tuple contains:
            - the candidate knoxel
            - total score
            - score breakdown metadata
        """
        scored_candidates: List[Tuple[KnoxelBase, float, Dict[str, float]]] = []
        reference_timestamp = None
        if candidates:
            reference_timestamp = max(self._get_knoxel_timestamp(candidate) for candidate in candidates)
        for candidate in candidates:
            if candidate.id in scoring.banned_source_ids:
                continue
            if any(ban_lambda(candidate) for ban_lambda in scoring.lambdas_ban_knoxel):
                continue

            total_score = 0.0
            breakdown: Dict[str, float] = {}

            for index, (weight, target_embedding) in enumerate(scoring.weighted_embeddings):
                similarity = self._cosine_similarity(candidate.embedding or [], target_embedding or [])
                breakdown[f"embedding_{index}"] = similarity
                total_score += weight * similarity

            candidate_state = self._coerce_mental_vector(getattr(candidate, "mental_state_appraisal", None))
            for index, (weight, target_state) in enumerate(scoring.weighted_mental_state):
                similarity = self._cosine_similarity(candidate_state, self._coerce_mental_vector(target_state))
                breakdown[f"mental_state_{index}"] = similarity
                total_score += weight * similarity

            candidate_delta = self._coerce_mental_vector(getattr(candidate, "mental_state_delta", None))
            for index, (weight, target_delta) in enumerate(scoring.weighted_mental_delta):
                similarity = self._cosine_similarity(candidate_delta, self._coerce_mental_vector(target_delta))
                breakdown[f"mental_delta_{index}"] = similarity
                total_score += weight * similarity

            candidate_text = (candidate.content or "").lower()
            for index, (weight, phrase) in enumerate(scoring.weighted_key_phrases):
                phrase_hit = 1.0 if phrase.lower() in candidate_text else 0.0
                breakdown[f"key_phrase_{index}"] = phrase_hit
                total_score += weight * phrase_hit

            for index, weight_lambda in enumerate(scoring.lambdas_weight_knoxel):
                lambda_score = float(weight_lambda(candidate))
                breakdown[f"lambda_weight_{index}"] = lambda_score
                total_score += lambda_score

            if any(force_lambda(candidate) for force_lambda in scoring.lambdas_force_knoxel):
                breakdown["forced"] = 1.0
                total_score += 1_000_000.0

            if candidate.id in scoring.required_source_ids:
                breakdown["required_source_id"] = 1.0
                total_score += 1_000_000.0

            if (
                scoring.temporal_decay_per_day is not None
                and reference_timestamp is not None
                and not any(self._matches_type_spec(candidate, type_spec) for type_spec in scoring.temporal_decay_exempt_types)
            ):
                age_days = max(
                    0.0,
                    (reference_timestamp - self._get_knoxel_timestamp(candidate)).total_seconds() / 86400.0,
                )
                decay = scoring.temporal_decay_per_day ** age_days
                breakdown["temporal_decay"] = decay
                total_score *= decay

            scored_candidates.append((candidate, total_score, breakdown))
        return scored_candidates

    def _sort_scored_candidates(
        self,
        scored_candidates: List[Tuple[KnoxelBase, float, Dict[str, float]]],
        strategy: SortingStrategy,
    ) -> List[Tuple[KnoxelBase, float, Dict[str, float]]]:
        """
        Sort candidates according to the lane strategy.
        """
        if strategy == SortingStrategy.ScoreAsc:
            return sorted(scored_candidates, key=lambda item: (item[1], item[0].id))
        if strategy == SortingStrategy.ScoreDesc:
            return sorted(scored_candidates, key=lambda item: (-item[1], item[0].id))
        if strategy == SortingStrategy.TimeAsc:
            return sorted(scored_candidates, key=lambda item: (self._get_knoxel_timestamp(item[0]), item[0].id))
        if strategy == SortingStrategy.TimeDesc:
            return sorted(scored_candidates, key=lambda item: (self._get_knoxel_timestamp(item[0]), item[0].id), reverse=True)
        randomized = list(scored_candidates)
        random.Random(0).shuffle(randomized)
        return randomized

    def _apply_diversity_stage(
        self,
        scored_candidates: List[Tuple[KnoxelBase, float, Dict[str, float]]],
        diversity: ContextDiversitySettings,
        sorting_strategy: SortingStrategy,
    ) -> List[Tuple[KnoxelBase, float, Dict[str, float]]]:
        """
        Apply optional diversity/de-duplication stage.

        Notes
        -----
        For strongly chronological lanes, implementations may choose to bypass or
        weaken diversity so temporal continuity is not destroyed.
        """
        return scored_candidates

    def _sample_to_lane_budget(
        self,
        scored_candidates: List[Tuple[KnoxelBase, float, Dict[str, float]]],
        lane_state: ContextLaneRuntimeState,
    ) -> List[Union[KnoxelBase, VirtualKnoxel]]:
        """
        Select the final set of items for this lane under current token budget.

        Notes
        -----
        This is where:
        - deterministic top-k
        - stochastic temperature sampling
        - type distribution targets
        - complete timeline requirements
        are expected to be enforced.
        """
        token_budget = self._effective_lane_token_budget(lane_state)
        temporal_mode = lane_state.resolved.sampler.sample_temporal_mode
        selection_pool = list(scored_candidates)
        if temporal_mode == TemporalSamplingMode.RecentWindow:
            selected_entries = self._select_recent_window_entries(selection_pool, lane_state)
            selected_entries = self._sort_scored_candidates(selected_entries, lane_state.resolved.sorting_strategy)
            selected_entries = self._clip_entries_to_budget(selected_entries, token_budget)

            selected_items = [entry[0] for entry in selected_entries]
            selected_items = self._expand_selected_raw_to_topic_blocks(selected_items, lane_state)
            selected_scores = [
                {
                    "id": knoxel.id,
                    "score": score,
                    "tokens": self._estimate_knoxel_tokens(knoxel),
                    "breakdown": breakdown,
                }
                for knoxel, score, breakdown in selected_entries
            ]
            lane_state.metadata["selected_scores"] = selected_scores
            lane_state.metadata["candidate_count"] = len(scored_candidates)
            lane_state.metadata["selected_count"] = len(selected_entries)
            return selected_items
        if temporal_mode == TemporalSamplingMode.PreserveHistory:
            selected_items, adapter_candidate_count = self._select_preserve_history_items(selection_pool, lane_state)
            selected_items = self._expand_selected_raw_to_topic_blocks(selected_items, lane_state)
            selected_scores = []
            scored_by_id = {entry[0].id: entry for entry in scored_candidates}
            for item in selected_items:
                scored_entry = scored_by_id.get(item.id)
                selected_scores.append(
                    {
                        "id": item.id,
                        "score": scored_entry[1] if scored_entry is not None else 0.0,
                        "tokens": self._estimate_knoxel_tokens(item),
                        "breakdown": scored_entry[2] if scored_entry is not None else {"preserve_history_adapter": 1.0},
                    }
                )
            lane_state.metadata["selected_scores"] = selected_scores
            lane_state.metadata["candidate_count"] = adapter_candidate_count
            lane_state.metadata["selected_count"] = len(selected_items)
            return selected_items

        selected_entries: List[Tuple[KnoxelBase, float, Dict[str, float]]] = []
        selected_ids: set[int] = set()

        if lane_state.resolved.sampler.sample_knoxel_type_distribution:
            selected_entries = self._select_with_type_distribution(selection_pool, lane_state)
            selected_ids = {entry[0].id for entry in selected_entries}

        token_used = sum(self._estimate_knoxel_tokens(entry[0]) for entry in selected_entries)
        for entry in selection_pool:
            knoxel = entry[0]
            if knoxel.id in selected_ids:
                continue
            item_tokens = self._estimate_knoxel_tokens(knoxel)
            if selected_entries and token_used + item_tokens > token_budget:
                continue
            if not self._passes_type_ratio_constraints(selected_entries, entry, lane_state):
                continue
            selected_entries.append(entry)
            selected_ids.add(knoxel.id)
            token_used += item_tokens
            if token_used >= token_budget:
                break

        if lane_state.resolved.sampler.sample_resolve_cluster_to_children_pct > 0.0:
            selected_entries = self._resolve_selected_clusters_to_children(selected_entries, scored_candidates, lane_state)

        if temporal_mode in (TemporalSamplingMode.RecentWindow, TemporalSamplingMode.PreserveHistory):
            selected_entries = self._sort_scored_candidates(selected_entries, lane_state.resolved.sorting_strategy)

        selected_entries = self._clip_entries_to_budget(selected_entries, token_budget)

        selected_items = [entry[0] for entry in selected_entries]
        selected_items = self._expand_selected_raw_to_topic_blocks(selected_items, lane_state)
        selected_scores = [
            {
                "id": knoxel.id,
                "score": score,
                "tokens": self._estimate_knoxel_tokens(knoxel),
                "breakdown": breakdown,
            }
            for knoxel, score, breakdown in selected_entries
        ]
        lane_state.metadata["selected_scores"] = selected_scores
        lane_state.metadata["candidate_count"] = len(scored_candidates)
        lane_state.metadata["selected_count"] = len(selected_entries)
        return selected_items

    def _select_preserve_history_items(
        self,
        scored_candidates: List[Tuple[KnoxelBase, float, Dict[str, float]]],
        lane_state: ContextLaneRuntimeState,
    ) -> Tuple[List[KnoxelBase], int]:
        sampler = lane_state.resolved.sampler
        scoring = lane_state.resolved.scoring

        unsupported_sampler_reasons = []
        if sampler.sample_temperature != 0.0:
            unsupported_sampler_reasons.append("sample_temperature")
        if sampler.sample_knoxel_type_distribution:
            unsupported_sampler_reasons.append("sample_knoxel_type_distribution")
        if sampler.sample_enfore_knoxel_type_distribution:
            unsupported_sampler_reasons.append("sample_enfore_knoxel_type_distribution")
        if sampler.sample_type_distribution_unit != "tokens":
            unsupported_sampler_reasons.append("sample_type_distribution_unit")
        if sampler.sample_resolve_cluster_children_types:
            unsupported_sampler_reasons.append("sample_resolve_cluster_children_types")
        if sampler.sample_fill_shortfall_from_type_distribution:
            unsupported_sampler_reasons.append("sample_fill_shortfall_from_type_distribution")
        if sampler.sample_max_type_to_reference_ratio:
            unsupported_sampler_reasons.append("sample_max_type_to_reference_ratio")
        if unsupported_sampler_reasons:
            raise NotImplementedError(
                f"PreserveHistory via temporal_context_util does not support: {', '.join(unsupported_sampler_reasons)}"
            )

        unsupported_scoring_reasons = []
        if len(scoring.weighted_embeddings) != 1:
            unsupported_scoring_reasons.append("weighted_embeddings")
        if scoring.weighted_embeddings_strings:
            unsupported_scoring_reasons.append("weighted_embeddings_strings")
        if scoring.weighted_mental_state:
            unsupported_scoring_reasons.append("weighted_mental_state")
        if scoring.weighted_mental_delta:
            unsupported_scoring_reasons.append("weighted_mental_delta")
        if scoring.weighted_key_phrases:
            unsupported_scoring_reasons.append("weighted_key_phrases")
        if scoring.required_source_ids:
            unsupported_scoring_reasons.append("required_source_ids")
        if scoring.banned_source_ids:
            unsupported_scoring_reasons.append("banned_source_ids")
        if scoring.lambdas_ban_knoxel:
            unsupported_scoring_reasons.append("lambdas_ban_knoxel")
        if scoring.lambdas_force_knoxel:
            unsupported_scoring_reasons.append("lambdas_force_knoxel")
        if scoring.lambdas_weight_knoxel:
            unsupported_scoring_reasons.append("lambdas_weight_knoxel")
        if scoring.focus_instructions:
            unsupported_scoring_reasons.append("focus_instructions")
        if scoring.temporal_decay_per_day is not None:
            unsupported_scoring_reasons.append("temporal_decay_per_day")
        if unsupported_scoring_reasons:
            raise NotImplementedError(
                f"PreserveHistory via temporal_context_util does not support: {', '.join(unsupported_scoring_reasons)}"
            )

        if lane_state.resolved.sorting_strategy != SortingStrategy.TimeAsc:
            raise NotImplementedError("PreserveHistory via temporal_context_util currently requires SortingStrategy.TimeAsc")

        effective_prefilter: ContextPreFilterRules = lane_state.metadata.get("effective_prefilter")
        if effective_prefilter is None:
            raise Exception("PreserveHistory requires an effective prefilter in lane metadata")

        temporal_features = [
            knoxel
            for knoxel in self._plan_knoxels_current
            if isinstance(knoxel, Feature)
            and (effective_prefilter.temporal_scope_start is None or knoxel.timestamp_world_begin >= effective_prefilter.temporal_scope_start)
            and (effective_prefilter.temporal_scope_end is None or knoxel.timestamp_world_begin <= effective_prefilter.temporal_scope_end)
            and (effective_prefilter.causal_only is None or bool(getattr(knoxel, "causal", False)) == effective_prefilter.causal_only)
        ]
        temporal_memories = [
            knoxel
            for knoxel in self._plan_knoxels_current
            if isinstance(knoxel, MemoryClusterKnoxel)
            and knoxel.cluster_type == ClusterType.Temporal
            and (effective_prefilter.temporal_scope_start is None or knoxel.timestamp_world_end >= effective_prefilter.temporal_scope_start)
            and (effective_prefilter.temporal_scope_end is None or knoxel.timestamp_world_begin <= effective_prefilter.temporal_scope_end)
        ]
        temporal_ghost = SimpleNamespace(
            all_features=temporal_features,
            all_episodic_memories=temporal_memories,
        )
        adapter_candidate_count = len(temporal_features) + len(temporal_memories)
        lane_state.metadata["preserve_history_temporal_feature_count"] = len(temporal_features)
        lane_state.metadata["preserve_history_temporal_memory_count"] = len(temporal_memories)

        if lane_state.token_budget_current <= 0 or adapter_candidate_count == 0:
            return [], adapter_candidate_count

        _, target_embedding = scoring.weighted_embeddings[0]
        try:
            return build_temporal_context(
                ghost=temporal_ghost,
                embedding=target_embedding,
                ratio=sampler.sample_resolve_cluster_to_children_pct,
                max_tokens=lane_state.token_budget_current,
                require_temporal_history=sampler.sample_require_temporal_history,
                require_contiguous_temporal_span=sampler.sample_require_contiguous_temporal_span,
                run_final_checks=True,
            ), adapter_candidate_count
        except Exception:
            logger.exception(
                "\n".join(
                    [
                        "PreserveHistory adapter failure",
                        f"lane={lane_state.name}",
                        f"token_budget_current={lane_state.token_budget_current}",
                        f"token_budget_base={lane_state.token_budget_base}",
                        f"sample_resolve_cluster_to_children_pct={sampler.sample_resolve_cluster_to_children_pct}",
                        f"sample_require_temporal_history={sampler.sample_require_temporal_history}",
                        f"sample_require_contiguous_temporal_span={sampler.sample_require_contiguous_temporal_span}",
                        f"prefilter_start={effective_prefilter.temporal_scope_start}",
                        f"prefilter_end={effective_prefilter.temporal_scope_end}",
                        f"temporal_features={len(temporal_features)}",
                        f"temporal_memories={len(temporal_memories)}",
                        f"adapter_candidate_count={adapter_candidate_count}",
                    ]
                )
            )
            raise

    def _select_recent_window_entries(
        self,
        scored_candidates: List[Tuple[KnoxelBase, float, Dict[str, float]]],
        lane_state: ContextLaneRuntimeState,
    ) -> List[Tuple[KnoxelBase, float, Dict[str, float]]]:
        """
        Select a contiguous suffix of the most recent candidates that fits in budget.

        For recent lanes, continuity matters more than score shaping or type balancing:
        the output should represent the newest connected slice of the timeline rather
        than a scattered sample from across history.
        """
        if not scored_candidates:
            return []

        token_budget = lane_state.token_budget_current
        chronological_entries = sorted(
            scored_candidates,
            key=lambda item: (self._get_knoxel_timestamp(item[0]), item[0].id),
        )

        selected_entries: List[Tuple[KnoxelBase, float, Dict[str, float]]] = []
        token_used = 0
        for entry in reversed(chronological_entries):
            item_tokens = self._estimate_knoxel_tokens(entry[0])
            if selected_entries and token_used + item_tokens > token_budget:
                break
            selected_entries.append(entry)
            token_used += item_tokens
            if token_used >= token_budget:
                break

        selected_entries.reverse()
        return selected_entries

    def _build_virtual_knoxels_from_overwrite_data(
        self,
        lane_state: ContextLaneRuntimeState,
    ) -> List[VirtualKnoxel]:
        """
        Convert static overwrite strings into VirtualKnoxels.

        This allows static lanes to flow through the same output structure as all
        other lanes.
        """
        now = datetime.datetime.utcnow()
        return [
            VirtualKnoxel(
                prompt_turn="assistant",
                causal_timestamp_utc=now + datetime.timedelta(seconds=index),
                content=content,
            )
            for index, content in enumerate(lane_state.resolved.overwrite_data or [])
        ]

    def _run_lane_post_process(
        self,
        lane_items: List[Union[KnoxelBase, VirtualKnoxel]],
        lane_state: ContextLaneRuntimeState,
    ) -> List[Union[KnoxelBase, VirtualKnoxel]]:
        """
        Apply lane-level synthetic augmentation such as time-skip markers or system notes.
        """
        if not lane_state.resolved.post_process.additional_data_insert_timestamp_headers:
            return lane_items

        header_items: List[Union[KnoxelBase, VirtualKnoxel]] = []
        previous_day = None
        for item in lane_items:
            timestamp = self._get_output_item_timestamp(item)
            if timestamp is not None:
                current_day = timestamp.date()
                should_insert = (
                    lane_state.resolved.post_process.additional_data_timestamp_header_mode == "every_item"
                    or previous_day is None
                    or (
                        lane_state.resolved.post_process.additional_data_timestamp_header_mode == "day_change"
                        and current_day != previous_day
                    )
                )
                if should_insert:
                    header_items.append(
                        VirtualKnoxel(
                            prompt_turn="assistant",
                            causal_timestamp_utc=timestamp,
                            content=f"[{timestamp.strftime('%Y-%m-%d %H:%M UTC')}]",
                        )
                    )
                previous_day = current_day
            header_items.append(item)
        return header_items

    def _measure_lane_output(
        self,
        lane_items: List[Union[KnoxelBase, VirtualKnoxel]],
        lane_state: ContextLaneRuntimeState,
    ) -> Tuple[int, Optional[datetime.datetime], Optional[datetime.datetime]]:
        """
        Measure:
        - total tokens used by the lane
        - first timestamp in output
        - last timestamp in output
        """
        if not lane_items:
            return 0, None, None

        token_used = 0
        timestamps: List[datetime.datetime] = []
        for item in lane_items:
            if isinstance(item, VirtualKnoxel):
                token_used += self._estimate_text_tokens(item.content)
                timestamps.append(item.causal_timestamp_utc)
            else:
                token_used += self._estimate_knoxel_tokens(item)
                timestamps.append(self._get_knoxel_timestamp(item))
        return token_used, min(timestamps), max(timestamps)

    def _update_lane_runtime_metrics(
        self,
        lane_state: ContextLaneRuntimeState,
        lane_items: List[Union[KnoxelBase, VirtualKnoxel]],
        token_used: int,
        first_timestamp: Optional[datetime.datetime],
        last_timestamp: Optional[datetime.datetime],
        pass_name: str,
    ) -> None:
        """
        Update mutable per-lane runtime metrics after selection.
        """
        lane_state.selected_items = lane_items
        lane_state.selected_source_ids = [item.id for item in lane_items if isinstance(item, KnoxelBase)]
        lane_state.token_budget_used = token_used
        lane_state.token_budget_missing = max(0, token_used - lane_state.token_budget_current)
        lane_state.token_budget_surplus = max(0, lane_state.token_budget_current - token_used)
        lane_state.first_timestamp_utc = first_timestamp
        lane_state.last_timestamp_utc = last_timestamp
        lane_state.metadata["last_pass_name"] = pass_name

    def _resolve_neighbor_temporal_constraints(
        self,
        lane_states: List[ContextLaneRuntimeState],
    ) -> List[str]:
        """
        Determine which lanes need a second pass because neighbor-derived temporal bounds changed.

        Returns
        -------
        List[str]
            Names of lanes that must be rerun.
        """
        affected_lane_names: List[str] = []
        for index, lane_state in enumerate(lane_states):
            previous_lane = lane_states[index - 1] if index > 0 else None
            next_lane = lane_states[index + 1] if index + 1 < len(lane_states) else None
            if lane_state.resolved.pre_filter.begin_after_previous_lane and previous_lane and previous_lane.last_timestamp_utc:
                affected_lane_names.append(lane_state.name)
            if (
                lane_state.resolved.pre_filter.end_before_next_lane
                and (
                    (next_lane and next_lane.first_timestamp_utc)
                    or (previous_lane and previous_lane.first_timestamp_utc)
                )
            ):
                affected_lane_names.append(lane_state.name)
        return sorted(set(affected_lane_names))

    def _get_temporal_lane_signature(
        self,
        lane_states: List[ContextLaneRuntimeState],
    ) -> Tuple[Tuple[str, Optional[datetime.datetime], Optional[datetime.datetime], Tuple[int, ...]], ...]:
        """
        Summarize the temporal shape of the current lane outputs.

        This lets the planner iterate temporal reruns until the actual boundaries stop
        changing, not merely until the same lane names remain eligible for rerun.
        """
        signature: List[Tuple[str, Optional[datetime.datetime], Optional[datetime.datetime], Tuple[int, ...]]] = []
        for lane_state in lane_states:
            if not (
                lane_state.resolved.pre_filter.begin_after_previous_lane
                or lane_state.resolved.pre_filter.end_before_next_lane
                or lane_state.resolved.sampler.sample_temporal_mode in (TemporalSamplingMode.RecentWindow, TemporalSamplingMode.PreserveHistory)
            ):
                continue
            signature.append(
                (
                    lane_state.name,
                    lane_state.first_timestamp_utc,
                    lane_state.last_timestamp_utc,
                    tuple(lane_state.selected_source_ids),
                )
            )
        return tuple(signature)

    def _clear_lane_outputs(
        self,
        lane_states: List[ContextLaneRuntimeState],
        lane_names: List[str],
    ) -> None:
        """
        Clear previously selected items and usage metrics for the named lanes before rerun.
        """
        for lane_state in lane_states:
            if lane_state.name not in lane_names:
                continue
            lane_state.selected_items = []
            lane_state.selected_source_ids = []
            lane_state.token_budget_used = 0
            lane_state.token_budget_missing = 0
            lane_state.token_budget_surplus = 0
            lane_state.first_timestamp_utc = None
            lane_state.last_timestamp_utc = None
            lane_state.metadata["selected_scores"] = []
            lane_state.metadata["candidate_count"] = 0
            lane_state.metadata["selected_count"] = 0

    def _should_run_budget_redistribution(
        self,
        lane_states: List[ContextLaneRuntimeState],
    ) -> bool:
        """
        Decide whether another redistribution pass is worth running.

        A typical implementation would return True if:
        - at least one lane has surplus budget
        - at least one lane is meaningfully underfilled
        """
        # Temporal split lanes derive meaning from their fixed relative budgets:
        # growing a recent suffix after the historical cutoff has been established
        # can move the cutoff and invalidate the preserved-history partition.
        if any(
            lane_state.resolved.sampler.sample_temporal_mode in (TemporalSamplingMode.RecentWindow, TemporalSamplingMode.PreserveHistory)
            for lane_state in lane_states
        ):
            return False

        has_surplus = any(lane_state.token_budget_surplus > 0 for lane_state in lane_states)
        has_absorber = any(
            lane_state.metadata.get("candidate_count", 0) > lane_state.metadata.get("selected_count", 0)
            for lane_state in lane_states
        )
        already_redistributed = any(lane_state.metadata.get("redistribution_round") for lane_state in lane_states)
        return has_surplus and has_absorber and not already_redistributed

    def _redistribute_unused_budget(
        self,
        lane_states: List[ContextLaneRuntimeState],
        lane_priority: List[ContextLaneSettings],
        total_token_limit: int,
        redistribution_round: int,
    ) -> bool:
        """
        Reassign unused tokens from underfilled low-priority lanes to lanes that can still absorb more context.

        Returns
        -------
        bool
            True if any lane's token budget changed.
        """
        surplus_pool = sum(lane_state.token_budget_surplus for lane_state in lane_states)
        if surplus_pool <= 0:
            return False

        changed = False
        priority_names = [lane.name for lane in lane_priority] or [lane_state.name for lane_state in lane_states]
        lane_map = {lane_state.name: lane_state for lane_state in lane_states}
        for lane_state in lane_states:
            lane_state.metadata["budget_changed"] = False
            if lane_state.token_budget_surplus > 0:
                lane_state.token_budget_current = lane_state.token_budget_used

        for lane_name in priority_names:
            if surplus_pool <= 0:
                break
            lane_state = lane_map.get(lane_name)
            if lane_state is None:
                continue
            if lane_state.metadata.get("candidate_count", 0) <= lane_state.metadata.get("selected_count", 0):
                continue

            lane_state.token_budget_current += surplus_pool
            lane_state.metadata["budget_changed"] = True
            lane_state.metadata["redistribution_round"] = redistribution_round
            surplus_pool = 0
            changed = True

        return changed

    def _get_lanes_with_changed_budgets(
        self,
        lane_states: List[ContextLaneRuntimeState],
    ) -> List[str]:
        """
        Return the names of lanes whose token budgets changed in the most recent redistribution round.
        """
        return [lane_state.name for lane_state in lane_states if lane_state.metadata.get("budget_changed")]

    def _run_global_post_process(
        self,
        lane_states: List[ContextLaneRuntimeState],
        settings: ContextPlannerSettings,
    ) -> None:
        """
        Apply planner-wide post-processing after all lane passes are complete.

        Good place for:
        - cross-lane cleanup
        - final global time-skip notes
        - planner-wide metadata consistency checks
        """
        return None

    def _materialize_lane_data(
        self,
        lane_states: List[ContextLaneRuntimeState],
    ) -> Dict[str, List[Union[KnoxelBase, VirtualKnoxel]]]:
        """
        Convert runtime lane states into the final output dictionary.
        """
        return {lane_state.name: list(lane_state.selected_items) for lane_state in lane_states}

    def _build_output_metadata(
        self,
        lane_states: List[ContextLaneRuntimeState],
        settings: ContextPlannerSettings,
    ) -> Dict[str, Any]:
        """
        Build free-form planner metadata for debugging, analysis, and dry runs.

        Recommended contents
        --------------------
        - total token budget
        - per-lane base/current/used token counts
        - redistribution history
        - per-lane first/last timestamps
        - optional score summaries
        - pass execution notes
        """
        return {
            "total_token_budget": settings.token_length,
            "lanes": {
                lane_state.name: {
                    "token_budget_base": lane_state.token_budget_base,
                    "token_budget_current": lane_state.token_budget_current,
                    "token_budget_used": lane_state.token_budget_used,
                    "token_budget_missing": lane_state.token_budget_missing,
                    "token_budget_surplus": lane_state.token_budget_surplus,
                    "selected_source_ids": list(lane_state.selected_source_ids),
                    "selected_scores": lane_state.metadata.get("selected_scores", []),
                    "candidate_count": lane_state.metadata.get("candidate_count", 0),
                    "first_timestamp_utc": lane_state.first_timestamp_utc,
                    "last_timestamp_utc": lane_state.last_timestamp_utc,
                    "last_pass_name": lane_state.metadata.get("last_pass_name"),
                }
                for lane_state in lane_states
            },
        }

    @staticmethod
    def _cosine_similarity(left: List[float], right: List[float]) -> float:
        return cosine_sim(left, right)

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        return get_token_count(text)

    def _estimate_knoxel_tokens(self, knoxel: KnoxelBase) -> int:
        try:
            return self._estimate_text_tokens(knoxel.get_story_element())
        except Exception:
            return self._estimate_text_tokens(knoxel.content)

    @staticmethod
    def _coerce_mental_vector(value: Union[FullMentalState, List[float], None]) -> List[float]:
        if value is None:
            return []
        if isinstance(value, FullMentalState):
            return value.to_list()
        if isinstance(value, list):
            return value
        return []

    @staticmethod
    def _get_knoxel_timestamp(knoxel: KnoxelBase) -> datetime.datetime:
        return knoxel.timestamp_world_begin

    def _get_output_item_timestamp(self, item: Union[KnoxelBase, VirtualKnoxel]) -> Optional[datetime.datetime]:
        if isinstance(item, VirtualKnoxel):
            return item.causal_timestamp_utc
        return self._get_knoxel_timestamp(item)

    def _select_with_type_distribution(
        self,
        selection_pool: List[Tuple[KnoxelBase, float, Dict[str, float]]],
        lane_state: ContextLaneRuntimeState,
    ) -> List[Tuple[KnoxelBase, float, Dict[str, float]]]:
        token_budget = lane_state.token_budget_current
        selected_entries: List[Tuple[KnoxelBase, float, Dict[str, float]]] = []
        selected_ids: set[int] = set()
        current_tokens = 0

        distribution = lane_state.resolved.sampler.sample_knoxel_type_distribution
        target_tokens = {self._type_spec_key(type_spec): token_budget * weight for weight, type_spec in distribution}
        current_type_tokens = {self._type_spec_key(type_spec): 0 for _, type_spec in distribution}

        def add_entry(entry: Tuple[KnoxelBase, float, Dict[str, float]]) -> bool:
            nonlocal current_tokens
            knoxel = entry[0]
            item_tokens = self._estimate_knoxel_tokens(knoxel)
            if knoxel.id in selected_ids:
                return False
            if selected_entries and current_tokens + item_tokens > token_budget:
                return False
            if not self._passes_type_ratio_constraints(selected_entries, entry, lane_state):
                return False
            selected_entries.append(entry)
            selected_ids.add(knoxel.id)
            current_tokens += item_tokens
            for _, type_spec in distribution:
                if self._matches_type_spec(knoxel, type_spec):
                    current_type_tokens[self._type_spec_key(type_spec)] += item_tokens
            return True

        for _, type_spec in distribution:
            key = self._type_spec_key(type_spec)
            for entry in selection_pool:
                if current_type_tokens[key] >= target_tokens[key]:
                    break
                if not self._matches_type_spec(entry[0], type_spec):
                    continue
                add_entry(entry)

        for missing_type, fallback_type in lane_state.resolved.sampler.sample_fill_shortfall_from_type_distribution:
            missing_key = self._type_spec_key(missing_type)
            while current_type_tokens.get(missing_key, 0) < target_tokens.get(missing_key, 0):
                changed = False
                for entry in selection_pool:
                    if not self._matches_type_spec(entry[0], fallback_type):
                        continue
                    if add_entry(entry):
                        current_type_tokens[missing_key] = current_type_tokens.get(missing_key, 0) + self._estimate_knoxel_tokens(entry[0])
                        changed = True
                        break
                if not changed:
                    break

        return selected_entries

    def _passes_type_ratio_constraints(
        self,
        selected_entries: List[Tuple[KnoxelBase, float, Dict[str, float]]],
        candidate_entry: Tuple[KnoxelBase, float, Dict[str, float]],
        lane_state: ContextLaneRuntimeState,
    ) -> bool:
        """
        # disabled cause it doesnt work!
        candidate = candidate_entry[0]
        for ratio, target_type, reference_types in lane_state.resolved.sampler.sample_max_type_to_reference_ratio:
            if not self._matches_type_spec(candidate, target_type):
                continue
            target_count = sum(1 for entry in selected_entries if self._matches_type_spec(entry[0], target_type)) + 1
            reference_count = sum(
                1
                for entry in selected_entries
                if any(self._matches_type_spec(entry[0], reference_type) for reference_type in reference_types)
            )
            if reference_count == 0:
                return False
            if target_count > ratio * reference_count:
                return False
        """
        return True

    def _resolve_selected_clusters_to_children(
        self,
        selected_entries: List[Tuple[KnoxelBase, float, Dict[str, float]]],
        scored_candidates: List[Tuple[KnoxelBase, float, Dict[str, float]]],
        lane_state: ContextLaneRuntimeState,
    ) -> List[Tuple[KnoxelBase, float, Dict[str, float]]]:
        selected_cluster_entries = [entry for entry in selected_entries if getattr(entry[0], "included_event_ids", None)]
        if not selected_cluster_entries:
            return selected_entries

        max_clusters_to_expand = max(
            1,
            int(round(len(selected_cluster_entries) * lane_state.resolved.sampler.sample_resolve_cluster_to_children_pct)),
        )
        scored_by_id = {entry[0].id: entry for entry in scored_candidates}
        resolved_entries: List[Tuple[KnoxelBase, float, Dict[str, float]]] = []

        expanded = 0
        for entry in selected_entries:
            knoxel = entry[0]
            if (
                expanded < max_clusters_to_expand
                and getattr(knoxel, "included_event_ids", None)
            ):
                child_ids = [
                    int(value.strip())
                    for value in knoxel.included_event_ids.split(",")
                    if value.strip()
                ]
                child_entries = []
                for child_id in child_ids:
                    child_entry = scored_by_id.get(child_id)
                    if child_entry is None:
                        continue
                    if lane_state.resolved.sampler.sample_resolve_cluster_children_types and not any(
                        self._matches_type_spec(child_entry[0], type_spec)
                        for type_spec in lane_state.resolved.sampler.sample_resolve_cluster_children_types
                    ):
                        continue
                    child_entries.append(child_entry)
                if child_entries:
                    resolved_entries.extend(child_entries)
                    expanded += 1
                    continue
            resolved_entries.append(entry)
        return resolved_entries

    def _clip_entries_to_budget(
        self,
        selected_entries: List[Tuple[KnoxelBase, float, Dict[str, float]]],
        token_budget: int,
    ) -> List[Tuple[KnoxelBase, float, Dict[str, float]]]:
        clipped_entries: List[Tuple[KnoxelBase, float, Dict[str, float]]] = []
        token_used = 0
        for entry in selected_entries:
            entry_tokens = self._estimate_knoxel_tokens(entry[0])
            if clipped_entries and token_used + entry_tokens > token_budget:
                continue
            clipped_entries.append(entry)
            token_used += entry_tokens
            if token_used >= token_budget:
                break
        return clipped_entries

    def _clip_output_items_to_budget(
        self,
        items: List[Union[KnoxelBase, VirtualKnoxel]],
        token_budget: int,
    ) -> List[Union[KnoxelBase, VirtualKnoxel]]:
        clipped_items: List[Union[KnoxelBase, VirtualKnoxel]] = []
        token_used = 0
        for item in items:
            item_tokens = self._estimate_text_tokens(item.content) if isinstance(item, VirtualKnoxel) else self._estimate_knoxel_tokens(item)
            if clipped_items and token_used + item_tokens > token_budget:
                continue
            clipped_items.append(item)
            token_used += item_tokens
            if token_used >= token_budget:
                break
        return clipped_items

    def _effective_lane_token_budget(
        self,
        lane_state: ContextLaneRuntimeState,
    ) -> int:
        return lane_state.token_budget_current + lane_state.resolved.sampler.sample_generation_budget_slack_tokens

    def _expand_selected_raw_to_topic_blocks(
        self,
        selected_items: List[KnoxelBase],
        lane_state: ContextLaneRuntimeState,
    ) -> List[KnoxelBase]:
        sampler = lane_state.resolved.sampler
        if not sampler.sample_expand_relevant_raw_to_topic_blocks:
            return selected_items

        selected_features = [item for item in selected_items if isinstance(item, Feature)]
        if not selected_features:
            return selected_items

        topical_clusters = [
            knoxel
            for knoxel in self._plan_knoxels_current
            if isinstance(knoxel, MemoryClusterKnoxel) and knoxel.cluster_type == ClusterType.Topical and knoxel.included_event_ids
        ]
        if not topical_clusters:
            return selected_items

        clusters_with_features: List[Tuple[MemoryClusterKnoxel, List[Feature]]] = []
        feature_to_cluster_ids: Dict[int, List[int]] = {}
        for cluster in sorted(topical_clusters, key=lambda item: (item.timestamp_world_begin, item.id)):
            event_ids = [int(value.strip()) for value in cluster.included_event_ids.split(",") if value.strip()]
            features = [
                feature
                for event_id in event_ids
                for feature in [self._plan_knoxels_current_by_id().get(event_id)]
                if isinstance(feature, Feature)
            ]
            if not features:
                continue
            clusters_with_features.append((cluster, features))
            for feature in features:
                feature_to_cluster_ids.setdefault(feature.id, []).append(cluster.id)

        if not clusters_with_features:
            return selected_items

        cluster_index_by_id = {cluster.id: index for index, (cluster, _) in enumerate(clusters_with_features)}
        cluster_by_id = {cluster.id: (cluster, features) for cluster, features in clusters_with_features}
        selected_cluster_ids: List[int] = []
        for feature in selected_features:
            cluster_ids = feature_to_cluster_ids.get(feature.id, [])
            if not cluster_ids:
                continue
            # Prefer the smallest containing topical cluster when multiple exist.
            best_cluster_id = min(cluster_ids, key=lambda cluster_id: len(cluster_by_id[cluster_id][1]))
            if best_cluster_id not in selected_cluster_ids:
                selected_cluster_ids.append(best_cluster_id)

        if not selected_cluster_ids:
            return selected_items

        expanded_feature_ids: set[int] = set()
        expanded_cluster_ids: set[int] = set()
        min_count = sampler.sample_min_raw_feature_block_count
        min_tokens = sampler.sample_min_raw_feature_block_tokens

        for cluster_id in selected_cluster_ids:
            candidate_cluster_ids = [cluster_id]
            current_features = list(cluster_by_id[cluster_id][1])
            left_index = cluster_index_by_id[cluster_id] - 1
            right_index = cluster_index_by_id[cluster_id] + 1

            while True:
                current_count = len(current_features)
                current_tokens = sum(self._estimate_knoxel_tokens(feature) for feature in current_features)
                if current_count >= min_count and current_tokens >= min_tokens:
                    break
                took_any = False
                if left_index >= 0:
                    left_cluster_id = clusters_with_features[left_index][0].id
                    candidate_cluster_ids.append(left_cluster_id)
                    current_features.extend(clusters_with_features[left_index][1])
                    left_index -= 1
                    took_any = True
                if current_count >= min_count and current_tokens >= min_tokens:
                    break
                if right_index < len(clusters_with_features):
                    right_cluster_id = clusters_with_features[right_index][0].id
                    candidate_cluster_ids.append(right_cluster_id)
                    current_features.extend(clusters_with_features[right_index][1])
                    right_index += 1
                    took_any = True
                if not took_any:
                    break

            expanded_cluster_ids.update(candidate_cluster_ids)
            expanded_feature_ids.update(feature.id for feature in current_features)

        non_feature_items = [item for item in selected_items if not isinstance(item, Feature)]
        kept_non_feature_items: List[KnoxelBase] = []
        for item in non_feature_items:
            if not isinstance(item, MemoryClusterKnoxel) or item.cluster_type != ClusterType.Temporal:
                kept_non_feature_items.append(item)
                continue
            start = item.timestamp_world_begin
            end = item.timestamp_world_end
            overlapping_selected_features = [
                feature
                for feature_id in expanded_feature_ids
                for feature in [self._plan_knoxels_current_by_id().get(feature_id)]
                if isinstance(feature, Feature) and start <= feature.timestamp_world_begin <= end
            ]
            if not overlapping_selected_features:
                kept_non_feature_items.append(item)
                continue
            all_features_in_span = [
                knoxel
                for knoxel in self._plan_knoxels_current
                if isinstance(knoxel, Feature) and knoxel.causal and start <= knoxel.timestamp_world_begin <= end
            ]
            if all_features_in_span and all(feature.id in expanded_feature_ids for feature in all_features_in_span):
                continue
            kept_non_feature_items.append(item)

        expanded_features = [
            feature
            for feature_id in expanded_feature_ids
            for feature in [self._plan_knoxels_current_by_id().get(feature_id)]
            if isinstance(feature, Feature)
        ]
        merged_items = [*kept_non_feature_items, *expanded_features]
        merged_items = sorted(merged_items, key=lambda item: (self._get_knoxel_timestamp(item), item.id))
        lane_state.metadata["expanded_topic_cluster_ids"] = sorted(expanded_cluster_ids)
        lane_state.metadata["expanded_raw_feature_count"] = len(expanded_features)
        return merged_items

    def _plan_knoxels_current_by_id(self) -> Dict[int, KnoxelBase]:
        return {knoxel.id: knoxel for knoxel in self._plan_knoxels_current}

    @staticmethod
    def _type_spec_key(type_spec: Union[Type[KnoxelBase], KnoxelType, KnoxelSubtypeBase]) -> str:
        if isinstance(type_spec, type):
            return type_spec.__name__
        return str(type_spec)

    def _knoxel_matches_prefilter(self, knoxel: KnoxelBase, prefilter: ContextPreFilterRules) -> bool:
        if prefilter.allowed_types and not any(self._matches_type_spec(knoxel, type_spec) for type_spec in prefilter.allowed_types):
            return False
        if prefilter.banned_types and any(self._matches_type_spec(knoxel, type_spec) for type_spec in prefilter.banned_types):
            return False
        if prefilter.temporal_scope_start and self._get_knoxel_timestamp(knoxel) < prefilter.temporal_scope_start:
            return False
        if prefilter.temporal_scope_end and self._get_knoxel_timestamp(knoxel) > prefilter.temporal_scope_end:
            return False
        if prefilter.causal_only is not None:
            is_causal = bool(getattr(knoxel, "causal", False))
            if prefilter.causal_only != is_causal:
                return False
        return True

    @staticmethod
    def _matches_type_spec(knoxel: KnoxelBase, type_spec: Union[Type[KnoxelBase], KnoxelType, KnoxelSubtypeBase]) -> bool:
        if isinstance(type_spec, type):
            return isinstance(knoxel, type_spec)
        if isinstance(type_spec, KnoxelType):
            return knoxel.type == type_spec
        if isinstance(type_spec, KnoxelSubtypeBase):
            for attr_name in ("feature_type", "stimulus_type", "action_type", "entity_class", "cluster_type", "narrative_type"):
                if getattr(knoxel, attr_name, None) == type_spec:
                    return True
        return False
