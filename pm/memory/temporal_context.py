"""Pick the memory records that show one unbroken time line inside a token budget.

Two record kinds compete for the same space. A raw feature is the original
event. A temporal summary is a consolidated block that covers a range of
events. Each feature in an exact cover comes from its own record or from a
summary whose recorded members account for the contiguous part it replaces.
Time alone cannot establish that two simultaneous exchanges belong together.

A dynamic-programming search solves this exactly. Each start index holds every
legal way to continue the cover from that point. The search walks the start
indices backwards, and keeps only the states that no other state beats at the
same or a lower token spend. A summary wins when it compresses a long span
that is not important enough to show in full.

The ``ratio`` argument sets the share of the budget that goes to raw features.
The reachable shares form a discrete set, so the solver runs the search over a
fixed sweep of feature-bonus values and keeps the closest one.

This module is pure. It reads no store, calls no model, and embeds no text.
The caller supplies the records, a vector lookup, and the query vector.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field
from numpy.typing import NDArray

import numpy as np

from pm.records.enums import ClusterType, FeatureType
from pm.records.memory_records import Feature, MemoryClusterRecord
from pm.memory.cluster_membership import ClusterMembership, is_episode_cluster

logger = logging.getLogger(__name__)

MIN_FEATURE_BONUS = -6.0
MAX_FEATURE_BONUS = 12.0
USE_NUMPY_BUDGET_SOLVER = True

# The sweep visits this many feature-bonus values. The step divisor is one
# less, so the last step index lands exactly on MAX_FEATURE_BONUS.
RATIO_SWEEP_STEPS = 97
RATIO_SWEEP_DIVISOR = 96.0

def _cosine_sim(left: list[float] | None, right: list[float] | None) -> float:
    """Return the dot product of two vectors, or 0.0 when one is unusable.

    Stored vectors are already normalized, so the dot product is the cosine.
    A record with no vector, and a pair with different lengths, both score 0.0.
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    return math.sumprod(left, right)


@dataclass(frozen=True)
class _Candidate:
    # Exact contiguous coverage span in feature-index space.
    start_index: int
    end_index: int
    # Token cost if this candidate is selected into the result.
    tokens: int
    # Raw-feature tokens contributed by this candidate. Summaries contribute 0 here.
    feature_tokens: int
    # Base utility independent of the target raw-detail ratio.
    utility_base: float
    # Utility multiplied by the ratio search parameter.
    utility_feature_bonus: float
    item: TemporalHistoryItem
    is_feature: bool


@dataclass
class _State:
    # DP state for one exact-cover suffix solution.
    utility: float
    feature_tokens: int
    total_tokens: int
    items: list[TemporalHistoryItem]


@dataclass(frozen=True)
class _BudgetStates:
    # Sparse states sorted by tokens above the minimum cover of this suffix.
    # Choice arrays point to a candidate and the next suffix's extra spend.
    tokens: NDArray[np.int64]
    utilities: NDArray[np.float64]
    feature_tokens: NDArray[np.int64]
    candidate_indices: NDArray[np.int64]
    tail_tokens: NDArray[np.int64]


def _feature_ratio(feature_tokens: int, total_tokens: int) -> float:
    """Return the raw-feature token share of one solution.

    A cover with no tokens has share 0.0. This cannot happen for a normal
    timeline, because every feature carries text. It keeps the ratio sweep from
    dividing by zero when the caller supplies features with empty text.
    """
    if total_tokens <= 0:
        return 0.0
    return feature_tokens / total_tokens


class TemporalHistoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    record_id: int
    started_at: datetime
    ended_at: datetime
    kind: Literal["feature", "summary"]
    level: int
    content: str
    source: str
    feature_type: FeatureType | None
    evidence_role: str = ""
    source_ids: tuple[int, ...] = ()
    covered_feature_ids: tuple[int, ...] = Field(default=(),
        description="Exact accounted feature IDs resolved from stored membership. Empty for unverified old captures.")
    relevance: float
    token_cost: int = Field(ge=0)

    def __setstate__(
        self, state: dict[str,  # pm-lint: allow PM004 -- Pydantic's external pickle-state mapping
                          Any],  # pm-lint: allow PM002 -- pickle state holds mixed Pydantic internals from older captures
    ) -> None:
        """Accept Pydantic's pickle wire format from captures predating explicit coverage."""
        state["__dict__"].setdefault("covered_feature_ids", ())
        super().__setstate__(state)


class TemporalHistoryInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    items: tuple[TemporalHistoryItem, ...]


@dataclass(frozen=True, slots=True)
class TemporalSelectionSettings:
    max_tokens: int
    raw_ratio: float = 0.4
    allow_coverage_omissions: bool = False
    require_temporal_history: bool = True
    require_contiguous_temporal_span: bool = True
    run_final_checks: bool = True


@dataclass(frozen=True, slots=True)
class TemporalCoverageSelection:
    items: tuple[TemporalHistoryItem, ...]
    total_tokens: int
    raw_feature_tokens: int
    minimum_cover_tokens: int
    covered_feature_ids: tuple[int, ...]
    omitted_feature_ids: tuple[int, ...]
    omitted_record_ids: tuple[int, ...] = ()


class RequiredTemporalCoverageTooLarge(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _TemporalGraph:
    features: list[TemporalHistoryItem]
    temporal_summaries: list[TemporalHistoryItem]
    feature_scores: list[float]
    feature_tokens: list[int]
    raw_total_tokens: int
    index_by_feature_id: dict[int, int]
    item_token_count: dict[int, int]
    candidates_by_start: dict[int, list[_Candidate]]
    candidates_by_id: dict[int, _Candidate]


@dataclass(frozen=True, slots=True, order=True)
class _FeatureHistoryOrder:
    started_at: datetime
    record_id: int
    item: TemporalHistoryItem = field(compare=False)


@dataclass(frozen=True, slots=True, order=True)
class _SummaryHistoryOrder:
    started_at: datetime
    ended_at: datetime
    level: int
    record_id: int
    item: TemporalHistoryItem = field(compare=False)


@dataclass(frozen=True, slots=True, order=True)
class _RankedState:
    utility: float
    feature_tokens: int
    negative_total_tokens: int
    state: _State = field(compare=False)


@dataclass(frozen=True, slots=True, order=True)
class _RankedStatePosition:
    utility: float
    feature_tokens: int
    negative_total_tokens: int
    index: int = field(compare=False)


def _build_temporal_graph(input: TemporalHistoryInput) -> _TemporalGraph:
    # Work in pure causal chronological order. The whole algorithm assumes that
    # "history" means an exact partition over this feature list.
    features = [entry.item for entry in sorted(_FeatureHistoryOrder(item.started_at, item.record_id, item)
        for item in input.items if item.kind == "feature")]
    temporal_summaries = [entry.item for entry in sorted(_SummaryHistoryOrder(item.started_at, item.ended_at,
        item.level, item.record_id, item) for item in input.items if item.kind == "summary")]
    # Relevance is measured once up front. The rest of the algorithm only consumes
    # these scalar scores instead of doing any more embedding math.
    feature_scores = [
        max(0.0, feature.relevance) ** 2 for feature in features
    ]
    feature_tokens = [feature.token_cost for feature in features]
    raw_total_tokens = sum(feature_tokens)

    index_by_feature_id = {feature.record_id: index for index, feature in enumerate(features)}
    item_token_count: dict[int, int] = {
        feature.record_id: feature_tokens[index] for index, feature in enumerate(features)
    }
    cumulative_feature_tokens = [0]
    cumulative_feature_scores = [0.0]
    for token_total, score in zip(feature_tokens, feature_scores):
        cumulative_feature_tokens.append(cumulative_feature_tokens[-1] + token_total)
        cumulative_feature_scores.append(cumulative_feature_scores[-1] + score)

    # At every start index we store all allowed ways to continue the exact temporal
    # cover from that point: either the single raw feature, or a summary spanning
    # multiple raw features.
    candidates_by_start: dict[int, list[_Candidate]] = {index: [] for index in range(len(features))}
    for index, feature in enumerate(features):
        score = feature_scores[index]
        tokens = feature_tokens[index]
        candidates_by_start[index].append(
            _Candidate(
                start_index=index,
                end_index=index,
                tokens=tokens,
                feature_tokens=tokens,
                utility_base=(tokens * score) + 0.25,
                utility_feature_bonus=tokens * score,
                item=feature,
                is_feature=True,
            )
        )

    for summary in temporal_summaries:
        # Work in this view's feature indices. A summary with interleaved
        # members remains valid for retrieval, but cannot replace a timeline
        # interval containing another event it never accounted for.
        positions = {index_by_feature_id[key] for key in summary.covered_feature_ids if key in index_by_feature_id}
        if not positions:
            continue
        start_index = min(positions)
        end_index = max(positions)
        span_length = end_index - start_index + 1
        if len(positions) != span_length:
            continue
        span_raw_tokens = cumulative_feature_tokens[end_index + 1] - cumulative_feature_tokens[start_index]
        span_mean_score = (cumulative_feature_scores[end_index + 1] - cumulative_feature_scores[start_index]) / span_length
        span_peak_score = max(feature_scores[start_index:end_index + 1])
        level_weight = min(float(summary.level), 6.0) / 6.0
        # The enclosing Story prompt may already contain this exact record.
        summary_tokens = summary.token_cost
        item_token_count[summary.record_id] = summary_tokens
        candidates_by_start[start_index].append(
            _Candidate(
                start_index=start_index,
                end_index=end_index,
                tokens=summary_tokens,
                feature_tokens=0,
                utility_base=(
                    # Summaries become attractive when they compress a long span
                    # that is not important enough to justify full raw expansion.
                    (span_raw_tokens * (0.15 * level_weight * ((0.55 * span_mean_score) + (0.45 * span_peak_score))))
                    + (0.04 * span_length * level_weight)
                    - (0.001 * summary_tokens)
                ),
                utility_feature_bonus=0.0,
                item=summary,
                is_feature=False,
            )
        )

    candidates_by_id: dict[int, _Candidate] = {}
    for candidates in candidates_by_start.values():
        for candidate in candidates:
            candidates_by_id[candidate.item.record_id] = candidate
    return _TemporalGraph(features, temporal_summaries, feature_scores, feature_tokens,
        raw_total_tokens, index_by_feature_id, item_token_count, candidates_by_start, candidates_by_id)


@dataclass(frozen=True, slots=True, order=True)
class OmissionPriority:
    covered_weight_per_token: float
    relevance: float
    position: int
    record_id: int


class TemporalCoverageProblem:
    def __init__(self, input: TemporalHistoryInput) -> None:
        ids = [item.record_id for item in input.items]
        if len(ids) != len(set(ids)):
            raise ValueError("Temporal history requires unique source identities.")
        self.input = input
        self.graph = _build_temporal_graph(input)
        self.minimum_suffix_tokens = [0] * (len(self.graph.features) + 1)
        self.minimum_state = self.compute_min_token_cover()

        priorities: list[OmissionPriority] = []
        for position, item in enumerate(self.minimum_state.items):
            candidate = self.graph.candidates_by_id[item.record_id]
            scores = self.graph.feature_scores[candidate.start_index:candidate.end_index + 1]
            relevance = 0.55 * sum(scores) / len(scores) + 0.45 * max(scores)
            # A cheap calendar summary can retain hundreds of originals. Give
            # each covered event a vote, then weight its span by focus relevance.
            weight_per_token = len(scores) * (1 + relevance) / max(1, candidate.tokens)
            priorities.append(OmissionPriority(weight_per_token, relevance, position, item.record_id))
        self.omission_order = tuple(sorted(priorities))

    def minimum_cover(self) -> TemporalCoverageSelection:
        if self.graph.features:
            _validate_temporal_partition(self.minimum_state.items, self.graph.features, self.graph.candidates_by_start,
                require_temporal_history=True, require_contiguous_temporal_span=True)
        return self.selection(tuple(self.minimum_state.items))

    def selection(self, items: tuple[TemporalHistoryItem, ...],
                  omitted_record_ids: tuple[int, ...] = ()) -> TemporalCoverageSelection:
        covered: set[int] = set()
        raw_tokens = 0
        total_tokens = 0
        for item in items:
            candidate = self.graph.candidates_by_id[item.record_id]
            total_tokens += candidate.tokens
            raw_tokens += candidate.feature_tokens
            for feature in self.graph.features[candidate.start_index:candidate.end_index + 1]:
                covered.add(feature.record_id)
        return TemporalCoverageSelection(items=items, total_tokens=total_tokens,
            raw_feature_tokens=raw_tokens, minimum_cover_tokens=self.minimum_state.total_tokens,
            covered_feature_ids=tuple(item.record_id for item in self.graph.features if item.record_id in covered),
            omitted_feature_ids=tuple(item.record_id for item in self.graph.features if item.record_id not in covered),
            omitted_record_ids=omitted_record_ids)

    def select(self, settings: TemporalSelectionSettings) -> TemporalCoverageSelection:
        if settings.max_tokens < 0:
            raise ValueError("A temporal renderer must supply its actual nonnegative remaining token budget.")
        if not 0 <= settings.raw_ratio <= 1:
            raise ValueError("The raw temporal detail ratio must lie between zero and one.")
        if self.minimum_state.total_tokens > settings.max_tokens:
            if not settings.allow_coverage_omissions:
                raise RequiredTemporalCoverageTooLarge(
                    f"Temporal history cheapest complete cover requires {self.minimum_state.total_tokens} tokens; "
                    f"actual available budget is {settings.max_tokens}.")
            return self.omit_least_relevant_units(settings.max_tokens)
        if not self.graph.features:
            return self.minimum_cover()
        items = _TemporalRefinement(self, settings).refine()
        return self.selection(tuple(items))

    def include_visible_records(self, selected: TemporalCoverageSelection,
                                visible_ids: set[int]) -> TemporalCoverageSelection:
        """Credit already-rendered evidence even when its coarser DP unit was omitted."""
        missing = set(selected.omitted_feature_ids)
        if not missing:
            return selected
        for record_id in visible_ids:
            candidate = self.graph.candidates_by_id.get(record_id)
            if candidate is None:
                continue  # A sparse recollection supplies recall, not a contiguous interval.
            if candidate.is_feature:
                missing.discard(record_id)
            else:
                missing.difference_update(candidate.item.covered_feature_ids)
        return replace(selected,
            covered_feature_ids=tuple(item.record_id for item in self.graph.features if item.record_id not in missing),
            omitted_feature_ids=tuple(item.record_id for item in self.graph.features if item.record_id in missing),
            omitted_record_ids=tuple(key for key in selected.omitted_record_ids if key not in visible_ids))

    def omit_least_relevant_units(self, max_tokens: int) -> TemporalCoverageSelection:
        if max_tokens < 0 or max_tokens >= self.minimum_state.total_tokens:
            raise ValueError("Temporal omissions require a nonnegative budget below the cheapest complete cover.")
        total_tokens = self.minimum_state.total_tokens
        omitted: list[int] = []
        for priority in self.omission_order:
            if total_tokens <= max_tokens:
                break
            if self.graph.candidates_by_id[priority.record_id].tokens == 0:
                continue
            omitted.append(priority.record_id)
            total_tokens -= self.graph.candidates_by_id[priority.record_id].tokens
        omitted_ids = set(omitted)
        kept = tuple(item for item in self.minimum_state.items if item.record_id not in omitted_ids)
        return self.selection(kept, tuple(omitted))

    def compute_min_token_cover(self) -> _State:
        features = self.graph.features
        candidates_by_start = self.graph.candidates_by_start
        # Diagnostic-only DP: ignore utility entirely and compute the cheapest exact
        # gapless cover that exists in the current candidate graph.
        best_tokens = self.minimum_suffix_tokens
        # Store one next step per position. Copying each suffix's entire list
        # requires quadratic memory when most of an older home is still raw.
        next_choice: list[_Candidate | None] = [None] * len(features)
        for start_index in range(len(features) - 1, -1, -1):
            chosen: _Candidate | None = None
            for candidate in candidates_by_start[start_index]:
                next_index = candidate.end_index + 1
                total_tokens = candidate.tokens + best_tokens[next_index]
                if chosen is None or total_tokens < best_tokens[start_index]:
                    chosen = candidate
                    best_tokens[start_index] = total_tokens
            if chosen is None:
                raise ValueError(f"Temporal graph has no cover starting at feature {features[start_index].record_id}.")
            next_choice[start_index] = chosen

        items: list[TemporalHistoryItem] = []
        raw_tokens = 0
        position = 0
        while position < len(features):
            step = next_choice[position]
            assert step is not None  # Every start index received a candidate above.
            items.append(step.item)
            raw_tokens += step.feature_tokens
            position = step.end_index + 1
        return _State(utility=0.0, feature_tokens=raw_tokens, total_tokens=best_tokens[0], items=items)



class _TemporalRefinement:
    def __init__(self, problem: TemporalCoverageProblem, settings: TemporalSelectionSettings) -> None:
        self.problem = problem
        self.graph = problem.graph
        self.features = self.graph.features
        self.temporal_summaries = self.graph.temporal_summaries
        self.feature_tokens = self.graph.feature_tokens
        self.raw_total_tokens = self.graph.raw_total_tokens
        self.item_token_count = self.graph.item_token_count
        self.candidates_by_start = self.graph.candidates_by_start
        self.ratio = settings.raw_ratio
        self.max_tokens = settings.max_tokens
        self.require_temporal_history = settings.require_temporal_history
        self.require_contiguous_temporal_span = settings.require_contiguous_temporal_span
        self.run_final_checks = settings.run_final_checks
        self.solve_target_cache: dict[float, _State] = {}

    def prune_states(self, states: dict[int, _State]) -> dict[int, _State]:
        # For the same or greater token spend, a lower-utility suffix can never win.
        # This cheap skyline pruning keeps the DP from exploding completely.
        ordered = sorted(states.items())
        pruned: dict[int, _State] = {}
        best_utility = float("-inf")
        for token_total, state in ordered:
            if state.utility > best_utility:
                pruned[token_total] = state
                best_utility = state.utility
        return pruned

    def log_budget_failure_diagnostics(self, token_cap: int) -> None:
        min_cover_state = self.problem.minimum_state
        feature_only_tokens = self.raw_total_tokens
        summary_counts_by_level: dict[int, int] = {}
        summary_tokens_by_level: dict[int, int] = {}
        for summary in self.temporal_summaries:
            summary_counts_by_level[summary.level] = summary_counts_by_level.get(summary.level, 0) + 1
            # A summary that covers no feature never becomes a candidate, so it has
            # no entry here. Count it as 0 tokens instead of losing the whole report.
            summary_tokens = self.item_token_count.get(summary.record_id, 0)
            summary_tokens_by_level[summary.level] = summary_tokens_by_level.get(summary.level, 0) + summary_tokens

        candidate_count_by_start = [len(self.candidates_by_start[index]) for index in range(len(self.features))]
        starts_with_only_raw = [
            index
            for index, count in enumerate(candidate_count_by_start)
            if count == 1 and self.candidates_by_start[index][0].is_feature
        ]
        chosen_item_descriptions: list[str] = []
        for item in min_cover_state.items[:40]:
            if item.kind == "feature":
                chosen_item_descriptions.append(
                    f"feature id={item.record_id} ts={item.started_at.isoformat()} tokens={self.item_token_count[item.record_id]}"
                )
            else:
                chosen_item_descriptions.append(
                    f"summary id={item.record_id} level={item.level} ts={item.started_at.isoformat()}..{item.ended_at.isoformat()} tokens={self.item_token_count[item.record_id]}"
                )

        logger.error(
            "\n".join(
                [
                    "temporal_context budget failure diagnostics",
                    f"max_tokens={token_cap}",
                    f"requested_ratio={self.ratio}",
                    f"require_temporal_history={self.require_temporal_history}",
                    f"require_contiguous_temporal_span={self.require_contiguous_temporal_span}",
                    f"feature_count={len(self.features)} temporal_summary_count={len(self.temporal_summaries)}",
                    f"feature_time_start={self.features[0].started_at.isoformat()}",
                    f"feature_time_end={self.features[-1].ended_at.isoformat()}",
                    f"feature_only_tokens={feature_only_tokens}",
                    f"min_exact_cover_tokens={min_cover_state.total_tokens}",
                    f"min_exact_cover_feature_tokens={min_cover_state.feature_tokens}",
                    f"min_exact_cover_item_count={len(min_cover_state.items)}",
                    f"candidate_count_min={min(candidate_count_by_start)}",
                    f"candidate_count_max={max(candidate_count_by_start)}",
                    f"candidate_count_avg={round(sum(candidate_count_by_start) / len(candidate_count_by_start), 3)}",
                    f"raw_only_start_count={len(starts_with_only_raw)}",
                    f"raw_only_start_sample={starts_with_only_raw[:20]}",
                    f"summary_counts_by_level={summary_counts_by_level}",
                    f"summary_tokens_by_level={summary_tokens_by_level}",
                    "min_exact_cover_prefix=" + " | ".join(chosen_item_descriptions),
                ]
            )
        )

    def solve_with_budget_legacy(self, feature_bonus: float, token_cap: int) -> _State:
        # Exact-cover DP with a token budget. The suffix state at position i contains
        # every non-dominated way to cover features[i:].
        best_from_index: list[dict[int, _State]] = [{} for _ in range(len(self.features) + 1)]
        best_from_index[len(self.features)] = {0: _State(utility=0.0, feature_tokens=0, total_tokens=0, items=[])}

        for start_index in range(len(self.features) - 1, -1, -1):
            current_states: dict[int, _State] = {}
            for candidate in self.candidates_by_start[start_index]:
                next_index = candidate.end_index + 1
                for tail_tokens, tail_state in best_from_index[next_index].items():
                    # Every candidate must join with a suffix that starts exactly
                    # after its covered span. That is what guarantees gaplessness.
                    total_tokens = candidate.tokens + tail_tokens
                    if total_tokens > token_cap:
                        continue
                    utility = candidate.utility_base + (candidate.utility_feature_bonus * feature_bonus) + tail_state.utility
                    feature_token_total = candidate.feature_tokens + tail_state.feature_tokens
                    existing = current_states.get(total_tokens)
                    if existing is not None and existing.utility >= utility:
                        continue
                    current_states[total_tokens] = _State(
                        utility=utility,
                        feature_tokens=feature_token_total,
                        total_tokens=total_tokens,
                        items=[candidate.item, *tail_state.items],
                    )
            best_from_index[start_index] = self.prune_states(current_states)

        if not best_from_index[0]:
            self.log_budget_failure_diagnostics(token_cap)
            raise Exception("temporal_context: no exact history cover fits inside max_tokens")
        return max(_RankedState(state.utility, state.feature_tokens, -state.total_tokens, state)
            for state in best_from_index[0].values()).state

    def solve_with_budget(self, feature_bonus: float, token_cap: int) -> _State:
        # NumPy-backed exact-cover DP. This keeps the same skyline state model as
        # solve_with_budget_legacy, but vectorizes each candidate x suffix join and
        # stores backpointers instead of copying item lists through every state.
        # Each suffix must pay for its cheapest complete cover. Index its states
        # by additional spend so a large mandatory history never inflates the
        # search table. The per-step differences telescope to the exact cost.
        minimum = self.problem.minimum_suffix_tokens
        extra_budget = token_cap - minimum[0]
        if extra_budget < 0:
            raise RequiredTemporalCoverageTooLarge(f"The minimum cover costs {minimum[0]} tokens; budget is {token_cap}.")
        empty_int = np.array([], dtype=np.int64)
        empty_float = np.array([], dtype=np.float64)
        best_from_index: list[_BudgetStates] = [
            _BudgetStates(empty_int, empty_float, empty_int, empty_int, empty_int)
            for _ in range(len(self.features) + 1)
        ]
        best_from_index[len(self.features)] = _BudgetStates(
            tokens=np.array([0], dtype=np.int64),
            utilities=np.array([0.0], dtype=np.float64),
            feature_tokens=np.array([0], dtype=np.int64),
            candidate_indices=np.array([-1], dtype=np.int64),
            tail_tokens=np.array([-1], dtype=np.int64),
        )

        for start_index in range(len(self.features) - 1, -1, -1):
            current_utilities = np.full(extra_budget + 1, -np.inf, dtype=np.float64)
            current_feature_tokens = np.zeros(extra_budget + 1, dtype=np.int64)
            current_candidate_indices = np.full(extra_budget + 1, -1, dtype=np.int64)
            current_tail_tokens = np.full(extra_budget + 1, -1, dtype=np.int64)

            for candidate_index, candidate in enumerate(self.candidates_by_start[start_index]):
                tail_states = best_from_index[candidate.end_index + 1]
                if tail_states.tokens.size == 0:
                    continue

                choice_cost = candidate.tokens + minimum[candidate.end_index + 1] - minimum[start_index]
                extra_tokens = tail_states.tokens + choice_cost
                valid_mask = extra_tokens <= extra_budget
                if not np.any(valid_mask):
                    continue

                valid_total_tokens = extra_tokens[valid_mask]
                utilities = (
                    tail_states.utilities[valid_mask]
                    + candidate.utility_base
                    + (candidate.utility_feature_bonus * feature_bonus)
                )
                improved_mask = utilities > current_utilities[valid_total_tokens]
                if not np.any(improved_mask):
                    continue

                improved_tokens = valid_total_tokens[improved_mask]
                current_utilities[improved_tokens] = utilities[improved_mask]
                current_feature_tokens[improved_tokens] = (
                    tail_states.feature_tokens[valid_mask][improved_mask] + candidate.feature_tokens
                )
                current_candidate_indices[improved_tokens] = candidate_index
                current_tail_tokens[improved_tokens] = tail_states.tokens[valid_mask][improved_mask]

            occupied_tokens = np.flatnonzero(np.isfinite(current_utilities)).astype(np.int64)
            if occupied_tokens.size == 0:
                best_from_index[start_index] = _BudgetStates(empty_int, empty_float, empty_int, empty_int, empty_int)
                continue

            occupied_utilities = current_utilities[occupied_tokens]
            previous_best = np.empty_like(occupied_utilities)
            previous_best[0] = -np.inf
            if occupied_utilities.size > 1:
                previous_best[1:] = np.maximum.accumulate(occupied_utilities[:-1])
            keep_mask = occupied_utilities > previous_best
            kept_tokens = occupied_tokens[keep_mask]
            best_from_index[start_index] = _BudgetStates(
                tokens=kept_tokens,
                utilities=occupied_utilities[keep_mask],
                feature_tokens=current_feature_tokens[kept_tokens],
                candidate_indices=current_candidate_indices[kept_tokens],
                tail_tokens=current_tail_tokens[kept_tokens],
            )

        root_states = best_from_index[0]
        if root_states.tokens.size == 0:
            self.log_budget_failure_diagnostics(token_cap)
            raise Exception("temporal_context: no exact history cover fits inside max_tokens")

        best_position = max(_RankedStatePosition(float(root_states.utilities[index]),
            int(root_states.feature_tokens[index]), -int(root_states.tokens[index]), index)
            for index in range(root_states.tokens.size)).index
        items: list[TemporalHistoryItem] = []
        start_index = 0
        remaining_extra = int(root_states.tokens[best_position])
        utility = float(root_states.utilities[best_position])
        feature_token_total = int(root_states.feature_tokens[best_position])

        while start_index < len(self.features):
            states = best_from_index[start_index]
            state_position = int(np.searchsorted(states.tokens, remaining_extra))
            candidate = self.candidates_by_start[start_index][int(states.candidate_indices[state_position])]
            items.append(candidate.item)
            remaining_extra = int(states.tail_tokens[state_position])
            start_index = candidate.end_index + 1

        return _State(
            utility=utility,
            feature_tokens=feature_token_total,
            total_tokens=minimum[0] + int(root_states.tokens[best_position]),
            items=items,
        )

    def solve_without_budget(self, feature_bonus: float) -> _State:
        # Same exact-cover DP, but now only utility matters because there is no
        # token cap. This naturally tends toward raw expansion in relevant spans.
        best_utility = [0.0] * (len(self.features) + 1)
        best_items: list[list[TemporalHistoryItem]] = [[] for _ in range(len(self.features) + 1)]
        best_feature_tokens = [0] * (len(self.features) + 1)
        best_total_tokens = [0] * (len(self.features) + 1)

        for start_index in range(len(self.features) - 1, -1, -1):
            chosen_state = _State(utility=float("-inf"), feature_tokens=0, total_tokens=0, items=[])
            for candidate in self.candidates_by_start[start_index]:
                next_index = candidate.end_index + 1
                utility = candidate.utility_base + (candidate.utility_feature_bonus * feature_bonus) + best_utility[next_index]
                feature_token_total = candidate.feature_tokens + best_feature_tokens[next_index]
                total_tokens = candidate.tokens + best_total_tokens[next_index]
                if utility <= chosen_state.utility:
                    continue
                chosen_state = _State(
                    utility=utility,
                    feature_tokens=feature_token_total,
                    total_tokens=total_tokens,
                    items=[candidate.item, *best_items[next_index]],
                )
            best_utility[start_index] = chosen_state.utility
            best_items[start_index] = chosen_state.items
            best_feature_tokens[start_index] = chosen_state.feature_tokens
            best_total_tokens[start_index] = chosen_state.total_tokens

        return _State(
            utility=best_utility[0],
            feature_tokens=best_feature_tokens[0],
            total_tokens=best_total_tokens[0],
            items=best_items[0],
        )

    def solve_target(self, feature_bonus: float) -> _State:
        # The ratio search and the final checks ask for the same feature_bonus values
        # multiple times, so memoizing here saves a lot of repeated DP work.
        if feature_bonus in self.solve_target_cache:
            return self.solve_target_cache[feature_bonus]
        if self.max_tokens > 0:
            if USE_NUMPY_BUDGET_SOLVER:
                state = self.solve_with_budget(feature_bonus=feature_bonus, token_cap=int(self.max_tokens))
            else:
                state = self.solve_with_budget_legacy(feature_bonus=feature_bonus, token_cap=int(self.max_tokens))
        else:
            state = self.solve_without_budget(feature_bonus=feature_bonus)
        self.solve_target_cache[feature_bonus] = state
        return state

    def refine(self) -> list[TemporalHistoryItem]:

        if self.max_tokens > 0 and self.raw_total_tokens <= int(self.max_tokens):
            # If the full raw history already fits, compressed summaries are pure
            # redundancy. The intended behavior is to return the raw timeline only.
            result = list(self.features)
            if self.run_final_checks:
                _validate_temporal_partition(result, self.features, self.candidates_by_start,
                    require_temporal_history=self.require_temporal_history,
                    require_contiguous_temporal_span=self.require_contiguous_temporal_span)
            return result

        # Sweep a small fixed range of ratio-control values and keep the solution whose
        # raw-token share lands closest to the requested ratio.
        best_feature_bonus = MIN_FEATURE_BONUS
        best_state = self.solve_target(feature_bonus=best_feature_bonus)
        best_gap = abs(_feature_ratio(best_state.feature_tokens, best_state.total_tokens) - self.ratio)
        for step_index in range(RATIO_SWEEP_STEPS):
            current_feature_bonus = MIN_FEATURE_BONUS + ((MAX_FEATURE_BONUS - MIN_FEATURE_BONUS) * step_index / RATIO_SWEEP_DIVISOR)
            state = self.solve_target(feature_bonus=current_feature_bonus)
            actual_ratio = _feature_ratio(state.feature_tokens, state.total_tokens)
            gap = abs(actual_ratio - self.ratio)
            if gap < best_gap:
                best_gap = gap
                best_state = state
                best_feature_bonus = current_feature_bonus

        if self.run_final_checks:
            _validate_projection_context(
                result=best_state.items,
                features=self.features,
                ratio=self.ratio,
                max_tokens=self.max_tokens,
                feature_tokens=self.feature_tokens,
                item_token_count=self.item_token_count,
                candidates_by_start=self.candidates_by_start,
                solved_states=self.solve_target_cache,
                feature_bonus=best_feature_bonus,
                require_temporal_history=self.require_temporal_history,
                require_contiguous_temporal_span=self.require_contiguous_temporal_span,
            )
        return best_state.items


def _validate_temporal_partition(result: list[TemporalHistoryItem], features: list[TemporalHistoryItem],
                                 candidates_by_start: dict[int, list[_Candidate]], *,
                                 require_temporal_history: bool,
                                 require_contiguous_temporal_span: bool) -> None:
    if not result:
        raise ValueError("Empty temporal cover for a nonempty timeline.")

    spans_by_id: dict[int, _Candidate] = {}
    for candidates in candidates_by_start.values():
        for candidate in candidates:
            spans_by_id[candidate.item.record_id] = candidate
    next_feature = 0
    for item in result:
        span = spans_by_id[item.record_id]
        if span.start_index != next_feature:
            raise ValueError(f"Temporal record {item.record_id} does not continue at feature index {next_feature}.")
        next_feature = span.end_index + 1
        if item.kind == "feature":
            if features[span.start_index].record_id != item.record_id or span.end_index != span.start_index:
                raise ValueError(f"Raw record {item.record_id} must cover exactly itself.")
        elif require_temporal_history:
            members = set(item.covered_feature_ids)
            for source in features[span.start_index:next_feature]:
                if source.record_id not in members:
                    raise ValueError(f"Temporal record {item.record_id} cannot cover unrelated feature {source.record_id}.")
    if next_feature != len(features):
        raise ValueError(f"Temporal cover ends at feature index {next_feature} of {len(features)}.")
    if require_contiguous_temporal_span:
        # World-time ranges can overlap. Source order breaks ties between
        # simultaneous events, including two people talking in one tick.
        for earlier, later in zip(features, features[1:]):
            if (earlier.started_at, earlier.record_id) > (later.started_at, later.record_id):
                raise ValueError(f"Temporal features {earlier.record_id}, {later.record_id} are out of order.")




def _validate_projection_context(
    result: list[TemporalHistoryItem],
    features: list[TemporalHistoryItem],
    ratio: float,
    max_tokens: int,
    feature_tokens: list[int],
    item_token_count: dict[int, int],
    candidates_by_start: dict[int, list[_Candidate]],
    solved_states: dict[float, _State],
    feature_bonus: float,
    *,
    require_temporal_history: bool = True,
    require_contiguous_temporal_span: bool = True,
) -> None:
    """Make sure that one result is a valid, optimal cover of the timeline.

    These checks are intentionally strict and crash fast. This function is meant
    for inspecting optimizer behavior, not for graceful runtime recovery. It
    changes nothing, except that it re-runs the solver through ``solve_target``.

    The covered features are the eligible records in this prompt's frozen view.
    """
    _validate_temporal_partition(result, features, candidates_by_start,
        require_temporal_history=require_temporal_history,
        require_contiguous_temporal_span=require_contiguous_temporal_span)
    index_by_feature_id = {feature.record_id: index for index, feature in enumerate(features)}
    total_tokens = sum(item_token_count[item.record_id] for item in result)
    raw_token_total = sum(feature_tokens[index_by_feature_id[item.record_id]] for item in result if item.kind == "feature")
    actual_ratio = _feature_ratio(raw_token_total, total_tokens)
    all_raw_selected = len(result) == len(features) and all(item.kind == "feature" for item in result)

    if max_tokens > 0 and total_tokens > int(max_tokens):
        raise Exception("temporal_context: result exceeds max_tokens")

    if not all_raw_selected:
        # The ratio is not continuous because the candidate space is discrete. So we
        # compare against the best ratio actually reachable by this optimizer sweep.
        best_ratio_gap = abs(actual_ratio - ratio)
        for step_index in range(RATIO_SWEEP_STEPS):
            current_feature_bonus = MIN_FEATURE_BONUS + ((MAX_FEATURE_BONUS - MIN_FEATURE_BONUS) * step_index / RATIO_SWEEP_DIVISOR)
            current_state = solved_states[current_feature_bonus]
            current_ratio = _feature_ratio(current_state.feature_tokens, current_state.total_tokens)
            best_ratio_gap = min(best_ratio_gap, abs(current_ratio - ratio))
        if abs(actual_ratio - ratio) > best_ratio_gap + 0.01:
            raise Exception("temporal_context: ratio is achievable more closely than the returned output")

    # Re-solve for the selected feature_bonus and make sure we did not somehow
    # mutate or return a non-optimal path afterwards.
    optimal_state = solved_states[feature_bonus]
    optimal_ids = [item.record_id for item in optimal_state.items]
    result_ids = [item.record_id for item in result]
    if result_ids != optimal_ids:
        raise Exception("temporal_context: result is not the actual optimizer output")



def build_temporal_context(features: list[Feature], temporal_summaries: list[MemoryClusterRecord],
                           embedding: list[float] | None, ratio: float, max_tokens: int, *,
                           vectors: dict[int, list[float] | None], token_costs: dict[int, int],
                           membership: ClusterMembership | None = None,
                           relevance_scores: Mapping[int, float] | None = None,  # pm-lint: allow PM004 -- planner relevance by source ID
                           require_temporal_history: bool = True, require_contiguous_temporal_span: bool = True,
                           run_final_checks: bool = True, allow_coverage_omissions: bool = False) -> list[Feature | MemoryClusterRecord]:
    """Select an exact or explicitly partial cover over these eligible features.

    A context-plan lane supplies membership for its whole scoped snapshot so a
    temporal parent can resolve children which belong to another lane.
    """
    items: list[TemporalHistoryItem] = []
    records: dict[int, Feature | MemoryClusterRecord] = {}
    for feature in features:
        if not feature.causal:
            continue
        records[feature.id] = feature
        relevance = (relevance_scores[feature.id] if relevance_scores is not None
                     else _cosine_sim(vectors[feature.id], embedding))
        items.append(TemporalHistoryItem(record_id=feature.id, started_at=feature.timestamp_world_begin,
            ended_at=feature.timestamp_world_end, kind="feature", level=0, content=feature.content,
            source=feature.source, feature_type=feature.feature_type,
            relevance=relevance, token_cost=token_costs[feature.id]))
    records.update((cluster.id, cluster) for cluster in temporal_summaries)
    membership = membership or ClusterMembership(records)  # pm-lint: allow PM005 -- visible knoxels indexed by ID
    for summary in temporal_summaries:
        if summary.cluster_type != ClusterType.Temporal and not is_episode_cluster(summary):
            continue
        items.append(TemporalHistoryItem(record_id=summary.id, started_at=summary.timestamp_world_begin,
            ended_at=summary.timestamp_world_end, kind="summary", level=summary.level, content=summary.content,
            source="", feature_type=None, covered_feature_ids=tuple(sorted(membership.source_ids(summary))),
            relevance=0.0, token_cost=token_costs[summary.id]))
    problem = TemporalCoverageProblem(TemporalHistoryInput(items=tuple(items)))
    settings = TemporalSelectionSettings(max_tokens=max(0, max_tokens), raw_ratio=ratio, allow_coverage_omissions=allow_coverage_omissions,
        require_temporal_history=require_temporal_history,
        require_contiguous_temporal_span=require_contiguous_temporal_span, run_final_checks=run_final_checks)
    if max_tokens <= 0 and problem.graph.features and not allow_coverage_omissions:
        selected_items = _TemporalRefinement(problem, settings).refine()
    else:
        selected_items = list(problem.select(settings).items)
    return [records[item.record_id] for item in selected_items]


__all__ = ["build_temporal_context", "TemporalHistoryItem", "TemporalHistoryInput",
    "TemporalCoverageProblem", "TemporalCoverageSelection", "TemporalSelectionSettings",
    "RequiredTemporalCoverageTooLarge", "MIN_FEATURE_BONUS", "MAX_FEATURE_BONUS",
    "RATIO_SWEEP_STEPS", "RATIO_SWEEP_DIVISOR", "USE_NUMPY_BUDGET_SOLVER"]
