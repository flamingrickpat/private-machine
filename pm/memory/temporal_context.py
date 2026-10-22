"""Pick the memory records that show one unbroken time line inside a token budget.

Two record kinds compete for the same space. A raw feature is the original
event. A temporal summary is a consolidated block that covers a range of
events. The result must cover every causal feature, from the first one to the
last one. Each covered feature comes either from its own raw record, or from
one summary whose time range contains it. The result can have no gap, and it
cannot mix a summary with a raw record that the summary already covers.

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
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Callable

import numpy as np
import tiktoken

from pm.memory.retrieval import record_text
from pm.records.enums import ClusterType, FeatureType
from pm.records.memory_records import Feature, MemoryClusterRecord

logger = logging.getLogger(__name__)

MIN_FEATURE_BONUS = -6.0
MAX_FEATURE_BONUS = 12.0
USE_NUMPY_BUDGET_SOLVER = True

# The sweep visits this many feature-bonus values. The step divisor is one
# less, so the last step index lands exactly on MAX_FEATURE_BONUS.
RATIO_SWEEP_STEPS = 97
RATIO_SWEEP_DIVISOR = 96.0

_ENCODING = tiktoken.get_encoding("o200k_base")


def _token_count(text: str) -> int:
    """Return the tiktoken estimate for one prompt text.

    The count is an estimate for a budget, not the count a serving endpoint
    reports. ``disallowed_special=()`` keeps a special-token marker in record
    content from raising an error.
    """
    from pm.llm.token_estimate import text_tokens
    return text_tokens(text)


def _cosine_sim(left: list[float] | None, right: list[float]) -> float:
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
    item: Feature | MemoryClusterRecord
    is_feature: bool


@dataclass
class _State:
    # DP state for one exact-cover suffix solution.
    utility: float
    feature_tokens: int
    total_tokens: int
    items: list[Feature | MemoryClusterRecord]


@dataclass(frozen=True)
class _BudgetStates:
    # Sparse, token-sorted states for one suffix index. Choice arrays are
    # backpointers into candidates_by_start[index] and the next suffix token.
    tokens: np.ndarray
    utilities: np.ndarray
    feature_tokens: np.ndarray
    candidate_indices: np.ndarray
    tail_tokens: np.ndarray


def _feature_ratio(feature_tokens: int, total_tokens: int) -> float:
    """Return the raw-feature token share of one solution.

    A cover with no tokens has share 0.0. This cannot happen for a normal
    timeline, because every feature carries text. It keeps the ratio sweep from
    dividing by zero when the caller supplies features with empty text.
    """
    if total_tokens <= 0:
        return 0.0
    return feature_tokens / total_tokens


def build_temporal_context(
    features: list[Feature],
    temporal_summaries: list[MemoryClusterRecord],
    embedding: list[float],
    ratio: float,
    max_tokens: int,
    *,
    vector_of: Callable[[int], list[float] | None],
    require_temporal_history: bool = True,
    require_contiguous_temporal_span: bool = True,
    run_final_checks: bool = True,
) -> list[Feature | MemoryClusterRecord]:
    """Return the features and summaries that cover the whole time line.

    Inputs:
        ``features``: candidate raw events. Only the ones with ``causal=True``
            take part; the caller normally passes only those.
        ``temporal_summaries``: candidate summaries. Only the ones with
            ``cluster_type == ClusterType.Temporal`` take part. Topical
            clusters are ignored on purpose, because the goal is gapless
            temporal coverage, not semantic grouping.
        ``embedding``: the query vector that sets relevance.
        ``ratio``: the wanted share of tokens spent on raw features, 0.0 to 1.0.
        ``max_tokens``: the token budget. A value of 0 or less removes the cap,
            and the solver then picks by utility alone.
        ``vector_of``: returns the stored vector for one record id, or None.
            The solver calls it as ``vector_of(feature.id)``. It reads no
            summary vector, because a summary scores by the span it covers.
        ``require_temporal_history``: used by the final checks only.
        ``require_contiguous_temporal_span``: used by the final checks only.
        ``run_final_checks``: run ``validate_temporal_context`` on the result.

    The result is one exact partition over the causal timeline, in time order.
    It is empty when no causal feature exists. This function does not change
    the lists it receives.

    Raises ``Exception`` when no cover fits ``max_tokens``, or when the final
    checks find a broken result. The caller decides how to recover.
    """
    # Work in pure causal chronological order. The whole algorithm assumes that
    # "history" means an exact partition over this feature list.
    features = sorted(
        [feature for feature in features if feature.causal],
        key=lambda item: (item.timestamp_world_begin, item.id),
    )
    # Only temporal summaries participate here. Topical clusters are intentionally
    # ignored because the goal is gapless temporal coverage, not semantic grouping.
    temporal_summaries = sorted(
        [memory for memory in temporal_summaries if memory.cluster_type == ClusterType.Temporal],
        key=lambda item: (item.timestamp_world_begin, item.timestamp_world_end, item.level, item.id),
    )
    if not features:
        return []

    timestamps = [feature.timestamp_world_begin for feature in features]
    # Relevance is measured once up front. The rest of the algorithm only consumes
    # these scalar scores instead of doing any more embedding math.
    feature_scores = [
        max(0.0, _cosine_sim(vector_of(feature.id), embedding)) ** 2 for feature in features
    ]
    feature_tokens = [_token_count(record_text(feature)) for feature in features]
    raw_total_tokens = sum(feature_tokens)

    index_by_feature_id = {feature.id: index for index, feature in enumerate(features)}
    item_token_count: dict[int, int] = {
        feature.id: feature_tokens[index] for index, feature in enumerate(features)
    }
    cumulative_feature_tokens = [0]
    cumulative_feature_scores = [0.0]
    for token_total, score in zip(feature_tokens, feature_scores):
        cumulative_feature_tokens.append(cumulative_feature_tokens[-1] + token_total)
        cumulative_feature_scores.append(cumulative_feature_scores[-1] + score)

    def raw_token_sum(start_index: int, end_index: int) -> int:
        # O(1) token sum over a contiguous feature span.
        return cumulative_feature_tokens[end_index + 1] - cumulative_feature_tokens[start_index]

    def raw_score_sum(start_index: int, end_index: int) -> float:
        # O(1) relevance sum over a contiguous feature span.
        return cumulative_feature_scores[end_index + 1] - cumulative_feature_scores[start_index]

    def span_from_item(item: Feature | MemoryClusterRecord) -> tuple[int, int]:
        # Convert either a raw feature or a summary into feature-index coverage.
        # The solver path computes spans inline; this helper is for inspection.
        if isinstance(item, Feature):
            index = index_by_feature_id[item.id]
            return index, index
        start_index = bisect_left(timestamps, item.timestamp_world_begin)
        end_index = bisect_right(timestamps, item.timestamp_world_end) - 1
        return start_index, end_index

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
        # Snap the summary to the raw-feature timeline. If it covers no feature, it
        # cannot help build a valid exact history cover.
        start_index = bisect_left(timestamps, summary.timestamp_world_begin)
        end_index = bisect_right(timestamps, summary.timestamp_world_end) - 1
        if start_index > end_index:
            continue
        span_length = end_index - start_index + 1
        span_raw_tokens = raw_token_sum(start_index, end_index)
        span_mean_score = raw_score_sum(start_index, end_index) / span_length
        span_peak_score = max(feature_scores[start_index:end_index + 1])
        level_weight = min(float(summary.level), 6.0) / 6.0
        summary_tokens = max(1, summary.token or _token_count(record_text(summary)))
        item_token_count[summary.id] = summary_tokens
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

    def prune_states(states: dict[int, _State]) -> dict[int, _State]:
        # For the same or greater token spend, a lower-utility suffix can never win.
        # This cheap skyline pruning keeps the DP from exploding completely.
        ordered = sorted(states.items(), key=lambda pair: pair[0])
        pruned: dict[int, _State] = {}
        best_utility = float("-inf")
        for token_total, state in ordered:
            if state.utility > best_utility:
                pruned[token_total] = state
                best_utility = state.utility
        return pruned

    def compute_min_token_cover() -> _State:
        # Diagnostic-only DP: ignore utility entirely and compute the cheapest exact
        # gapless cover that exists in the current candidate graph.
        best_tokens = [0] * (len(features) + 1)
        best_feature_tokens = [0] * (len(features) + 1)
        best_items: list[list[Feature | MemoryClusterRecord]] = [[] for _ in range(len(features) + 1)]

        for start_index in range(len(features) - 1, -1, -1):
            chosen_total_tokens: int | None = None
            chosen_feature_tokens = 0
            chosen_items: list[Feature | MemoryClusterRecord] = []
            for candidate in candidates_by_start[start_index]:
                next_index = candidate.end_index + 1
                total_tokens = candidate.tokens + best_tokens[next_index]
                if chosen_total_tokens is not None and total_tokens >= chosen_total_tokens:
                    continue
                chosen_total_tokens = total_tokens
                chosen_feature_tokens = candidate.feature_tokens + best_feature_tokens[next_index]
                chosen_items = [candidate.item, *best_items[next_index]]
            if chosen_total_tokens is None:
                raise Exception("temporal_context: candidate graph is broken and cannot cover the timeline")
            best_tokens[start_index] = chosen_total_tokens
            best_feature_tokens[start_index] = chosen_feature_tokens
            best_items[start_index] = chosen_items

        return _State(
            utility=0.0,
            feature_tokens=best_feature_tokens[0],
            total_tokens=best_tokens[0],
            items=best_items[0],
        )

    def log_budget_failure_diagnostics(token_cap: int) -> None:
        min_cover_state = compute_min_token_cover()
        feature_only_tokens = raw_total_tokens
        summary_counts_by_level: dict[int, int] = {}
        summary_tokens_by_level: dict[int, int] = {}
        for summary in temporal_summaries:
            summary_counts_by_level[summary.level] = summary_counts_by_level.get(summary.level, 0) + 1
            # A summary that covers no feature never becomes a candidate, so it has
            # no entry here. Count it as 0 tokens instead of losing the whole report.
            summary_tokens = item_token_count.get(summary.id, 0)
            summary_tokens_by_level[summary.level] = summary_tokens_by_level.get(summary.level, 0) + summary_tokens

        candidate_count_by_start = [len(candidates_by_start[index]) for index in range(len(features))]
        starts_with_only_raw = [
            index
            for index, count in enumerate(candidate_count_by_start)
            if count == 1 and candidates_by_start[index][0].is_feature
        ]
        chosen_item_descriptions = []
        for item in min_cover_state.items[:40]:
            if isinstance(item, Feature):
                chosen_item_descriptions.append(
                    f"feature id={item.id} ts={item.timestamp_world_begin.isoformat()} tokens={item_token_count[item.id]}"
                )
            else:
                chosen_item_descriptions.append(
                    f"summary id={item.id} level={item.level} ts={item.timestamp_world_begin.isoformat()}..{item.timestamp_world_end.isoformat()} tokens={item_token_count[item.id]}"
                )

        logger.error(
            "\n".join(
                [
                    "temporal_context budget failure diagnostics",
                    f"max_tokens={token_cap}",
                    f"requested_ratio={ratio}",
                    f"require_temporal_history={require_temporal_history}",
                    f"require_contiguous_temporal_span={require_contiguous_temporal_span}",
                    f"feature_count={len(features)} temporal_summary_count={len(temporal_summaries)}",
                    f"feature_time_start={features[0].timestamp_world_begin.isoformat()}",
                    f"feature_time_end={features[-1].timestamp_world_end.isoformat()}",
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

    def solve_with_budget_legacy(feature_bonus: float, token_cap: int) -> _State:
        # Exact-cover DP with a token budget. The suffix state at position i contains
        # every non-dominated way to cover features[i:].
        best_from_index: list[dict[int, _State]] = [{} for _ in range(len(features) + 1)]
        best_from_index[len(features)] = {0: _State(utility=0.0, feature_tokens=0, total_tokens=0, items=[])}

        for start_index in range(len(features) - 1, -1, -1):
            current_states: dict[int, _State] = {}
            for candidate in candidates_by_start[start_index]:
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
            best_from_index[start_index] = prune_states(current_states)

        if not best_from_index[0]:
            log_budget_failure_diagnostics(token_cap)
            raise Exception("temporal_context: no exact history cover fits inside max_tokens")
        return max(best_from_index[0].values(), key=lambda state: (state.utility, state.feature_tokens, -state.total_tokens))

    def solve_with_budget(feature_bonus: float, token_cap: int) -> _State:
        # NumPy-backed exact-cover DP. This keeps the same skyline state model as
        # solve_with_budget_legacy, but vectorizes each candidate x suffix join and
        # stores backpointers instead of copying item lists through every state.
        empty_int = np.array([], dtype=np.int64)
        empty_float = np.array([], dtype=np.float64)
        best_from_index: list[_BudgetStates] = [
            _BudgetStates(empty_int, empty_float, empty_int, empty_int, empty_int)
            for _ in range(len(features) + 1)
        ]
        best_from_index[len(features)] = _BudgetStates(
            tokens=np.array([0], dtype=np.int64),
            utilities=np.array([0.0], dtype=np.float64),
            feature_tokens=np.array([0], dtype=np.int64),
            candidate_indices=np.array([-1], dtype=np.int64),
            tail_tokens=np.array([-1], dtype=np.int64),
        )

        for start_index in range(len(features) - 1, -1, -1):
            current_utilities = np.full(token_cap + 1, -np.inf, dtype=np.float64)
            current_feature_tokens = np.zeros(token_cap + 1, dtype=np.int64)
            current_candidate_indices = np.full(token_cap + 1, -1, dtype=np.int64)
            current_tail_tokens = np.full(token_cap + 1, -1, dtype=np.int64)

            for candidate_index, candidate in enumerate(candidates_by_start[start_index]):
                tail_states = best_from_index[candidate.end_index + 1]
                if tail_states.tokens.size == 0:
                    continue

                total_tokens = tail_states.tokens + candidate.tokens
                valid_mask = total_tokens <= token_cap
                if not np.any(valid_mask):
                    continue

                valid_total_tokens = total_tokens[valid_mask]
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
            log_budget_failure_diagnostics(token_cap)
            raise Exception("temporal_context: no exact history cover fits inside max_tokens")

        best_position = max(
            range(root_states.tokens.size),
            key=lambda index: (
                root_states.utilities[index],
                root_states.feature_tokens[index],
                -root_states.tokens[index],
            ),
        )
        items: list[Feature | MemoryClusterRecord] = []
        start_index = 0
        total_tokens = int(root_states.tokens[best_position])
        utility = float(root_states.utilities[best_position])
        feature_token_total = int(root_states.feature_tokens[best_position])

        while start_index < len(features):
            states = best_from_index[start_index]
            state_position = int(np.searchsorted(states.tokens, total_tokens))
            candidate = candidates_by_start[start_index][int(states.candidate_indices[state_position])]
            items.append(candidate.item)
            total_tokens = int(states.tail_tokens[state_position])
            start_index = candidate.end_index + 1

        return _State(
            utility=utility,
            feature_tokens=feature_token_total,
            total_tokens=int(root_states.tokens[best_position]),
            items=items,
        )

    def solve_without_budget(feature_bonus: float) -> _State:
        # Same exact-cover DP, but now only utility matters because there is no
        # token cap. This naturally tends toward raw expansion in relevant spans.
        best_utility = [0.0] * (len(features) + 1)
        best_items: list[list[Feature | MemoryClusterRecord]] = [[] for _ in range(len(features) + 1)]
        best_feature_tokens = [0] * (len(features) + 1)
        best_total_tokens = [0] * (len(features) + 1)

        for start_index in range(len(features) - 1, -1, -1):
            chosen_state = _State(utility=float("-inf"), feature_tokens=0, total_tokens=0, items=[])
            for candidate in candidates_by_start[start_index]:
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

    solve_target_cache: dict[float, _State] = {}

    def solve_target(feature_bonus: float) -> _State:
        # The ratio search and the final checks ask for the same feature_bonus values
        # multiple times, so memoizing here saves a lot of repeated DP work.
        if feature_bonus in solve_target_cache:
            return solve_target_cache[feature_bonus]
        if max_tokens > 0:
            solver = solve_with_budget if USE_NUMPY_BUDGET_SOLVER else solve_with_budget_legacy
            state = solver(feature_bonus=feature_bonus, token_cap=int(max_tokens))
        else:
            state = solve_without_budget(feature_bonus=feature_bonus)
        solve_target_cache[feature_bonus] = state
        return state

    if max_tokens > 0 and raw_total_tokens <= int(max_tokens):
        # If the full raw history already fits, compressed summaries are pure
        # redundancy. The intended behavior is to return the raw timeline only.
        result = list(features)
        if run_final_checks:
            validate_temporal_context(
                result=result,
                features=features,
                temporal_summaries=temporal_summaries,
                ratio=ratio,
                max_tokens=max_tokens,
                feature_tokens=feature_tokens,
                item_token_count=item_token_count,
                candidates_by_start=candidates_by_start,
                solve_target=solve_target,
                feature_bonus=999.0,
                require_temporal_history=require_temporal_history,
                require_contiguous_temporal_span=require_contiguous_temporal_span,
            )
        return result

    # Sweep a small fixed range of ratio-control values and keep the solution whose
    # raw-token share lands closest to the requested ratio.
    best_feature_bonus = MIN_FEATURE_BONUS
    best_state = solve_target(feature_bonus=best_feature_bonus)
    best_gap = abs(_feature_ratio(best_state.feature_tokens, best_state.total_tokens) - ratio)
    for step_index in range(RATIO_SWEEP_STEPS):
        current_feature_bonus = MIN_FEATURE_BONUS + ((MAX_FEATURE_BONUS - MIN_FEATURE_BONUS) * step_index / RATIO_SWEEP_DIVISOR)
        state = solve_target(feature_bonus=current_feature_bonus)
        actual_ratio = _feature_ratio(state.feature_tokens, state.total_tokens)
        gap = abs(actual_ratio - ratio)
        if gap < best_gap:
            best_gap = gap
            best_state = state
            best_feature_bonus = current_feature_bonus

    if run_final_checks:
        validate_temporal_context(
            result=best_state.items,
            features=features,
            temporal_summaries=temporal_summaries,
            ratio=ratio,
            max_tokens=max_tokens,
            feature_tokens=feature_tokens,
            item_token_count=item_token_count,
            candidates_by_start=candidates_by_start,
            solve_target=solve_target,
            feature_bonus=best_feature_bonus,
            require_temporal_history=require_temporal_history,
            require_contiguous_temporal_span=require_contiguous_temporal_span,
        )
    return best_state.items


def validate_temporal_context(
    result: list[Feature | MemoryClusterRecord],
    features: list[Feature],
    temporal_summaries: list[MemoryClusterRecord],
    ratio: float,
    max_tokens: int,
    feature_tokens: list[int],
    item_token_count: dict[int, int],
    candidates_by_start: dict[int, list[_Candidate]],
    solve_target: Callable[[float], _State],
    feature_bonus: float,
    *,
    require_temporal_history: bool = True,
    require_contiguous_temporal_span: bool = True,
) -> None:
    """Make sure that one result is a valid, optimal cover of the timeline.

    These checks are intentionally strict and crash fast. This function is meant
    for inspecting optimizer behavior, not for graceful runtime recovery. It
    changes nothing, except that it re-runs the solver through ``solve_target``.

    ``features`` and ``temporal_summaries`` are the same filtered lists the
    solver used. The dialogue coverage check measures against those lists, so a
    caller that passes a partial pool gets a partial reference.
    """
    timestamps = [feature.timestamp_world_begin for feature in features]
    index_by_feature_id = {feature.id: index for index, feature in enumerate(features)}
    all_dialog = [feature for feature in features if feature.feature_type == FeatureType.Dialogue]
    all_temporal_memories = temporal_summaries

    def item_span(item: Feature | MemoryClusterRecord) -> tuple[int, int]:
        # Translate output items back into exact raw-feature coverage spans.
        if isinstance(item, Feature):
            index = index_by_feature_id[item.id]
            return index, index
        start_index = bisect_left(timestamps, item.timestamp_world_begin)
        end_index = bisect_right(timestamps, item.timestamp_world_end) - 1
        return start_index, end_index

    if not result:
        raise Exception("temporal_context: empty result")

    spans = [item_span(item) for item in result]
    if spans[0][0] != 0:
        raise Exception("temporal_context: history does not start at the first causal feature")
    if spans[-1][1] != len(features) - 1:
        raise Exception("temporal_context: history does not end at the last causal feature")

    # The chosen items must form an exact partition over the entire timeline.
    for previous, current in zip(spans, spans[1:]):
        if previous[1] + 1 != current[0]:
            raise Exception("temporal_context: history is not gapless")

    if require_contiguous_temporal_span:
        for previous, current in zip(result, result[1:]):
            if previous.timestamp_world_begin > current.timestamp_world_begin:
                raise Exception("temporal_context: output order is not chronological")
            if previous.timestamp_world_end > current.timestamp_world_begin and item_span(previous)[1] < item_span(current)[0]:
                raise Exception("temporal_context: timestamps overlap while spans do not")

    raw_selected_ids = {item.id for item in result if isinstance(item, Feature)}
    covered_by_summary = [False] * len(features)
    for item, (start_index, end_index) in zip(result, spans):
        if isinstance(item, MemoryClusterRecord):
            for index in range(start_index, end_index + 1):
                covered_by_summary[index] = True

    # Every raw feature not emitted directly must still be covered by at least one
    # selected summary. Otherwise the history has a silent hole.
    for index, feature in enumerate(features):
        if feature.id in raw_selected_ids:
            continue
        if not covered_by_summary[index]:
            raise Exception("temporal_context: summarized feature is not covered by any selected summary")

    # Mixing a summary with raw items fully inside that summary is redundant and
    # therefore forbidden.
    for item, (start_index, end_index) in zip(result, spans):
        if isinstance(item, Feature):
            continue
        for other_item, (other_start, other_end) in zip(result, spans):
            if not isinstance(other_item, Feature):
                continue
            if start_index <= other_start <= other_end <= end_index:
                raise Exception("temporal_context: summary overlaps selected raw data")

    total_tokens = sum(item_token_count[item.id] for item in result)
    raw_token_total = sum(feature_tokens[index_by_feature_id[item.id]] for item in result if isinstance(item, Feature))
    actual_ratio = _feature_ratio(raw_token_total, total_tokens)
    all_raw_selected = len(result) == len(features) and all(isinstance(item, Feature) for item in result)

    if max_tokens > 0 and total_tokens > int(max_tokens):
        raise Exception("temporal_context: result exceeds max_tokens")

    if not all_raw_selected:
        # The ratio is not continuous because the candidate space is discrete. So we
        # compare against the best ratio actually reachable by this optimizer sweep.
        best_ratio_gap = abs(actual_ratio - ratio)
        for step_index in range(RATIO_SWEEP_STEPS):
            current_feature_bonus = MIN_FEATURE_BONUS + ((MAX_FEATURE_BONUS - MIN_FEATURE_BONUS) * step_index / RATIO_SWEEP_DIVISOR)
            current_state = solve_target(current_feature_bonus)
            current_ratio = _feature_ratio(current_state.feature_tokens, current_state.total_tokens)
            best_ratio_gap = min(best_ratio_gap, abs(current_ratio - ratio))
        if abs(actual_ratio - ratio) > best_ratio_gap + 0.01:
            raise Exception("temporal_context: ratio is achievable more closely than the returned output")

    # Re-solve for the selected feature_bonus and make sure we did not somehow
    # mutate or return a non-optimal path afterwards.
    optimal_state = solve_target(feature_bonus)
    optimal_ids = [item.id for item in optimal_state.items]
    result_ids = [item.id for item in result]
    if result_ids != optimal_ids:
        raise Exception("temporal_context: result is not the actual optimizer output")

    if require_temporal_history and len(all_temporal_memories) > 0:
        # If the memory store exists, selected output should cover dialogue at least
        # as well as the temporal memory base itself.
        missing_dialog = []
        missing_dialog_reference = []
        for dialog in all_dialog:
            if not any(dialog.timestamp_world_begin >= item.timestamp_world_begin and dialog.timestamp_world_end <= item.timestamp_world_end for item in result):
                missing_dialog.append(dialog)
            if not any(dialog.timestamp_world_begin >= memory.timestamp_world_begin and dialog.timestamp_world_end <= memory.timestamp_world_end for memory in all_temporal_memories):
                missing_dialog_reference.append(dialog)
        if len(missing_dialog) != len(missing_dialog_reference):
            print(f"dialog not in history parts: {len(missing_dialog)}")
            raise Exception("temporal_context: dialog coverage diverged from temporal memory coverage")

    # Every raw feature start index must have at least one legal continuation.
    for start_index, candidates in candidates_by_start.items():
        if not candidates:
            raise Exception("temporal_context: a feature start index has no candidates")
        if candidates[0].start_index != start_index:
            raise Exception("temporal_context: candidate indexing is corrupted")


__all__ = [
    "build_temporal_context",
    "validate_temporal_context",
    "MIN_FEATURE_BONUS",
    "MAX_FEATURE_BONUS",
    "RATIO_SWEEP_STEPS",
    "RATIO_SWEEP_DIVISOR",
    "USE_NUMPY_BUDGET_SOLVER",
]
