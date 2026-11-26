"""Plan which memory records go into one prompt, lane by lane, within a token budget.

This is the v2 port of the ``pm_next`` ``ContextPlanner`` idea. A plan has
ordered lanes. Each lane owns a share of the token budget and selects records
from one part of memory:

- a ``recent`` lane takes the newest contiguous records that fit, for example
  the verbatim dialogue window;
- a ``score`` lane ranks its candidates against a ``Focus`` and fills its
  budget greedily, for example summaries of earlier times, facts, or
  narratives;
- a ``focus`` lane holds only the records that the current workspace points
  at, for example the targets of winning attention nominations.

Lane rules:

- A record appears in at most one lane. Lanes run in processing order:
  ``recent`` lanes first (they fix the time boundary), then the others in
  the order the spec lists them.
- ``before_lane`` keeps a lane strictly older than the oldest record of the
  named lane, so a history summary never repeats the verbatim window.
- ``latest_per`` keeps only the newest record per key, for example the latest
  refinement of each narrative.
- ``max_similarity`` skips near duplicates of records already chosen in the
  lane, so a fact that repeats in five wordings fills one place.
- After a full pass, unused budget from lanes that ran out of candidates moves
  to the lanes in ``priority`` that still had candidates left, and the whole
  plan runs again. This repeats at most ``redistribution_rounds`` times.

The planner is pure. It never reads the MemoryStore, calls a model, or embeds
text. The caller supplies typed record references (already filtered by frontier and
scope), a vector lookup, and a token counter. Records without a vector score
by their lexical and importance terms only.
"""
from __future__ import annotations

import math
from array import array
from collections import Counter
from datetime import datetime
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from pm.memory.embeddings import valid_vector
from pm.memory.cluster_membership import ClusterMembership, is_episode_cluster
from pm.records.base import RecordBase
from pm.records.memory_records import Feature, MemoryClusterRecord, DeclarativeFactRecord, SemanticRecord, Narrative
from pm.memory.retrieval import distinctive_terms, record_text, utc_time


class LaneWeights(BaseModel):
    """Score terms of a ``score`` lane. Missing signals contribute 0."""

    model_config = ConfigDict(extra="forbid")
    semantic: float = 1.0
    lexical: float = 0.35
    importance: float = 0.0
    # Multiply the score by this factor per day of record age. None disables it.
    decay_per_day: float | None = None


class LaneSpec(BaseModel):
    """One lane of a context plan."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    budget_share: float = Field(ge=0.0, le=1.0)
    select: Literal["recent", "score", "focus"] = "score"
    order: Literal["time", "score"] = "time"
    # Serialized record types, for example "Feature" or "MemoryClusterKnoxel".
    record_types: list[str] = Field(default_factory=list)
    feature_types: list[str] = Field(default_factory=list)
    cluster_types: list[str] = Field(default_factory=list)
    # Filter SemanticallyClassifiedRecord by its record_type taxonomy value.
    # Example: ["fact", "preference", "trait"] selects only those claim kinds.
    # Empty list means no semantic filter (all semantic records of matching type pass).
    semantic_types: list[str] = Field(default_factory=list)
    latest_per: list[str] = Field(default_factory=list)
    before_lane: str | None = None
    # Skip a candidate whose vector is at least this similar to an already
    # chosen record, or whose text is identical. None disables the check.
    max_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    weights: LaneWeights = Field(default_factory=LaneWeights)
    # Temporal sampling mode. "score_only" fills greedily by score.
    # "preserve_history" uses the DP solver for gapless timeline coverage.
    temporal_mode: Literal["score_only", "preserve_history"] = "score_only"
    # For preserve_history: target fraction of tokens spent on raw features
    # vs temporal summaries. 0.0 = all summaries, 1.0 = all raw.
    raw_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    # Hard cap on this lane's budget regardless of redistribution.
    # None means no cap. Prevents a high-priority lane from absorbing all spare.
    max_tokens: int | None = Field(default=None, gt=0)
    type_token_shares: dict[str, float] = Field(default_factory=dict[str, float])
    type_item_limits: dict[str, int] = Field(default_factory=dict[str, int])
    expand_topical_blocks: bool = False
    expand_summary_children: bool = False



class PlanSpec(BaseModel):
    """A named plan: total budget, lanes, and redistribution priority."""

    model_config = ConfigDict(extra="forbid")
    name: str
    total_tokens: int = Field(gt=0)
    lanes: list[LaneSpec]
    priority: list[str] = Field(default_factory=list)
    redistribution_rounds: int = Field(default=3, ge=0)


class WeightedFocusText(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    weight: float
    text: str


class WeightedFocusVector(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    weight: float
    vector: list[float]


@dataclass(frozen=True, slots=True)
class NormalizedFocusVector:
    weight: float
    vector: array[float]


@dataclass(frozen=True, slots=True)
class WeightedFocusTerm:
    weight: float
    term: str


class Focus(BaseModel):
    """What the current workspace is about.

    ``texts`` and ``vectors`` are weighted queries: the workspace experience,
    the current message, and the winning attention nominations. ``required_ids``
    are records the workspace points at directly. ``exclude_ids`` never
    enter any lane, for example the current message itself.
    """

    model_config = ConfigDict(extra="forbid")
    texts: list[WeightedFocusText] = Field(default_factory=list[WeightedFocusText])
    vectors: list[WeightedFocusVector] = Field(default_factory=list[WeightedFocusVector])
    required_ids: list[int] = Field(default_factory=list[int])
    exclude_ids: list[int] = Field(default_factory=list[int])


class CompetingRecord(BaseModel):
    id: int
    score: float | None
    reason: Literal["similar", "budget"]
    tokens: int | None = None


class BudgetRedistribution(BaseModel):
    round: int
    spare: int
    to: str


class LanePlan(BaseModel):
    """The result of one lane."""

    name: str
    budget: int
    used: int = 0
    ids: list[int] = Field(default_factory=list[int])
    candidates: int = 0
    exhausted: bool = False
    scores: dict[int, float] = Field(default_factory=dict[int, float])
    competing: list[CompetingRecord] = Field(default_factory=list[CompetingRecord])


class ContextPlan(BaseModel):
    """All lanes in spec order, plus the redistribution trace."""

    spec: str
    lanes: list[LanePlan]
    rounds: list[BudgetRedistribution] = Field(default_factory=list[BudgetRedistribution])

    def lane(self, name: str) -> LanePlan:
        return next(item for item in self.lanes if item.name == name)

    def ids(self) -> list[int]:
        return [key for lane in self.lanes for key in lane.ids]


def _timestamp(raw: RecordBase) -> datetime:
    return utc_time(raw.timestamp_world_begin)


def _end_timestamp(raw: RecordBase) -> datetime:
    return utc_time(raw.timestamp_world_end)


def _matches(raw: RecordBase, lane: LaneSpec) -> bool:
    if lane.record_types and raw.type.value not in lane.record_types:
        return False
    # When both cluster_types and feature_types are set, a record passes if it
    # matches either filter. A Feature cannot be a MemoryClusterRecord and vice
    # versa, so requiring both would always reject.
    if lane.cluster_types or lane.feature_types:
        from pm.records.memory_records import SemanticallyClassifiedRecord
        matched = False
        if lane.cluster_types and isinstance(raw, MemoryClusterRecord) and raw.cluster_type.value in lane.cluster_types:
            matched = True
        if lane.feature_types and isinstance(raw, Feature) and raw.feature_type.value in lane.feature_types:
            matched = True
        if not matched:
            return False
    if lane.semantic_types:
        from pm.records.memory_records import SemanticallyClassifiedRecord
        if not isinstance(raw, SemanticallyClassifiedRecord) or raw.record_type.value not in lane.semantic_types:
            return False
    return True


def _latest_per(keys: list[int], records: dict[int, RecordBase], fields: list[str]) -> list[int]:
    if not fields:
        return keys
    newest: dict[tuple[str, int | None], int] = {}
    for key in keys:
        record = records[key]
        if fields != ["narrative_type", "entity_id"] or not isinstance(record, Narrative):
            raise ValueError(f"Unsupported latest-per context grouping: {fields} for {record.type}.")
        group = (record.narrative_type, record.entity_id)
        if group not in newest or (_timestamp(records[key]), key) > (_timestamp(records[newest[group]]), newest[group]):
            newest[group] = key
    kept = set(newest.values())
    return [key for key in keys if key in kept]


class _Scorer:
    """Score records against one focus. Lexical weights use inverse frequency."""

    def __init__(self, records: dict[int, RecordBase], focus: Focus,
                 vectors: dict[int, list[float] | None], now: datetime) -> None:
        self.records = records
        self.focus = focus
        self.vectors = vectors
        # One plan owns a frozen record view. Keep vectors and semantic scores
        # across its lanes and budget rounds. A new plan gets fresh caches.
        self._vectors: dict[int, array[float] | None] = {}
        self._semantic: dict[int, float] = {}
        self._queries = [NormalizedFocusVector(query.weight, self._normalized(query.vector)) for query in focus.vectors]
        self.now = now
        self.terms = [WeightedFocusTerm(query.weight, term) for query in focus.texts for term in distinctive_terms(query.text)]
        texts = {key: record_text(raw).casefold() for key, raw in records.items()}
        self.texts = texts
        frequency = Counter(item.term for item in self.terms for text in texts.values() if item.term in text)
        self.idf = {item.term: math.log(1 + len(texts) / (1 + frequency[item.term])) for item in self.terms}
        self.lexical_max = max((self._lexical_raw(key) for key in texts), default=1.0) or 1.0

    @staticmethod
    def _normalized(vector: list[float]) -> array[float]:
        """Validate once and retain a compact unit vector for dot products."""
        valid_vector(vector, len(vector))
        norm = math.hypot(*vector)
        return array("d", (value / norm for value in vector))

    def _vector(self, key: int) -> array[float] | None:
        """Read each record vector once during this frozen plan."""
        if key not in self._vectors:
            vector = self.vectors[key]
            self._vectors[key] = self._normalized(vector) if vector is not None else None
        return self._vectors[key]

    def _lexical_raw(self, key: int) -> float:
        text = self.texts.get(key, "")
        return sum(item.weight * self.idf[item.term] for item in self.terms if item.term in text)

    def similar(self, key: int, chosen: list[int], threshold: float) -> bool:
        """Return whether a candidate repeats selected text or a unit vector.

        Large remote contexts can admit thousands of records. Batch their dot
        products in NumPy, then confirm possible matches with ``math.sumprod``.
        The conservative floating-point margin keeps threshold decisions with
        the original calculation, including values close to the boundary.
        """
        text = self.texts.get(key, "")
        vector = self._vector(key)
        selected_vectors: list[array[float]] = []
        for other in chosen:
            if text and text == self.texts.get(other, ""):
                return True
            other_vector = self._vector(other)
            if vector is not None and other_vector is not None:
                selected_vectors.append(other_vector)
        if vector is None or not selected_vectors:
            return False
        matrix = np.asarray(selected_vectors, dtype=np.float64)
        similarities = np.einsum("ij,j->i", matrix, np.asarray(vector), optimize=False)
        # Each unit-vector dot product has at most one rounding error per
        # coordinate. Leave extra room for NumPy's reduction order; all possible
        # matches still use the original scalar comparison.
        rounding_margin = 4 * len(vector) * np.finfo(np.float64).eps
        possible_matches = np.flatnonzero(similarities >= threshold - rounding_margin)
        for index in possible_matches:
            if math.sumprod(vector, selected_vectors[int(index)]) >= threshold:
                return True
        return False

    def score(self, key: int, weights: LaneWeights) -> float:
        if key not in self._semantic:
            vector = self._vector(key)
            total = sum(query.weight for query in self._queries) or 1.0
            self._semantic[key] = (sum(query.weight * math.sumprod(vector, query.vector) for query in self._queries) / total
                                   if vector is not None and self._queries else 0.0)
        semantic = self._semantic[key]
        lexical = self._lexical_raw(key) / self.lexical_max
        record = self.records[key]
        importance = record.importance if isinstance(record, (DeclarativeFactRecord, SemanticRecord)) else 0.0
        value = weights.semantic * semantic + weights.lexical * lexical + weights.importance * importance
        if weights.decay_per_day is not None:
            age = max(0.0, (self.now - _timestamp(self.records[key])).total_seconds() / 86400.0)
            value *= weights.decay_per_day ** age
        return round(value, 6)


def selection_group(key: int, lane: LaneSpec, pool: set[int], records: dict[int, RecordBase],
                    already_selected: set[int]) -> list[int]:
    from pm.records.enums import ClusterType
    record = records[key]
    if isinstance(record, MemoryClusterRecord) and lane.expand_summary_children:
        references = record.included_event_ids or record.included_cluster_ids
        if references is None:
            raise ValueError(f"Cluster {key} has no child references to expand.")
        children = [int(value.strip()) for value in references.split(",") if value.strip()]
        if all(child in pool or child in already_selected for child in children):
            return [child for child in children if child not in already_selected]
    if lane.expand_topical_blocks:
        for cluster_id in sorted(records):
            cluster = records[cluster_id]
            if not isinstance(cluster, MemoryClusterRecord) or cluster.cluster_type is not ClusterType.Topical:
                continue
            if cluster.included_event_ids is None:
                raise ValueError(f"Topical cluster {cluster_id} has no event references.")
            children = [int(value.strip()) for value in cluster.included_event_ids.split(",") if value.strip()]
            if key in children and all(child in pool or child in already_selected for child in children):
                return [child for child in children if child not in already_selected]
    return [key]


def _run_pass(records: dict[int, RecordBase], spec: PlanSpec, budgets: dict[str, int], scorer: _Scorer,
              token_costs: dict[int, int]) -> list[LanePlan]:
    """Run every lane once with the given budgets."""
    excluded = {int(key) for key in scorer.focus.exclude_ids}
    required = [int(key) for key in scorer.focus.required_ids if int(key) in records and int(key) not in excluded]
    used_ids: set[int] = set()
    results: dict[str, LanePlan] = {}
    order = [lane for lane in spec.lanes if lane.select == "recent"] + [lane for lane in spec.lanes if lane.select != "recent"]
    for lane in order:
        plan = LanePlan(name=lane.name, budget=budgets[lane.name])
        pool = [key for key in records if key not in excluded and key not in used_ids and _matches(records[key], lane)]
        if lane.select == "focus":
            pool = [key for key in required if key not in used_ids and _matches(records[key], lane)]
        if lane.before_lane and results.get(lane.before_lane) and results[lane.before_lane].ids:
            boundary = min(_timestamp(records[key]) for key in results[lane.before_lane].ids)
            pool = [key for key in pool if _end_timestamp(records[key]) < boundary]
        pool = _latest_per(pool, records, lane.latest_per)
        plan.candidates = len(pool)
        chosen: list[int]
        if lane.temporal_mode == "preserve_history":
            chosen = _run_preserve_history_lane(lane, pool, records, scorer, token_costs, plan)
        elif lane.select == "recent":
            ranked = sorted(pool, key=lambda key: (_timestamp(records[key]), key), reverse=True)
            chosen = []
            budget_limited = False
            for key in ranked:
                cost = token_costs[key]
                if plan.used + cost > plan.budget:
                    budget_limited = True
                    break
                chosen.append(key)
                plan.used += cost
            plan.exhausted = not budget_limited
        else:
            plan.scores = {key: scorer.score(key, lane.weights) for key in pool}
            ranked = sorted(pool, key=lambda key: (-plan.scores[key], -key))
            chosen = []
            budget_limited = False
            for key in ranked:
                if lane.max_similarity is not None and scorer.similar(key, chosen, lane.max_similarity):
                    plan.competing.append(CompetingRecord(id=key, score=plan.scores.get(key), reason="similar"))
                    continue
                if key in chosen:
                    continue
                group = selection_group(key, lane, set(pool), records, used_ids | set(chosen))
                cost = sum(token_costs[record_id] for record_id in group)
                type_tokens: dict[str, int] = {}
                type_items: dict[str, int] = {}
                for record_id in [*chosen, *group]:
                    record_type = records[record_id].type.value
                    type_tokens[record_type] = type_tokens.get(record_type, 0) + token_costs[record_id]
                    type_items[record_type] = type_items.get(record_type, 0) + 1
                exceeds_type_budget = any(type_tokens.get(record_type, 0) > int(plan.budget * share)
                                          for record_type, share in lane.type_token_shares.items())
                exceeds_type_count = any(type_items.get(record_type, 0) > limit
                                         for record_type, limit in lane.type_item_limits.items())
                if plan.used + cost > plan.budget or exceeds_type_budget or exceeds_type_count:
                    budget_limited = True
                    plan.competing.append(CompetingRecord(id=key, score=plan.scores.get(key), tokens=cost, reason="budget"))
                    continue
                chosen.extend(group)
                plan.used += cost
            plan.exhausted = not budget_limited
            plan.competing = plan.competing[:8]
        if lane.order == "time":
            chosen.sort(key=lambda key: (_timestamp(records[key]), key))
        plan.ids = chosen
        used_ids.update(chosen)
        results[lane.name] = plan
    return [results[lane.name] for lane in spec.lanes]


def _run_preserve_history_lane(lane: LaneSpec, pool: list[int], records: dict[int, RecordBase],
                               scorer: _Scorer, token_costs: dict[int, int],
                               plan: LanePlan) -> list[int]:
    """Spend this lane's allowance on verified chronological coverage and score useful detail."""
    from pm.memory.temporal_context import build_temporal_context
    from pm.records.memory_records import Feature, MemoryClusterRecord
    from pm.records.enums import ClusterType

    features: list[Feature] = []
    summaries: list[MemoryClusterRecord] = []
    for key in pool:
        record = records[key]
        if isinstance(record, Feature) and record.causal:
            features.append(record)
        elif isinstance(record, MemoryClusterRecord):
            if record.cluster_type == ClusterType.Temporal or is_episode_cluster(record):
                summaries.append(record)
    if not features:
        plan.exhausted = True
        return []
    plan.scores = {key: scorer.score(key, lane.weights) for key in pool}
    selected = build_temporal_context(
        features=features,
        temporal_summaries=summaries,
        embedding=None,
        ratio=lane.raw_ratio,
        max_tokens=plan.budget,
        vectors=scorer.vectors,
        token_costs=token_costs,
        membership=ClusterMembership(records),
        relevance_scores=plan.scores,
        require_temporal_history=True,
        require_contiguous_temporal_span=True,
        run_final_checks=True, allow_coverage_omissions=True,
    )
    chosen = [item.id for item in selected]
    plan.used = sum(token_costs[key] for key in chosen)
    if plan.used > plan.budget:
        from pm.context_render.budget import RequiredContextTooLarge
        raise RequiredContextTooLarge(f"History selection uses {plan.used} tokens; lane budget is {plan.budget}.")
    plan.exhausted = len(chosen) >= len(pool)
    return chosen


def plan_context(records: dict[int, RecordBase], spec: PlanSpec, focus: Focus, *,
                 vectors: dict[int, list[float] | None], token_costs: dict[int, int],
                 now: datetime, total_tokens: int | None = None) -> ContextPlan:
    """Plan all lanes, then move unused budget to priority lanes and plan again.

    ``records`` maps each record ID to its shared typed record. ``vector_of`` returns a
    stored record embedding or None. ``tokens_of`` returns the rendered token cost of
    one record. Raises ``ValueError`` for a lane name that appears twice or
    an unknown lane in ``priority`` or ``before_lane``.
    Vectors must remain fixed during this call. Each call owns its caches,
    including missing vectors. A later call reads newly available vectors.
    """
    names = [lane.name for lane in spec.lanes]
    if len(set(names)) != len(names):
        raise ValueError(f"Plan {spec.name} repeats a lane name.")
    for name in [*spec.priority, *(lane.before_lane for lane in spec.lanes if lane.before_lane)]:
        if name not in names:
            raise ValueError(f"Plan {spec.name} names unknown lane {name}.")
    scorer = _Scorer(records, focus, vectors, now)
    total = total_tokens if total_tokens is not None else spec.total_tokens
    # A zero-share focus lane holds explicit pins on demand. Charge their
    # measured cost once before dividing the remainder among optional lanes.
    remaining = total
    unassigned_pins = set(focus.required_ids).intersection(records).difference(focus.exclude_ids)
    budgets: dict[str, int] = {}  # pm-lint: allow PM004 -- per-lane token allowances keyed by lane name
    for focus_lane in spec.lanes:
        if focus_lane.select == "focus" and focus_lane.budget_share == 0:
            pins = {key for key in unassigned_pins if _matches(records[key], focus_lane)}
            pin_cost = sum(token_costs[key] for key in pins)
            budget = min(pin_cost, remaining, focus_lane.max_tokens or total)
            budgets[focus_lane.name] = budget
            remaining -= budget
            unassigned_pins.difference_update(pins)
    for lane in spec.lanes:
        if lane.name not in budgets:
            share = int(remaining * lane.budget_share)
            budgets[lane.name] = min(share, lane.max_tokens) if lane.max_tokens is not None else share
    rounds: list[BudgetRedistribution] = []
    lanes = _run_pass(records, spec, budgets, scorer, token_costs)
    for round_index in range(spec.redistribution_rounds):
        # Only redistribute budget from lanes that genuinely cannot use more:
        # no candidates, or all candidates already selected. Lanes exhausted
        # by max_similarity filtering keep their budget (they would use it if
        # the filter were relaxed).
        spare = sum(
            lane.budget - lane.used
            for lane in lanes
            if lane.exhausted and (lane.candidates == 0 or len(lane.ids) >= lane.candidates)
        )
        hungry = [name for name in spec.priority if not next(item for item in lanes if item.name == name).exhausted]
        if spare <= 0 or not hungry:
            break
        # Distribute spare to hungry lanes in priority order, respecting each
        # lane's max_tokens cap. Only truly-exhausted lanes give up budget.
        for lane in lanes:
            if lane.exhausted and (lane.candidates == 0 or len(lane.ids) >= lane.candidates):
                budgets[lane.name] = lane.used
        remaining = spare
        for name in hungry:
            if remaining <= 0:
                break
            lane_spec = next(item for item in spec.lanes if item.name == name)
            cap = lane_spec.max_tokens
            room = (cap - budgets[name]) if cap is not None else remaining
            add = min(remaining, room)
            if add > 0:
                budgets[name] += add
                remaining -= add
        rounds.append(BudgetRedistribution(round=round_index + 1, spare=spare, to=hungry[0]))
        lanes = _run_pass(records, spec, budgets, scorer, token_costs)
    return ContextPlan(spec=spec.name, lanes=lanes, rounds=rounds)


__all__ = ["ContextPlan", "Focus", "LanePlan", "LaneSpec", "LaneWeights", "PlanSpec", "plan_context"]
