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
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from pm.memory.embeddings import valid_vector
from pm.records.base import RecordBase
from pm.records.memory_records import Feature, MemoryClusterRecord, DeclarativeFactRecord, SemanticRecord
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


class PlanSpec(BaseModel):
    """A named plan: total budget, lanes, and redistribution priority."""

    model_config = ConfigDict(extra="forbid")
    name: str
    total_tokens: int = Field(gt=0)
    lanes: list[LaneSpec]
    priority: list[str] = Field(default_factory=list)
    redistribution_rounds: int = Field(default=3, ge=0)


class Focus(BaseModel):
    """What the current workspace is about.

    ``texts`` and ``vectors`` are weighted queries: the workspace experience,
    the current message, and the winning attention nominations. ``required_ids``
    are records the workspace points at directly. ``exclude_ids`` never
    enter any lane, for example the current message itself.
    """

    model_config = ConfigDict(extra="forbid")
    texts: list[tuple[float, str]] = Field(default_factory=list)
    vectors: list[tuple[float, list[float]]] = Field(default_factory=list)
    required_ids: list[int] = Field(default_factory=list)
    exclude_ids: list[int] = Field(default_factory=list)


class LanePlan(BaseModel):
    """The result of one lane."""

    name: str
    budget: int
    used: int = 0
    ids: list[int] = Field(default_factory=list)
    candidates: int = 0
    exhausted: bool = False
    scores: dict[int, float] = Field(default_factory=dict)
    competing: list[dict] = Field(default_factory=list)


class ContextPlan(BaseModel):
    """All lanes in spec order, plus the redistribution trace."""

    spec: str
    lanes: list[LanePlan]
    rounds: list[dict] = Field(default_factory=list)

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
    newest: dict[tuple, int] = {}
    for key in keys:
        group = tuple(str(getattr(records[key], field, "")) for field in fields)
        if group not in newest or (_timestamp(records[key]), key) > (_timestamp(records[newest[group]]), newest[group]):
            newest[group] = key
    kept = set(newest.values())
    return [key for key in keys if key in kept]


class _Scorer:
    """Score records against one focus. Lexical weights use inverse frequency."""

    def __init__(self, records: dict[int, RecordBase], focus: Focus,
                 vector_of: Callable[[int], list[float] | None], now: datetime) -> None:
        self.records = records
        self.focus = focus
        self.vector_of = vector_of
        # One plan owns a frozen record view. Keep vectors and semantic scores
        # across its lanes and budget rounds. A new plan gets fresh caches.
        self._vectors: dict[int, array | None] = {}
        self._semantic: dict[int, float] = {}
        self._queries = [(weight, self._normalized(vector)) for weight, vector in focus.vectors]
        self.now = now
        self.terms = [(weight, term) for weight, text in focus.texts for term in distinctive_terms(text)]
        texts = {key: record_text(raw).casefold() for key, raw in records.items()}
        self.texts = texts
        frequency = Counter(term for _, term in self.terms for text in texts.values() if term in text)
        self.idf = {term: math.log(1 + len(texts) / (1 + frequency[term])) for _, term in self.terms}
        self.lexical_max = max((self._lexical_raw(key) for key in texts), default=1.0) or 1.0

    @staticmethod
    def _normalized(vector: list[float]) -> array:
        """Validate once and retain a compact unit vector for dot products."""
        valid_vector(vector, len(vector))
        norm = math.hypot(*vector)
        return array("d", (value / norm for value in vector))

    def _vector(self, key: int) -> array | None:
        """Read each record vector once during this frozen plan."""
        if key not in self._vectors:
            vector = self.vector_of(key)
            self._vectors[key] = self._normalized(vector) if vector is not None else None
        return self._vectors[key]

    def _lexical_raw(self, key: int) -> float:
        text = self.texts.get(key, "")
        return sum(weight * self.idf[term] for weight, term in self.terms if term in text)

    def similar(self, key: int, chosen: list[int], threshold: float) -> bool:
        """Return True when ``key`` nearly repeats a chosen record."""
        text = self.texts.get(key, "")
        vector = self._vector(key)
        for other in chosen:
            if text and text == self.texts.get(other, ""):
                return True
            other_vector = self._vector(other)
            if vector is not None and other_vector is not None and math.sumprod(vector, other_vector) >= threshold:
                return True
        return False

    def score(self, key: int, weights: LaneWeights) -> float:
        if key not in self._semantic:
            vector = self._vector(key)
            total = sum(weight for weight, _ in self._queries) or 1.0
            self._semantic[key] = (sum(weight * math.sumprod(vector, query) for weight, query in self._queries) / total
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


def _run_pass(records: dict[int, RecordBase], spec: PlanSpec, budgets: dict[str, int], scorer: _Scorer,
              tokens_of: Callable[[int], int]) -> list[LanePlan]:
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
        if lane.temporal_mode == "preserve_history":
            chosen = _run_preserve_history_lane(lane, pool, records, scorer, tokens_of, plan)
        elif lane.select == "recent":
            ranked = sorted(pool, key=lambda key: (_timestamp(records[key]), key), reverse=True)
            chosen = []
            budget_limited = False
            for key in ranked:
                cost = tokens_of(key)
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
                    plan.competing.append({"id": key, "score": plan.scores.get(key), "reason": "similar"})
                    continue
                cost = tokens_of(key)
                if plan.used + cost > plan.budget:
                    budget_limited = True
                    plan.competing.append({"id": key, "score": plan.scores.get(key), "tokens": cost, "reason": "budget"})
                    continue
                chosen.append(key)
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
                               scorer: _Scorer, tokens_of: Callable[[int], int],
                               plan: LanePlan) -> list[int]:
    """Select records for a preserve_history lane using the temporal DP solver.

    Falls back to greedy score selection when the solver cannot cover the timeline.
    """
    from pm.memory.temporal_context import build_temporal_context
    from pm.records.memory_records import Feature, MemoryClusterRecord
    from pm.records.enums import ClusterType

    features = [records[key] for key in pool if isinstance(records[key], Feature) and records[key].causal]
    summaries = [records[key] for key in pool
                 if isinstance(records[key], MemoryClusterRecord) and records[key].cluster_type == ClusterType.Temporal]
    if not features:
        plan.exhausted = True
        return []
    # Use the first focus vector as the relevance query for the DP solver.
    embedding: list[float] | None = None
    if scorer.focus.vectors:
        embedding = scorer.focus.vectors[0][1]
    if embedding is None:
        # No embedding available: fall back to greedy score selection.
        plan.scores = {key: scorer.score(key, lane.weights) for key in pool}
        ranked = sorted(pool, key=lambda key: (-plan.scores[key], -key))
        chosen: list[int] = []
        for key in ranked:
            cost = tokens_of(key)
            if plan.used + cost > plan.budget:
                break
            chosen.append(key)
            plan.used += cost
        plan.exhausted = len(chosen) >= len(pool)
        return chosen
    try:
        selected = build_temporal_context(
            features=features,
            temporal_summaries=summaries,
            embedding=embedding,
            ratio=lane.raw_ratio,
            max_tokens=plan.budget,
            vector_of=scorer.vector_of,
            require_temporal_history=True,
            require_contiguous_temporal_span=True,
            run_final_checks=False,
        )
    except Exception:
        # Solver failure: fall back to greedy score selection.
        plan.scores = {key: scorer.score(key, lane.weights) for key in pool}
        ranked = sorted(pool, key=lambda key: (-plan.scores[key], -key))
        chosen = []
        for key in ranked:
            cost = tokens_of(key)
            if plan.used + cost > plan.budget:
                break
            chosen.append(key)
            plan.used += cost
        plan.exhausted = len(chosen) >= len(pool)
        return chosen
    chosen = [item.id for item in selected]
    plan.used = sum(tokens_of(key) for key in chosen)
    # The DP uses its own tiktoken count. The caller adds per-record overhead.
    # Trim from the least-relevant end if the caller's count exceeds the budget.
    while plan.used > plan.budget and len(chosen) > 1:
        removed = chosen.pop() if lane.order == "time" else chosen.pop(0)
        plan.used -= tokens_of(removed)
    plan.exhausted = True  # The DP solver fills the budget optimally.
    return chosen


def plan_context(records: dict[int, RecordBase], spec: PlanSpec, focus: Focus, *,
                 vector_of: Callable[[int], list[float] | None], tokens_of: Callable[[int], int],
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
    scorer = _Scorer(records, focus, vector_of, now)
    total = total_tokens if total_tokens is not None else spec.total_tokens
    budgets = {lane.name: min(int(total * lane.budget_share), lane.max_tokens) if lane.max_tokens else int(total * lane.budget_share) for lane in spec.lanes}
    rounds: list[dict] = []
    lanes = _run_pass(records, spec, budgets, scorer, tokens_of)
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
        rounds.append({"round": round_index + 1, "spare": spare, "to": hungry[0]})
        lanes = _run_pass(records, spec, budgets, scorer, tokens_of)
    return ContextPlan(spec=spec.name, lanes=lanes, rounds=rounds)


__all__ = ["ContextPlan", "Focus", "LanePlan", "LaneSpec", "LaneWeights", "PlanSpec", "plan_context"]
