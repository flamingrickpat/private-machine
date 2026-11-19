"""Plan retrieved memory for standalone and runtime public replies.

A runtime reply retains the workspace shards of its deciding cycle and adds
retrieval from this planner. A standalone situation has no cycle and uses the
plan directly. Final Story prompt fitting belongs to
``workflows.dialogue_story.context``. The focus decides what the plan looks for:

- a workspace text of the situation,
- the current message,
- optional attention texts,
- the active goals,

each as a weighted query. Required ids enter the ``focus`` lane. The lanes
and budgets come from ``context_plans.yaml``.

Network work embeds the temporary focus queries. Record vectors come from
the central knoxels. Application startup embeds every missing record.
An explicitly absent encoder leaves direct planning on lexical scoring.
A configured encoder owns transport retries. Its permanent errors and internal
planning errors propagate instead of producing a partial reply context.
Cancellation propagates.
"""
from __future__ import annotations

from workflows.public_speech.context_trace import ContextPlanningTrace, ContextLaneTrace, SemanticContextTrace
from workflows.public_speech.public_context import order_dialogue

from pm import vclock

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Literal, NamedTuple, NotRequired, TypedDict

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, JsonValue
from pm.utils import yaml_util

from pm.records.base import RecordBase
from pm.records.memory_records import Feature
from pm.memory.context_plan import Focus, PlanSpec, WeightedFocusText, WeightedFocusVector, plan_context
from pm.memory.temporal_context import TemporalHistoryInput, TemporalHistoryItem
from pm.records.memory_records import MemoryClusterRecord
from pm.records.enums import FeatureType, ClusterType
from pm.memory.references import referenced_ids
from pm.context_render.memory_tool import render_memory_record, MemoryRecordRenderSettings
from pm.context_render.tokens import text_tokens
from pm.memory.retrieval import candidate, memory_kind, record_text, scoped_records, utc_time
from pm.memory.evidence import evidence_role
from workflows.public_speech.public_context import DialogueTurn, EvidenceItem

if TYPE_CHECKING:
    from pm.memory.embeddings import EmbeddingService
    from pm.memory.snapshot import Snapshot

PLANS_FILE = Path(__file__).resolve().parent / "context_plans.yaml"
THOUGHT_TYPES = {"Thought", "InnerMonologue"}
_ENCODING = tiktoken.get_encoding("o200k_base")


class FocusInput(BaseModel):
    """The texts and ids that a standalone context plan looks for."""

    workspace: str = ""
    current_message: str = ""
    attention_texts: list[str] = Field(default_factory=list[str])
    goals: list[str] = Field(default_factory=list[str])
    required_ids: list[int] = Field(default_factory=list[int])
    exclude_ids: list[int] = Field(default_factory=list[int])
    current_ids: tuple[int, ...] = ()


class PlannedContext(BaseModel):
    """The dialogue window, the lane-tagged evidence, and the trace of one reply context.

    ``story`` is the rendered workspace shards of the cycle, or empty for a
    standalone plan. ``trace`` is the diagnostic trace of the plan or the
    shard selection. Prompts never read it.
    """

    model_config = ConfigDict(extra="forbid")

    dialogue: list[DialogueTurn] = Field(default_factory=list[DialogueTurn])
    evidence: list[EvidenceItem] = Field(default_factory=list[EvidenceItem])
    story: str = ""
    trace: ContextPlanningTrace = Field(default_factory=ContextPlanningTrace)
    temporal_history: TemporalHistoryInput | None = None


def load_plan(name: str, path: Path | None = None) -> PlanSpec:
    """Return one typed named plan from the selected YAML file."""
    data = TypeAdapter(dict[str, dict[str, JsonValue]]).validate_python(
        yaml_util.safe_load((path or PLANS_FILE).read_text(encoding="utf-8")))
    return PlanSpec.model_validate({"name": name, **data[name]})


def focus_queries(focus: FocusInput) -> list[WeightedFocusText]:
    """Weight the workspace parts. The experience and the message weigh most."""
    queries = [WeightedFocusText(weight=1.0, text=focus.workspace),
        WeightedFocusText(weight=1.0, text=focus.current_message)]
    queries += [WeightedFocusText(weight=0.7, text=text) for text in focus.attention_texts]
    queries += [WeightedFocusText(weight=0.3, text=text) for text in focus.goals]
    return [WeightedFocusText(weight=query.weight, text=query.text.strip()) for query in queries if query.text.strip()]


def _scopes(view: Snapshot, ids: Sequence[int]) -> set[tuple[int, str]]:
    """Return the reality scopes of the current stimuli, or the in-world default."""
    scopes: set[tuple[int, str]] = set()
    for key in ids:
        if key in view.revisions:
            record = view.get(key)
            scopes.add((record.scope.reality_id, record.scope.frame.value))
    return scopes or {(-1, "in_world")}


def _turn(view: Snapshot, key: int, record: RecordBase) -> DialogueTurn:
    """Render one record as a transcript line, marked as speech or thought."""
    kind = "thought" if isinstance(record, Feature) and record.feature_type.value in THOUGHT_TYPES else "speech"
    speaker = record.source if isinstance(record, Feature) else "unknown"
    return DialogueTurn(speaker=speaker, text=" ".join(record.content.split()), knoxel_id=key,
                        timestamp=record.timestamp_world_begin.isoformat(), kind=kind)


class EmbeddingFocusTrace(TypedDict):
    """Persisted encoder state for one context plan."""

    status: Literal["disabled", "ready"]
    model: NotRequired[str]


class EmbeddedFocus(NamedTuple):
    """The existing weighted vector contract and its named trace boundary."""

    vectors: list[WeightedFocusVector]
    trace: EmbeddingFocusTrace


async def _embed_focus(
    embeddings: EmbeddingService | None,
    queries: list[WeightedFocusText],
    cancel: Event | None,
) -> EmbeddedFocus:
    """Embed each query. Configured encoder errors retain their real failure."""
    if embeddings is None:
        return EmbeddedFocus([], EmbeddingFocusTrace(status="disabled"))
    vectors: list[WeightedFocusVector] = []
    for query in queries:
        vector = await embeddings.embed(query.text, cancel=cancel)
        vectors.append(WeightedFocusVector(weight=query.weight, vector=vector))
    return EmbeddedFocus(vectors, EmbeddingFocusTrace(status="ready", model=embeddings.config.model))


def freeze_temporal_history(view: Snapshot, records: dict[int, RecordBase], focus: FocusInput,
                            embedded: EmbeddedFocus) -> TemporalHistoryInput:
    query = embedded.vectors[0].vector if embedded.vectors else []
    items: list[TemporalHistoryItem] = []
    for record_id, record in records.items():
        if record_id in focus.current_ids:
            continue
        if isinstance(record, Feature):
            if not record.causal:
                continue
            kind = "feature"
            level = 0
            feature_type = record.feature_type
            source = record.override_source or ""
            if not source and record.entity_id in view.revisions:
                source = view.get(record.entity_id).content
            content = record.content if record.feature_type == FeatureType.Dialogue else render_memory_record(
                record, MemoryRecordRenderSettings(include_identity=False))
            if memory_kind(record) == "internal_state":
                content = record_text(record)
        elif isinstance(record, MemoryClusterRecord) and record.cluster_type == ClusterType.Temporal:
            kind = "summary"
            level = record.level
            feature_type = None
            source = record.type.value
            content = render_memory_record(record, MemoryRecordRenderSettings(include_identity=False))
        else:
            continue
        relevance = 0.0
        if query and record.embedding:
            if len(query) != len(record.embedding):
                raise ValueError(f"History source {record_id} embedding dimension differs from focus.")
            relevance = max(0.0, sum(left * right for left, right in zip(query, record.embedding, strict=True)))
        items.append(TemporalHistoryItem(record_id=record_id, started_at=utc_time(record.timestamp_world_begin),
            ended_at=utc_time(record.timestamp_world_end), kind=kind, level=level, content=content, source=source,
            feature_type=feature_type, evidence_role=evidence_role(view, record),
            source_ids=tuple(sorted(referenced_ids(record))), relevance=relevance,
            token_cost=max(1, text_tokens(content))))
    return TemporalHistoryInput(items=tuple(items))


async def plan_reply_context(view: Snapshot, plan_name: str, focus: FocusInput,
                             embeddings: EmbeddingService | None = None, *,
                             now: datetime | None = None, cancel: Event | None = None,
                             total_tokens: int | None = None) -> PlannedContext:
    """Plan one reply's context from a frozen MemoryStore view.

    Scope comes from the current stimuli, so imagined or other-reality records
    never enter. The current stimuli themselves are excluded from every lane;
    the prompt shows them as the current line.
    """
    spec = load_plan(plan_name)
    records, _ = scoped_records(view, scopes=_scopes(view, focus.exclude_ids))
    from pm.llm.token_estimate import record_tokens
    token_costs = {key: record_tokens(record_text(record)) for key, record in records.items()}
    vectors = {key: record.embedding or None for key, record in records.items()}

    queries = focus_queries(focus)
    embedded_focus = await _embed_focus(embeddings, queries, cancel)
    temporal_history = freeze_temporal_history(view, records, focus, embedded_focus)
    spec = spec.model_copy(deep=True)
    spec.lanes = [lane for lane in spec.lanes if lane.name != "history"]
    spec.priority = [name for name in spec.priority if name != "history"]
    focus_model = Focus(texts=queries, vectors=embedded_focus.vectors, required_ids=focus.required_ids, exclude_ids=focus.exclude_ids)
    now = now or vclock.utcnow()
    plan = plan_context(records, spec, focus_model, vectors=vectors, token_costs=token_costs, now=now,
                        total_tokens=total_tokens)
    embedded = 0
    dialogue_lanes = [lane for lane in spec.lanes if lane.select == "recent"]
    dialogue_names = {lane.name for lane in dialogue_lanes}
    dialogue = [_turn(view, key, records[key]) for lane in plan.lanes if lane.name in dialogue_names for key in lane.ids]
    evidence: list[EvidenceItem] = []
    for lane in plan.lanes:
        if lane.name in dialogue_names:
            continue
        for key in lane.ids:
            rec = records[key]
            # A real spoken utterance belongs in the assistant dialogue, not as
            # a memory artifact. Thoughts, summaries, facts, and narratives stay
            # in evidence (the user turns). memory_kind separates them cleanly:
            # only Dialogue features are "utterance"; Thought/InnerMonologue are
            # "speculation" and never join the spoken transcript.
            if memory_kind(rec) == "utterance":
                dialogue.append(_turn(view, key, rec))
                continue
            item = candidate(view, key, rec, score=lane.scores.get(key, 0.0))
            item.lane = lane.name
            evidence.append(item)
    dialogue = order_dialogue(dialogue)
    # This trace is the persisted JSON boundary. Keep encoder state named until
    # serialization, while retaining the existing disabled/ready wire fields.
    semantic = SemanticContextTrace(status=embedded_focus.trace["status"],
        model=embedded_focus.trace["model"] if "model" in embedded_focus.trace else "")
    trace = ContextPlanningTrace(plan=plan.spec, semantic=semantic, new_embeddings=embedded,
        rounds=tuple(plan.rounds), lanes=tuple(ContextLaneTrace(name=lane.name, budget=lane.budget,
            used=lane.used, selected=len(lane.ids), candidates=lane.candidates, exhausted=lane.exhausted,
            competing=tuple(lane.competing[:4])) for lane in plan.lanes))
    return PlannedContext(dialogue=dialogue, evidence=evidence, trace=trace, temporal_history=temporal_history)


__all__ = ["FocusInput", "PlannedContext", "focus_queries", "load_plan", "plan_reply_context"]
