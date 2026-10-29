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

from pm import vclock

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, NotRequired, TypedDict

import tiktoken
from pydantic import BaseModel, ConfigDict, Field
from pm.utils import yaml_util

from pm.records.base import RecordBase
from pm.records.memory_records import Feature
from pm.memory.context_plan import ContextPlan, Focus, PlanSpec, plan_context
from pm.memory.retrieval import candidate, memory_kind, record_text, scoped_records
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
    attention_texts: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    required_ids: list[int] = Field(default_factory=list)
    exclude_ids: list[int] = Field(default_factory=list)


class PlannedContext(BaseModel):
    """The dialogue window, the lane-tagged evidence, and the trace of one reply context.

    ``story`` is the rendered workspace shards of the cycle, or empty for a
    standalone plan. ``trace`` is the diagnostic trace of the plan or the
    shard selection. Prompts never read it.
    """

    model_config = ConfigDict(extra="forbid")

    dialogue: list[DialogueTurn] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    story: str = ""
    trace: dict = Field(default_factory=dict)


def load_plan(name: str, path: Path | None = None) -> PlanSpec:
    """Return one typed named plan from the selected YAML file."""
    data = yaml_util.safe_load((path or PLANS_FILE).read_text(encoding="utf-8"))
    return PlanSpec.model_validate({"name": name, **data[name]})


def focus_queries(focus: FocusInput) -> list[tuple[float, str]]:
    """Weight the workspace parts. The experience and the message weigh most."""
    queries = [(1.0, focus.workspace), (1.0, focus.current_message)]
    queries += [(0.7, text) for text in focus.attention_texts]
    queries += [(0.3, text) for text in focus.goals]
    return [(weight, text.strip()) for weight, text in queries if text and text.strip()]


def _scopes(view: Snapshot, ids: Sequence[int]) -> set[tuple[int, str]]:
    """Return the reality scopes of the current stimuli, or the in-world default."""
    scopes: set[tuple[int, str]] = set()
    for key in ids:
        record = view.get_knoxel_by_id(key)
        if record is not None:
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

    vectors: list[tuple[float, list[float]]]
    trace: EmbeddingFocusTrace


async def _embed_focus(
    embeddings: EmbeddingService | None,
    queries: list[tuple[float, str]],
    cancel: Event | None,
) -> EmbeddedFocus:
    """Embed each query. Configured encoder errors retain their real failure."""
    if embeddings is None:
        return EmbeddedFocus([], EmbeddingFocusTrace(status="disabled"))
    vectors: list[tuple[float, list[float]]] = []
    for weight, text in queries:
        vector = await embeddings.embed(text, cancel=cancel)
        vectors.append((weight, vector))
    return EmbeddedFocus(vectors, EmbeddingFocusTrace(status="ready", model=embeddings.config.model))


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
    token_cache: dict[int, int] = {}

    def tokens_of(key: int) -> int:
        if key not in token_cache:
            from pm.llm.token_estimate import record_tokens
            token_cache[key] = record_tokens(record_text(records[key]))
        return token_cache[key]

    def vector_of(key: int) -> list[float] | None:
        """Return the stored vector of one record, or None when it has none."""
        return view.get(key).embedding or None

    queries = focus_queries(focus)
    embedded_focus = await _embed_focus(embeddings, queries, cancel)
    focus_model = Focus(texts=queries, vectors=embedded_focus.vectors, required_ids=focus.required_ids, exclude_ids=focus.exclude_ids)
    now = now or vclock.utcnow()
    plan = plan_context(records, spec, focus_model, vector_of=vector_of, tokens_of=tokens_of, now=now,
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
            item = EvidenceItem.model_validate(candidate(view, key, rec, score=lane.scores.get(key, 0.0)))
            item.lane = lane.name
            evidence.append(item)
    dialogue.sort(key=lambda t: (t.timestamp or "", t.knoxel_id or 0))
    # This trace is the persisted JSON boundary. Keep encoder state named until
    # serialization, while retaining the existing disabled/ready wire fields.
    semantic: dict[str, str] = {"status": embedded_focus.trace["status"]}
    if "model" in embedded_focus.trace:
        semantic["model"] = embedded_focus.trace["model"]
    return PlannedContext(dialogue=dialogue, evidence=evidence, trace=_trace(plan, semantic, embedded))


def _trace(plan: ContextPlan, semantic: dict[str, Any], embedded: int) -> dict[str, Any]:
    """Return the plan receipt that the reply trace shows beside the context."""
    return {
        "plan": plan.spec, "semantic": semantic, "new_embeddings": embedded, "rounds": plan.rounds,
        "lanes": [{"name": lane.name, "budget": lane.budget, "used": lane.used, "selected": len(lane.ids),
                   "candidates": lane.candidates, "exhausted": lane.exhausted, "competing": lane.competing[:4]}
                  for lane in plan.lanes],
    }


__all__ = ["FocusInput", "PlannedContext", "focus_queries", "load_plan", "plan_reply_context"]
