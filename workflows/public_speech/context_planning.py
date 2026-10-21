"""Plan the memory context of one standalone public reply.

A reply that main cognition chose reads the workspace shards of its cycle
(:mod:`workflows.public_speech.shard_context`). It does not retrieve memory
again. A standalone Story situation has no cycle, so it plans its context
here. The focus decides what the plan looks for:

- a workspace text of the situation,
- the current message,
- optional attention texts,
- the active goals,

each as a weighted query. Required ids enter the ``focus`` lane. The lanes
and budgets come from ``context_plans.yaml``.

Network work embeds the temporary focus queries. Record vectors come from
the central knoxels. Application startup embeds every missing record.
An unavailable embedding service falls back to lexical scoring and is
recorded in the trace; it never stops the reply. Cancellation propagates.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import tiktoken
from pydantic import BaseModel, ConfigDict, Field
from pm.utils import yaml_util

from pm.llm.http_client import ModelError
from pm.records.base import RecordBase
from pm.records.memory_records import Feature
from pm.memory.context_plan import ContextPlan, Focus, PlanSpec, plan_context
from pm.memory.retrieval import candidate, record_text, scoped_records
from workflows.public_speech.public_context import DialogueTurn, EvidenceItem

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


def _scopes(view, ids) -> set[tuple[int, str]]:
    scopes = set()
    for key in ids:
        record = view.get_knoxel_by_id(key)
        if record is not None:
            scopes.add((record.scope.reality_id, record.scope.frame.value))
    return scopes or {(-1, "in_world")}


def _turn(view, key: int, record: RecordBase) -> DialogueTurn:
    kind = "thought" if isinstance(record, Feature) and record.feature_type.value in THOUGHT_TYPES else "speech"
    speaker = record.source if isinstance(record, Feature) else "unknown"
    return DialogueTurn(speaker=speaker, text=" ".join(record.content.split()), knoxel_id=key,
                        timestamp=record.timestamp_world_begin.isoformat(), kind=kind)


async def _embed_focus(embeddings, queries, cancel) -> tuple[list[tuple[float, list[float]]], dict]:
    if embeddings is None:
        return [], {"status": "disabled"}
    try:
        vectors = [(weight, await embeddings.embed(text, cancel=cancel)) for weight, text in queries]
        return vectors, {"status": "ready", "model": embeddings.config.model}
    except ModelError as error:
        return [], {"status": "unavailable", "error": str(error), "fallback": "lexical"}


async def plan_reply_context(view, plan_name: str, focus: FocusInput, embeddings=None, *,
                             now: datetime | None = None, cancel=None) -> PlannedContext:
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
            token_cache[key] = 12 + len(_ENCODING.encode(record_text(records[key]), disallowed_special=()))
        return token_cache[key]

    def vector_of(key: int):
        return view.get(key).embedding or None

    queries = focus_queries(focus)
    vectors, semantic = await _embed_focus(embeddings, queries, cancel)
    focus_model = Focus(texts=queries, vectors=vectors, required_ids=focus.required_ids, exclude_ids=focus.exclude_ids)
    now = now or datetime.utcnow()
    plan = plan_context(records, spec, focus_model, vector_of=vector_of, tokens_of=tokens_of, now=now)
    embedded = 0
    dialogue_lanes = [lane for lane in spec.lanes if lane.select == "recent"]
    dialogue_names = {lane.name for lane in dialogue_lanes}
    dialogue = [_turn(view, key, records[key]) for lane in plan.lanes if lane.name in dialogue_names for key in lane.ids]
    evidence = []
    for lane in plan.lanes:
        if lane.name in dialogue_names:
            continue
        for key in lane.ids:
            item = EvidenceItem.model_validate(candidate(view, key, records[key], score=lane.scores.get(key, 0.0)))
            item.lane = lane.name
            evidence.append(item)
    return PlannedContext(dialogue=dialogue, evidence=evidence, trace=_trace(plan, semantic, embedded))


def _trace(plan: ContextPlan, semantic: dict, embedded: int) -> dict:
    return {
        "plan": plan.spec, "semantic": semantic, "new_embeddings": embedded, "rounds": plan.rounds,
        "lanes": [{"name": lane.name, "budget": lane.budget, "used": lane.used, "selected": len(lane.ids),
                   "candidates": lane.candidates, "exhausted": lane.exhausted, "competing": lane.competing[:4]}
                  for lane in plan.lanes],
    }


__all__ = ["FocusInput", "PlannedContext", "focus_queries", "load_plan", "plan_reply_context"]
