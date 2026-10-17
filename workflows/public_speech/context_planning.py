"""Plan the memory context of one public reply from the current workspace.

Story and Assistant both call ``plan_reply_context``. The workspace decides
what the plan looks for:

- the subjective experience of the winning attention frame,
- the current message,
- the texts of the winning attention nominations (A15),
- the active goals,

each as a weighted query. Records that the winning nominations target, and
records that cognition already selected, are required in the ``focus`` lane.
The lanes and budgets come from ``context_plans.yaml``.

Network work: each focus text is embedded through the shared cache, and at
most ``max_new_embeddings`` record vectors are fetched for records near the
cut of a score lane. Everything else scores from cached vectors or lexically.
An unavailable embedding service falls back to lexical scoring and is
recorded in the trace; it never stops the reply. Cancellation propagates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import tiktoken
from pm.utils import yaml_util

from pm.llm.http_client import ModelError
from pm.memory.context_plan import ContextPlan, Focus, PlanSpec, plan_context
from pm.memory.retrieval import candidate, index_chunks, record_text, scoped_records
from workflows.public_speech.public_context import DialogueTurn

PLANS_FILE = Path(__file__).resolve().parent / "context_plans.yaml"
THOUGHT_TYPES = {"Thought", "InnerMonologue"}
_ENCODING = tiktoken.get_encoding("o200k_base")


@dataclass
class FocusInput:
    """Workspace content of the cycle that selected this reply."""

    workspace: str = ""
    current_message: str = ""
    attention_texts: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    required_ids: list[int] = field(default_factory=list)
    exclude_ids: list[int] = field(default_factory=list)


@dataclass
class PlannedContext:
    """The planned dialogue window, lane-tagged evidence, and the plan trace."""

    dialogue: list[DialogueTurn]
    evidence: list[dict]
    trace: dict


def load_plan(name: str, path: Path | None = None) -> tuple[PlanSpec, int]:
    """Return one named plan and the new-embedding bound."""
    data = yaml_util.safe_load((path or PLANS_FILE).read_text(encoding="utf-8"))
    return PlanSpec.model_validate({"name": name, **data[name]}), int(data.get("max_new_embeddings", 32))


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


def _turn(view, key: int, raw: dict) -> DialogueTurn:
    record = view.get(key)
    data = raw["data"]
    kind = "thought" if str(data.get("feature_type", "")) in THOUGHT_TYPES else "speech"
    return DialogueTurn(speaker=str(getattr(record, "source", "") or data.get("override_source") or "unknown"),
                        text=" ".join(str(data.get("content", "")).split()), knoxel_id=key,
                        timestamp=str(data.get("timestamp_world_begin") or ""), kind=kind)


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
    spec, max_new = load_plan(plan_name)
    records, _ = scoped_records(view, scopes=_scopes(view, focus.exclude_ids))
    chunks, owners = index_chunks(records)
    first_chunk: dict[int, str] = {}
    for chunk_id in sorted(chunks):
        first_chunk.setdefault(owners[chunk_id], chunks[chunk_id])
    token_cache: dict[int, int] = {}

    def tokens_of(key: int) -> int:
        if key not in token_cache:
            token_cache[key] = 12 + len(_ENCODING.encode(record_text(records[key]), disallowed_special=()))
        return token_cache[key]

    def vector_of(key: int):
        return embeddings.cache.get(first_chunk[key]) if embeddings is not None and key in first_chunk else None

    queries = focus_queries(focus)
    vectors, semantic = await _embed_focus(embeddings, queries, cancel)
    focus_model = Focus(texts=queries, vectors=vectors, required_ids=focus.required_ids, exclude_ids=focus.exclude_ids)
    now = now or datetime.utcnow()
    plan = plan_context(records, spec, focus_model, vector_of=vector_of, tokens_of=tokens_of, now=now)
    embedded = 0
    if vectors:
        # Embed records near each score lane's cut that lack a vector, then
        # plan again, so the most relevant uncached records can compete.
        near = [key for lane in plan.lanes if lane.scores
                for key in [*lane.ids, *(item["id"] for item in lane.competing)]]
        missing = [key for key in dict.fromkeys(near) if key in first_chunk and vector_of(key) is None][:max_new]
        if missing:
            try:
                await embeddings.prepare_index({key: first_chunk[key] for key in missing}, cancel=cancel)
                embedded = len(missing)
            except ModelError as error:
                semantic = {**semantic, "record_embedding_error": str(error)}
            plan = plan_context(records, spec, focus_model, vector_of=vector_of, tokens_of=tokens_of, now=now)
    dialogue_lanes = [lane for lane in spec.lanes if lane.select == "recent"]
    dialogue_names = {lane.name for lane in dialogue_lanes}
    dialogue = [_turn(view, key, records[key]) for lane in plan.lanes if lane.name in dialogue_names for key in lane.ids]
    evidence = []
    for lane in plan.lanes:
        if lane.name in dialogue_names:
            continue
        for key in lane.ids:
            item = candidate(view, key, records[key], score=lane.scores.get(key, 0.0))
            item["lane"] = lane.name
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
