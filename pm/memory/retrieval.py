"""Scoped retrieval and activity queries over one accepted read snapshot.

FAISS ranks vectors stored on shared records outside the MemoryStore lock. Each
result keeps its original ID, revision, scope, and evidence links. Background
and simulated events remain available with explicit attribution.

Inputs are a :class:`pm.memory.snapshot.Snapshot` plus the caller's frontier,
scope, and record-type filters. Outputs are JSON-ready ranking results that a
workflow renders or a tool returns; nothing here writes a record or a vector.
The stages stay separate on purpose: scope filtering, then lexical and
semantic scoring, then evidence attribution in :func:`candidate`.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
import math
import re
from threading import Event
from typing import TYPE_CHECKING, Any, TypedDict

from pm.memory.references import referenced_ids
from pm.memory.evidence import memory_kind as memory_kind, evidence_role as evidence_role
from pm.records.base import RecordBase
from pm.records.memory_records import Feature
from pm.records.graph_records import GraphNode, GraphEdge, ConceptNode
from pm.memory.snapshot import Snapshot
from pm.llm.http_client import ModelError

if TYPE_CHECKING:
    from pm.memory.embeddings import EmbeddingService


_WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]+")

_STOPWORDS = frozenset({
    "about", "again", "also", "and", "any", "are", "because", "been", "but",
    "can", "could", "did", "does", "doing", "dont", "for", "from", "get",
    "going", "got", "had", "has", "have", "her", "here", "him", "his", "how",
    "into", "its", "just", "like", "made", "make", "more", "much", "not",
    "now", "off", "one", "only", "other", "our", "out", "over", "really",
    "right", "said", "same", "she", "should", "some", "still", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "too", "under", "very", "want", "was", "were", "what",
    "when", "where", "which", "while", "who", "will", "with", "would", "your",
    "youre", "youve", "well", "yeah", "okay", "even", "ever", "much", "back",
    "thing", "things", "know", "think", "thought", "tell", "said",
})


def distinctive_terms(text: str) -> list[str]:
    """Stable, lowercase, long-enough words used only as a pre-filter."""
    seen: dict[str, None] = {}
    for match in _WORD.finditer(text.casefold()):
        word = match.group(0)
        if len(word) < 4 or word in _STOPWORDS:
            continue
        seen.setdefault(word, None)
    return list(seen)


MEMORY_TYPES = frozenset({
    "Feature", "DeclarativeFactKnoxel", "SemanticRecordKnoxel", "Narrative",
    "MemoryClusterKnoxel", "CauseEffectKnoxel", "BeliefKnoxel",
    "WorldPlaceKnoxel", "WorldObjectKnoxel", "BodyEpisodeKnoxel",
})

GRAPH_TYPES = frozenset({"GraphNode", "GraphEdge", "ConceptNode"})


def record_text(raw: RecordBase | dict) -> str:
    """Expose accepted affect values instead of generated sensory descriptions.

    The original appraisal content remains available by its evidence ID.
    Its free-form interpretation cannot establish another factual experience.
    """
    if isinstance(raw, RecordBase):
        if memory_kind(raw) == "internal_state":
            return raw.metadata.get("affect_summary") or "Accepted internal state; no prepared affect summary."
        if isinstance(raw, GraphNode):
            fields = [raw.name, raw.content]
        elif isinstance(raw, GraphEdge):
            fields = [raw.fact_text, raw.content]
        elif isinstance(raw, ConceptNode):
            fields = [raw.description, raw.content]
        else:
            fields = [raw.content]
        return "\n".join(dict.fromkeys(text for text in fields if text))
    if memory_kind(raw) == "internal_state":
        return raw["data"].get("metadata", {}).get("affect_summary") or "Accepted internal state; no prepared affect summary."
    data = raw["data"]
    fields = {"GraphNode": ("name", "content"), "GraphEdge": ("fact_text", "content"),
              "ConceptNode": ("description", "content")}.get(raw["__type__"], ("content",))
    return "\n".join(dict.fromkeys(str(data[field]) for field in fields if data.get(field)))


def scoped_records(view: Snapshot, *, frontier: tuple[int, int] | None = None,
                   scopes: set[tuple[int, str]] | None = None,
                   record_types: Iterable[str] = MEMORY_TYPES) -> tuple[dict[int, RecordBase], dict[int, str]]:
    """Filter before indexing. A scope contains a reality ID and a frame name.

    Linked summaries are eligible only when every original record is visible
    and within the frontier. This prevents a summary from exposing hidden data.
    """
    all_records = view.all_knoxels
    eligible = {}
    rejected = {}
    for key, raw in all_records.items():
        position = (raw.tick_id, raw.sub_tick_id)
        scope = raw.scope
        if frontier is not None and position > frontier:
            rejected[key] = "beyond_frontier"
        elif scopes is not None and (scope.reality_id, scope.frame.value) not in scopes:
            rejected[key] = "different_scope"
        else:
            eligible[key] = raw
    records = {}
    for key, raw in eligible.items():
        if raw.type.value not in record_types or not record_text(raw).strip():
            continue
        # Legacy references can be incomplete. Such abstractions cannot supply
        # evidence in this invocation, but their visible originals still can.
        refs = referenced_ids(view.get(key))
        if not refs.issubset(eligible):
            rejected[key] = "unavailable_evidence"
            continue
        records[key] = raw
    # A summary of another summary must retain the same evidence boundary.
    # Propagate rejected abstractions until no remaining link can expose one.
    unavailable = set(rejected)
    while True:
        invalid = {key for key in records if referenced_ids(view.get(key)) & unavailable}
        if not invalid:
            break
        for key in invalid:
            records.pop(key)
            rejected[key] = "unavailable_evidence"
        unavailable.update(invalid)
    return records, rejected


def excerpt(text: str, terms: Iterable[str], limit: int = 1200) -> str:
    """Keep a matching passage rather than always truncating the record head."""
    if len(text) <= limit:
        return text
    positions = [(text.casefold().count(term), text.casefold().find(term))
                 for term in terms if term in text.casefold()]
    match = min(positions, default=(0, 0))[1]
    start = max(0, match - limit // 4)
    return ("…" if start else "") + text[start:start + limit] + ("…" if start + limit < len(text) else "")


def candidate(view: Snapshot, key: int, raw: RecordBase | dict[str, Any], *, terms: Iterable[str] | None = None,
              score: float = 0.0, reasons: Iterable[str] = (),
              passage: str | None = None) -> dict[str, Any]:
    """Render one result with inspectable source and historical-status fields.

    The returned mapping is the evidence contract that prompt planning and
    tool replies consume. It always holds ``knoxel_id``, ``text`` (label plus
    excerpt), ``revision``, ``memory_kind``, ``scope``, ``provenance``,
    ``source_ids``, ``timestamp``, ``historical``, ``score``, ``reasons``,
    ``deception_intent_id``, and ``evidence_role``. Later stages add ranking or
    metadata fields; none of them change the stored record.

    ``historical`` is true for a superseded, retracted, expired, or imported
    record and for every recalled world observation: a recalled observation
    never establishes current world state.
    """
    if isinstance(raw, RecordBase):
        raw = raw.as_payload(include_embedding=False)
    data = raw["data"]
    metadata = data.get("metadata", {})
    source_ids = set()
    pending = list(referenced_ids(view.get(key)))
    while pending:
        source_id = pending.pop()
        if source_id == key or source_id in source_ids:
            continue
        source_ids.add(source_id)
        if source_id in view.revisions:
            pending.extend(referenced_ids(view.get(source_id)) - source_ids)
    source_ids = sorted(source_ids)
    stale = (key not in view.all_knoxels or bool(metadata.get("stale")) or data.get("active") is False
             or data.get("status") in {"superseded", "retracted"}
             or bool(data.get("invalid_at") and utc_time(data["invalid_at"]) <= datetime.now(timezone.utc)))
    if raw["__type__"] in {"WorldPlaceKnoxel", "WorldObjectKnoxel", "BodyEpisodeKnoxel"}:
        stale = True  # A recalled observation never establishes current world state.
    kind = memory_kind(raw)
    imported_history = kind == "background" and bool(metadata.get("seed_sha256"))
    speaker = data.get("override_source", "")
    if not speaker and data.get("entity_id") in view.revisions:
        speaker = view.get(data["entity_id"]).content
    text = excerpt(passage if passage is not None else record_text(raw), terms or [])
    label = f"[{kind}; source={speaker or raw['__type__']}; evidence={key}"
    # The generated context must retain chronology, not just the trace JSON.
    # Otherwise, a reply can mistake an older record for post-restart speech.
    recorded_at = metadata.get("executed_at") or data.get("timestamp_world_begin")
    if recorded_at:
        label += f"; recorded_at={utc_time(recorded_at).isoformat()}"
    if data.get("belief_key"):
        label += f"; belief_key={data['belief_key']}"
    if stale:
        label += "; historical, reobserve or use the correction"
    elif imported_history:
        label += "; imported history, not a current observation"
    label += "] "
    return {
        "knoxel_id": key, "text": label + text, "revision": view.revisions[key],
        "memory_kind": kind, "scope": data.get("scope", {}),
        "provenance": data.get("provenance"), "source_ids": source_ids,
        "timestamp": data.get("timestamp_world_begin"), "historical": stale or imported_history,
        "score": round(score, 6), "reasons": list(reasons),
        "deception_intent_id": metadata.get("deception_intent_id"),
        "evidence_role": evidence_role(view, raw),
    }


async def retrieve(view: Snapshot, query: str, embeddings: EmbeddingService | None = None, *,
                   frontier: tuple[int, int] | None = None, scopes: set[tuple[int, str]] | None = None,
                   limit: int = 24, cancel: Event | None = None,
                   record_types: Iterable[str] = MEMORY_TYPES,
                   method: str = "combined", reserve_context: bool = True) -> dict[str, Any]:
    """Rank visible evidence with stored record vectors and lexical terms.

    Returns ``matches`` (see :func:`candidate`) and ``trace``: the requested and
    effective method, the query, coverage counts, the semantic service status,
    the selected and competing IDs, and the rejected records with reasons.

    Missing vectors are embedded in full. Provider failures retain lexical
    results. Cancellation propagates. No embedding files or chunk caches exist.
    """
    if not 1 <= limit <= 64:
        raise ValueError("The retrieval limit must be between 1 and 64.")
    if method not in {"lexical", "semantic", "combined"}:
        raise ValueError("Search method must be lexical, semantic, or combined.")
    records, rejected = scoped_records(view, frontier=frontier, scopes=scopes, record_types=record_types)
    terms = distinctive_terms(query)
    if not reserve_context:
        # Direct queries need short names such as oak, tea, and API.
        # The older cognition path retains its broad-context word filter.
        terms = list(dict.fromkeys(match.group(0).casefold() for match in _WORD.finditer(query)
                                   if match.group(0).casefold() not in _STOPWORDS))
    texts = {key: record_text(raw) for key, raw in records.items()}
    # Inverse document frequency over the visible set, so a term that appears
    # everywhere adds almost nothing and a rare term dominates the lexical leg.
    frequency = Counter(term for term in terms for text in texts.values() if term in text.casefold())
    lexical = {key: sum(math.log(1 + len(texts) / (1 + frequency[term]))
                        for term in terms if term in text.casefold()) for key, text in texts.items()}
    recent = sorted(records, key=lambda key: (records[key].tick_id, key), reverse=True)[:6]
    semantic = {}
    semantic_passages = {}
    semantic_status = {"status": "disabled"}
    if method != "lexical" and embeddings is None and not reserve_context:
        semantic_status = {"status": "unavailable", "reason": "embedding_service_not_configured", "fallback": "lexical"}
    coverage = {"indexed": 0, "total": len(texts), "missing": len(texts)}
    if method != "lexical" and embeddings is not None and texts:
        typed = records
        try:
            ranked = await embeddings.search(excerpt(query, terms, limit=2000), typed,
                                             index=view.embedding_index, cancel=cancel)
            semantic_status = {"status": "ready", "base_url": embeddings.config.base_url,
                               "model": embeddings.config.model}
        except ModelError as error:
            ranked = []
            semantic_status = {"status": "unavailable", "base_url": embeddings.config.base_url,
                               "model": embeddings.config.model, "error": str(error),
                               "fallback": "lexical_and_recent" if reserve_context else "lexical"}
        semantic = dict(ranked)
        indexed = sum(bool(record.embedding) for record in typed.values())
        coverage = {"indexed": indexed, "total": len(texts), "missing": len(texts) - indexed}
    lex_max = max(lexical.values(), default=1) or 1
    semantic_ready = semantic_status["status"] == "ready"
    # Prompt context keeps a semantic bias (0.65) with lexical terms as the
    # tie-breaker. A direct tool query uses the method it asked for, and drops
    # the semantic leg entirely when the provider is unavailable.
    scores = {key: (0.65 * semantic.get(key, 0) + 0.35 * lexical[key] / lex_max if reserve_context else
                    semantic.get(key, 0) if method == "semantic" and semantic_ready else
                    lexical[key] / lex_max if method == "lexical" or not semantic_ready else
                    0.65 * semantic.get(key, 0) + 0.35 * lexical[key] / lex_max) for key in records}
    ranked_ids = sorted(records, key=lambda key: (-scores[key], -key))
    if not reserve_context:
        ranked_ids = [key for key in ranked_ids if scores[key] > 0]
    # Reserve recent context without allowing it to consume the older pool.
    reserve = min(4, limit // 4) if reserve_context else 0
    # Repeated generated answers can outrank their original reports. Reserve
    # a separate source pool so recall can escape that feedback loop.
    source_reserve = min(8, limit // 3) if reserve_context else 0
    archive_ids = [key for key in ranked_ids
                   if records[key].provenance is not None and records[key].provenance.source_kind == "Migration"]
    original_ids = [key for key in archive_ids if evidence_role(view, records[key]) == "user_report"]
    episode_ids = [key for key in archive_ids if records[key].type.value in {"MemoryClusterKnoxel", "CauseEffectKnoxel"}]
    source_pool = list(dict.fromkeys([*episode_ids[:source_reserve // 2], *original_ids]))[:source_reserve]
    # Selection order is part of the result: ranked heads first, then the
    # source pool and the recent reserve that keep recall out of a feedback
    # loop, then the remaining ranked IDs to fill the limit.
    selected = list(dict.fromkeys([*ranked_ids[:limit - reserve - source_reserve], *source_pool,
                                   *recent[:reserve], *ranked_ids]))[:limit]
    matches = [candidate(view, key, records[key], terms=terms, score=scores[key], reasons=[
        *( ["semantic"] if key in semantic else []),
        *( ["lexical"] if lexical[key] else []),
        *( ["recent"] if key in recent else []),
    ], passage=semantic_passages.get(key) if not lexical[key] else None) for key in selected]
    for match in matches:
        key = match["knoxel_id"]
        match["score_components"] = {"lexical": round(lexical[key] / lex_max, 6),
                                     "semantic": semantic.get(key)}
    return {"matches": matches, "trace": {
        "requested_method": method,
        "effective_method": method if semantic_ready or method == "lexical" else "lexical",
        "query": query, "frontier": frontier, "scopes": sorted(scopes) if scopes is not None else None,
        "coverage": coverage, "semantic": semantic_status, "selected_ids": selected,
        "rejected_counts": dict(Counter(rejected.values())),
        "rejected": [{"id": key, "reason": reason} for key, reason in list(rejected.items())[:40]],
        "competing": [{"id": key, "score": round(scores[key], 6), "reason": "below_limit"}
                      for key in ranked_ids if key not in selected][:16],
    }}


def utc_time(value: str | datetime) -> datetime:
    """Normalize legacy naive timestamps as UTC, never as machine local time."""
    when = datetime.fromisoformat(value) if isinstance(value, str) else value
    return when.replace(tzinfo=timezone.utc) if when.tzinfo is None else when.astimezone(timezone.utc)


class ActivityEvent(TypedDict):
    """One record that shows executed work at a known UTC instant."""

    id: int
    executed_at: str
    kind: str
    text: str
    provenance: dict[str, Any] | None


class ActivitySummary(TypedDict):
    """Executed activity of one half-open UTC interval, in time order."""

    start: str
    end: str
    events: list[ActivityEvent]
    source_ids: list[int]
    summary: str


def activity(view: Snapshot, start: str | datetime, end: str | datetime) -> ActivitySummary:
    """Return actual atomic activity within the half-open UTC interval.

    Background, predictions, and simulations cannot establish executed work.
    Summaries keep the original IDs and do not extend events into offline time.
    """
    lower, upper = utc_time(start), utc_time(end)
    if lower >= upper:
        raise ValueError("The activity interval must have an increasing start and end.")
    events: list[ActivityEvent] = []
    for key, raw in view.serialize_knoxels().items():
        data = raw["data"]
        kind = memory_kind(raw)
        if kind not in {"utterance", "executed_event", "internal_state", "observation"}:
            continue
        if raw["__type__"] != "Feature":
            continue
        when = utc_time(data.get("metadata", {}).get("executed_at") or data["timestamp_creation"])
        if lower <= when < upper:
            events.append({"id": key, "executed_at": when.isoformat(), "kind": kind,
                           "text": data.get("metadata", {}).get("affect_summary", data["content"])
                           if kind == "internal_state" else data["content"],
                           "provenance": data.get("provenance")})
    events.sort(key=lambda item: (item["executed_at"], item["id"]))
    return {"start": lower.isoformat(), "end": upper.isoformat(), "events": events,
            "source_ids": [item["id"] for item in events],
            "summary": "\n".join(f"[{item['id']}] {item['executed_at']}: {item['text']}" for item in events)}
