"""Scoped retrieval and activity queries over one accepted read snapshot.

Network ranking uses the shared embedding cache outside the Ghost lock. Each
result keeps its original ID, revision, scope, and evidence links. Background
and simulated events remain available with explicit attribution.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import math

from pm_core.ghost.references import referenced_ids
from runtime.model_io import ModelError
from workflows.dialogue_story.context import distinctive_terms

MEMORY_TYPES = frozenset({
    "Feature", "DeclarativeFactKnoxel", "SemanticRecordKnoxel", "Narrative",
    "MemoryClusterKnoxel", "CauseEffectKnoxel", "BeliefKnoxel",
    "WorldPlaceKnoxel", "WorldObjectKnoxel", "BodyEpisodeKnoxel",
})


def memory_kind(raw: dict) -> str:
    """Return the evidence category without inferring truth from generated prose."""
    data = raw["data"]
    explicit = data.get("metadata", {}).get("memory_kind")
    if explicit:
        return explicit
    if data.get("feature_type") == "StateUpdate" and data.get("metadata", {}).get("affect_summary"):
        return "internal_state"
    if raw["__type__"] in {"Narrative", "CauseEffectKnoxel"}:
        return "interpretation"
    if data.get("feature_type") == "Dialogue":
        return "utterance"
    if data.get("feature_type") in {"Thought", "InnerMonologue"}:
        return "speculation"
    if data.get("feature_type") in {"Feeling", "SubjectiveExperience"}:
        return "interpretation"
    if raw["__type__"] in {"WorldPlaceKnoxel", "WorldObjectKnoxel", "BodyEpisodeKnoxel"}:
        return "observation"
    return "attributed_record"


def record_text(raw: dict) -> str:
    """Expose accepted affect values instead of generated sensory descriptions.

    The original appraisal content remains available by its evidence ID.
    Its free-form interpretation cannot establish another factual experience.
    """
    if memory_kind(raw) == "internal_state":
        return raw["data"].get("metadata", {}).get("affect_summary") or "Accepted internal state; no prepared affect summary."
    return raw["data"]["content"]


def scoped_records(view, *, frontier: tuple[int, int] | None = None,
                   scopes: set[tuple[int, str]] | None = None) -> tuple[dict, dict]:
    """Filter before indexing. A scope contains a reality ID and a frame name.

    Linked summaries are eligible only when every original record is visible
    and within the frontier. This prevents a summary from exposing hidden data.
    """
    all_records = view.serialize_knoxels()
    eligible = {}
    rejected = {}
    for key, raw in all_records.items():
        data = raw["data"]
        position = (data.get("tick_id", 0), data.get("sub_tick_id", 0))
        scope = data.get("scope", {})
        if frontier is not None and position > frontier:
            rejected[key] = "beyond_frontier"
        elif scopes is not None and (scope.get("reality_id", -1), scope.get("frame", "in_world")) not in scopes:
            rejected[key] = "different_scope"
        else:
            eligible[key] = raw
    records = {}
    for key, raw in eligible.items():
        if raw["__type__"] not in MEMORY_TYPES or not raw["data"].get("content", "").strip():
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


def excerpt(text: str, terms: list[str], limit: int = 1200) -> str:
    """Keep a matching passage rather than always truncating the record head."""
    if len(text) <= limit:
        return text
    positions = [(text.casefold().count(term), text.casefold().find(term))
                 for term in terms if term in text.casefold()]
    match = min(positions, default=(0, 0))[1]
    start = max(0, match - limit // 4)
    return ("…" if start else "") + text[start:start + limit] + ("…" if start + limit < len(text) else "")


def index_chunks(records: dict) -> tuple[dict[int, str], dict[int, int]]:
    """Split long evidence into bounded cache texts and retain original IDs.

    A chunk contains at most 2,000 characters with 200 characters of overlap.
    No more than 8,000 UTF-8 bytes can reach the 8,192-token local endpoint.
    The cache identity remains the exact chunk text and configured model.
    """
    texts, owners = {}, {}
    for key, raw in sorted(records.items()):
        text = record_text(raw)
        for start in range(0, len(text), 1800):
            chunk_id = len(texts)
            texts[chunk_id] = text[start:start + 2000]
            owners[chunk_id] = key
            if start + 2000 >= len(text):
                break
    return texts, owners


def candidate(view, key: int, raw: dict, *, terms=None, score=0.0, reasons=(), passage=None) -> dict:
    """Render one result with inspectable source and historical-status fields."""
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
    stale = bool(metadata.get("stale")) or data.get("active") is False or data.get("status") in {"superseded", "retracted"}
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


def evidence_role(view, raw: dict) -> str:
    """Separate original reports, actual events, and generated speech.

    Imported reports retain attribution. A summary needs its original leaves.
    A companion utterance establishes speech, never its described event.
    """
    data = raw["data"]
    if data.get("scope", {}).get("frame", "in_world") != "in_world":
        return "imagined"
    if raw["__type__"] == "ActionSchemaKnoxel":
        return "intent"
    kind = memory_kind(raw)
    if kind == "executed_event":
        return "runtime_event"
    if kind == "internal_state":
        return "internal_state"
    if data.get("feature_type") == "Dialogue":
        provenance = data.get("provenance") or {}
        entity = view.get_knoxel_by_id(data.get("entity_id"))
        human = entity is not None and str(getattr(entity, "entity_class", "")) == "Human"
        if provenance.get("source_kind") == "User" or human:
            return "user_report"
        return "companion_utterance"
    if raw["__type__"] in {"WorldPlaceKnoxel", "WorldObjectKnoxel", "BodyEpisodeKnoxel"}:
        return "world_observation"
    if kind in {"speculation", "interpretation"}:
        return "interpretation"
    if referenced_ids(view.get(data["id"])):
        return "derived"
    return "background"


async def retrieve(view, query: str, embeddings=None, *, frontier=None, scopes=None,
                   limit: int = 24, cancel=None) -> dict:
    """Rank older semantic and episodic evidence within one frozen snapshot.

    Cached semantic ranking covers every eligible record. At most 64 missing
    chunks receive embeddings per call. An explicit index preparation can
    establish full coverage. Invalid provider output records a lexical fallback.
    Cancellation and local cache errors propagate.
    """
    if not 1 <= limit <= 64:
        raise ValueError("The retrieval limit must be between 1 and 64.")
    records, rejected = scoped_records(view, frontier=frontier, scopes=scopes)
    terms = distinctive_terms(query)
    texts = {key: record_text(raw) for key, raw in records.items()}
    frequency = Counter(term for term in terms for text in texts.values() if term in text.casefold())
    lexical = {key: sum(math.log(1 + len(texts) / (1 + frequency[term]))
                        for term in terms if term in text.casefold()) for key, text in texts.items()}
    recent = sorted(records, key=lambda key: (records[key]["data"].get("tick_id", 0), key), reverse=True)[:6]
    lexical_ids = sorted(records, key=lambda key: (-lexical[key], -key))
    important = sorted(records, key=lambda key: (-records[key]["data"].get("importance", 0), key))[:8]
    semantic = {}
    semantic_passages = {}
    semantic_status = {"status": "disabled"}
    coverage = {"indexed": 0, "total": len(texts), "missing": len(texts)}
    if embeddings is not None and texts:
        chunks, owners = index_chunks(records)
        plan = embeddings.inspect_index(chunks)
        missing = set(plan.missing)
        pool = list(dict.fromkeys([*lexical_ids[:48], *recent, *important]))
        pool_order = {key: index for index, key in enumerate(pool)}
        pending = sorted((key for key in missing if owners[key] in pool_order),
                         key=lambda key: (pool_order[owners[key]],
                                          -sum(term in chunks[key].casefold() for term in terms), key))[:64]
        try:
            await embeddings.prepare_index({key: chunks[key] for key in pending}, cancel=cancel)
            indexed = set(plan.hits) | set(pending)
            ranked = await embeddings.search(excerpt(query, terms, limit=2000),
                                             {key: chunks[key] for key in indexed}, cancel=cancel)
            semantic_status = {"status": "ready", "base_url": embeddings.config.base_url,
                               "model": embeddings.config.model}
        except ModelError as error:
            # Invalid vectors remain rejected by the shared service. Preserve
            # scoped lexical evidence so a provider error cannot stop cognition.
            # Cancellation and filesystem failures retain their normal errors.
            indexed = set(embeddings.inspect_index(chunks).hits)
            ranked = []
            semantic_status = {"status": "unavailable", "base_url": embeddings.config.base_url,
                               "model": embeddings.config.model, "error": str(error), "fallback": "lexical_and_recent"}
        for chunk_id, score in ranked:
            owner = owners[chunk_id]
            if score > semantic.get(owner, -1.0):
                semantic[owner] = score
                semantic_passages[owner] = chunks[chunk_id]
        incomplete = {owners[key] for key in chunks if key not in indexed}
        complete = set(records) - incomplete
        coverage = {"indexed": len(complete), "total": len(texts), "missing": len(texts) - len(complete),
                    "indexed_chunks": len(indexed), "total_chunks": len(chunks), "missing_chunks": len(chunks) - len(indexed)}
    lex_max = max(lexical.values(), default=1) or 1
    scores = {key: 0.65 * semantic.get(key, 0) + 0.35 * lexical[key] / lex_max for key in records}
    ranked_ids = sorted(records, key=lambda key: (-scores[key], -key))
    # Reserve recent context without allowing it to consume the older pool.
    reserve = min(4, limit // 4)
    # Repeated generated answers can outrank their original reports. Reserve
    # a separate source pool so recall can escape that feedback loop.
    source_reserve = min(8, limit // 3)
    archive_ids = [key for key in ranked_ids
                   if (records[key]["data"].get("provenance") or {}).get("source_kind") == "Migration"]
    original_ids = [key for key in archive_ids if evidence_role(view, records[key]) == "user_report"]
    episode_ids = [key for key in archive_ids if records[key]["__type__"] in {"MemoryClusterKnoxel", "CauseEffectKnoxel"}]
    source_pool = list(dict.fromkeys([*episode_ids[:source_reserve // 2], *original_ids]))[:source_reserve]
    selected = list(dict.fromkeys([*ranked_ids[:limit - reserve - source_reserve], *source_pool,
                                   *recent[:reserve], *ranked_ids]))[:limit]
    matches = [candidate(view, key, records[key], terms=terms, score=scores[key], reasons=[
        *( ["semantic"] if key in semantic else []),
        *( ["lexical"] if lexical[key] else []),
        *( ["recent"] if key in recent else []),
    ], passage=semantic_passages.get(key) if not lexical[key] else None) for key in selected]
    return {"matches": matches, "trace": {
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


def activity(view, start: str | datetime, end: str | datetime) -> dict:
    """Return actual atomic activity within the half-open UTC interval.

    Background, predictions, and simulations cannot establish executed work.
    Summaries keep the original IDs and do not extend events into offline time.
    """
    lower, upper = utc_time(start), utc_time(end)
    if lower >= upper:
        raise ValueError("The activity interval must have an increasing start and end.")
    events = []
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
