"""Read original evidence through a workflow grant and a stable snapshot.

The accepted memory store owns revisions. These tools never write records.
Construction and refresh perform no network work. Search reuses the embedding
service. Cancellation rejects results after an awaited provider request.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import asyncio
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
from inspect import iscoroutinefunction, signature
import json
from threading import Event, RLock
from typing import Callable, Iterator
from typing import TYPE_CHECKING
from weakref import WeakMethod

from fastmcp import FastMCP

from pm.memory.references import referenced_ids
from pm.memory.retrieval import GRAPH_TYPES, MEMORY_TYPES, evidence_role, memory_kind, retrieve, utc_time
from pm.memory.snapshot import GhostCancelled, GhostConflict, Snapshot
from pm.records.base import RecordBase

if TYPE_CHECKING:
    from pm.memory.store import MemoryStore
    from pm.memory.embeddings import EmbeddingService


@contextmanager
def guard_revisions(store, revisions: dict[int, int], cancel: Event | None = None, *, heads=()) -> Iterator[None]:
    """Guard cited source revisions through a local commit under the owner lock.

    Changed sources raise GhostConflict. Cancellation raises GhostCancelled.
    ``heads`` names sources that must still be current. Exact historical IDs
    remain valid after supersession when the caller does not require a head.
    The caller must perform no model or network work inside this boundary.
    """
    with store.lock:
        if cancel is not None and cancel.is_set():
            raise GhostCancelled("The memory decision is cancelled.")
        changed = [key for key, revision in revisions.items() if store.revisions.get(key) != revision
                   or key in heads and key not in store._all_knoxels]
        if changed:
            raise GhostConflict(f"Inspected memory changed: {changed}. Refresh memory and reconsider the draft.")
        yield


@dataclass(frozen=True)
class MemoryView:
    """An explicit workflow grant, independent of query relevance.

    ``record_ids`` includes permitted source and speaker records. Filters never
    expand this set. Time bounds use a half-open UTC interval. Limits bound
    each response, not the records that semantic ranking can consider.
    """

    record_ids: frozenset[int]
    scopes: frozenset[tuple[int, str]]
    frontier: tuple[int, int] | None = None
    categories: frozenset[str] | None = None
    entity_ids: frozenset[int] | None = None
    start: datetime | None = None
    end: datetime | None = None
    result_limit: int = 16
    excerpt_limit: int = 1200
    page_size: int = 4000
    graph_page_size: int = 16

    def __post_init__(self) -> None:
        if not self.scopes:
            raise ValueError("A memory view requires explicit reality/frame scopes.")
        for value, maximum in ((self.result_limit, 64), (self.excerpt_limit, 4000),
                               (self.page_size, 16000), (self.graph_page_size, 64)):
            if not 1 <= value <= maximum:
                raise ValueError(f"Memory limits must be between 1 and {maximum}.")
        if self.start and self.end and utc_time(self.start) >= utc_time(self.end):
            raise ValueError("The memory interval must have an increasing start and end.")


@dataclass(frozen=True)
class MemoryToolConfig:
    """Optional read tools for one explicit view and accepted owner.

    At least one tool group must be enabled. Semantic search uses ``embeddings``
    when supplied. Its absence produces an explicit lexical fallback.
    The host can permit refresh through a selector over accepted memory.
    That selector must preserve the workflow's visibility policy.
    """

    store: MemoryStore
    view: MemoryView
    semantic: bool = False
    lexical: bool = True
    graph: bool = False
    embeddings: EmbeddingService | None = None
    cancel: Event | None = None
    refresh_view: Callable[[], MemoryView] | None = None
    trace: Callable[[dict], None] | None = None

    def __post_init__(self) -> None:
        if not (self.semantic or self.lexical or self.graph):
            raise ValueError("Enable at least one memory tool group.")


def _snapshot(config: MemoryToolConfig, grant: MemoryView) -> Snapshot:
    """Remove hidden references before any metadata, search, or graph read."""
    original = config.store.snapshot(grant.record_ids)
    records = original.serialize_knoxels()
    allowed = {}
    for key, raw in records.items():
        data = raw["data"]
        scope = data.get("scope", {})
        if (scope.get("reality_id", -1), scope.get("frame", "in_world")) not in grant.scopes:
            continue
        if grant.frontier is not None and (data.get("tick_id", 0), data.get("sub_tick_id", 0)) > grant.frontier:
            continue
        # Supporting records need the same visibility boundary as evidence.
        if raw["__type__"] in MEMORY_TYPES | GRAPH_TYPES:
            if grant.categories is not None and memory_kind(raw) not in grant.categories:
                continue
            if grant.entity_ids is not None and data.get("entity_id") not in grant.entity_ids:
                continue
            when = utc_time(data["timestamp_world_begin"])
            if grant.start is not None and when < utc_time(grant.start):
                continue
            if grant.end is not None and when >= utc_time(grant.end):
                continue
        allowed[key] = raw
    references = {key: referenced_ids(original.get(key)) for key in allowed}
    for key, raw in allowed.items():
        if raw["__type__"] == "GraphEdge":
            # Graph endpoints are required record IDs. The generic reference
            # helper treats zero as a sentinel for optional fields elsewhere.
            references[key].update(raw["data"][field] for field in ("source_id", "target_id", "source_memory_id"))
        elif raw["__type__"] == "GraphNode":
            references[key].add(raw["data"]["source_memory_id"])
    while True:
        hidden = {key for key in allowed if not references[key].issubset(allowed)}
        if not hidden:
            break
        for key in hidden:
            del allowed[key]
    return Snapshot({key: original.get(key) for key in allowed}, {},
                    {key: original.revisions[key] for key in allowed}, original.embedding_index)


def _weak_tool(method: Callable) -> Callable:
    """Preserve a tool's schema without owning its read session.

    Validators can outlive their listener. A retained callback must not keep
    the session's snapshot, serialized records, or accepted owner alive.
    A released owner raises GhostCancelled, as a closed session does.
    """
    reference = WeakMethod(method)
    if iscoroutinefunction(method):
        async def invoke(*args, **kwargs):
            owner_method = reference()
            if owner_method is None:
                raise GhostCancelled("The memory read session is closed or cancelled.")
            return await owner_method(*args, **kwargs)
    else:
        def invoke(*args, **kwargs):
            owner_method = reference()
            if owner_method is None:
                raise GhostCancelled("The memory read session is closed or cancelled.")
            return owner_method(*args, **kwargs)
    # functools.wraps adds __wrapped__, which would retain the bound method.
    invoke.__name__ = method.__name__
    invoke.__doc__ = method.__doc__
    invoke.__annotations__ = method.__annotations__
    invoke.__signature__ = signature(method)
    return invoke


class MemoryReadSession:
    """Own one read snapshot, source trace, and cached graph adjacency.

    The host closes this session after native agent cleanup. A refresh replaces
    its snapshot explicitly. Old revisions stay in the trace. Concurrent
    refresh cannot publish a result from the previous snapshot.
    Registered tools hold weak references. The host owns the session until
    agent requests finish and its listeners stop.
    """

    def __init__(self, config: MemoryToolConfig):
        self.config = config
        self._lock = RLock()
        self._closed = False
        self.accessed: dict[int, int] = {}
        self._install(config.view)

    def _install(self, grant: MemoryView) -> None:
        with self.config.store.lock:
            snapshot = _snapshot(self.config, grant)
            self.head_ids = set(snapshot.revisions) & set(self.config.store.all_knoxels)
        records = snapshot.serialize_knoxels()
        references = {key: referenced_ids(snapshot.get(key)) for key in records}
        grant_data = {key: sorted(value) if isinstance(value, frozenset) else
                      utc_time(value).isoformat() if isinstance(value, datetime) else value
                      for key, value in asdict(grant).items()}
        identity = {"revisions": dict(snapshot.revisions), "grant": grant_data}
        snapshot_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        adjacency = defaultdict(list)
        for key, raw in records.items():
            if raw["__type__"] == "GraphEdge":
                adjacency[raw["data"]["source_id"]].append(key)
                if raw["data"]["target_id"] != raw["data"]["source_id"]:
                    adjacency[raw["data"]["target_id"]].append(key)
        self.grant, self.snapshot, self.records = grant, snapshot, records
        self.references, self.snapshot_id, self.adjacency = references, snapshot_id, adjacency

    def check_cancelled(self) -> None:
        """Raise GhostCancelled when the owner revokes this read grant."""
        if self._closed or (self.config.cancel is not None and self.config.cancel.is_set()):
            raise GhostCancelled("The memory read session is closed or cancelled.")

    def _access_record(self, record_id: int) -> RecordBase:
        """Guard the frozen source grant without serializing arbitrary metadata."""
        if record_id not in self.records:
            raise ValueError("Record is unavailable in this memory view.")
        for key in [record_id, *self._sources(record_id)]:
            self.accessed[key] = self.snapshot.revisions[key]
        return self.snapshot.get(record_id)

    def _record(self, record_id: int) -> dict:
        self._access_record(record_id)
        return self.records[record_id]

    def _sources(self, record_id: int) -> list[int]:
        """Follow stored references with cycle detection inside the frozen grant."""
        pending, found = [record_id], set()
        while pending:
            key = pending.pop()
            if key in found:
                continue
            found.add(key)
            pending.extend(self.references[key] - found)
        return sorted(found - {record_id})

    def _metadata(self, record_id: int) -> dict:
        """Return attribution and validity fields while guarding their dependencies."""
        raw = self._record(record_id)
        data = raw["data"]
        fields = ("scope", "provenance", "entity_id", "override_source", "timestamp_creation",
                  "timestamp_world_begin", "tick_id", "sub_tick_id", "name", "labels", "label",
                  "source_id", "target_id", "source_memory_id", "concept_id", "valid_at", "invalid_at",
                  "active", "status", "superseded_by_id", "superseded_by_ids", "supersedes_record_ids")
        return {"record_id": record_id, "revision": self.snapshot.revisions[record_id],
                "record_type": raw["__type__"], "memory_kind": memory_kind(raw),
                "evidence_role": evidence_role(self.snapshot, raw),
                **{key: data[key] for key in fields if key in data}}

    def _trace(self, operation: str, result: dict) -> dict:
        """Retain source revisions and reject cancellation before result delivery."""
        self.check_cancelled()
        result["snapshot_id"] = self.snapshot_id
        if self.config.trace is not None:
            self.config.trace({"operation": operation, "snapshot_id": self.snapshot_id,
                               "accessed_revisions": dict(self.accessed), "result": result})
        return result

    def view(self) -> dict:
        """Return visible coverage, bounds, and revisions of inspected records."""
        with self._lock:
            self.check_cancelled()
            return self._trace("view", {"categories": dict(Counter(raw["__type__"] for raw in self.records.values())),
                "scopes": sorted(self.grant.scopes), "frontier": self.grant.frontier,
                "tools": {"semantic": self.config.semantic, "lexical": self.config.lexical, "graph": self.config.graph},
                "limits": {"results": self.grant.result_limit, "excerpt": self.grant.excerpt_limit,
                           "characters": self.grant.page_size, "neighbors": self.grant.graph_page_size},
                "accessed_revisions": dict(list(self.accessed.items())[:self.grant.result_limit]),
                "accessed_count": len(self.accessed)})

    async def search(self, query: str, method: str = "auto", limit: int = 8) -> dict:
        """Rank permitted records and expose semantic coverage and fallback.

        Provider errors use lexical ranking. Cancellation and index errors
        propagate. No network request holds the memory-store lock.
        """
        with self._lock:
            self.check_cancelled()
            if not 1 <= limit <= self.grant.result_limit or not query.strip():
                raise ValueError("Use a nonempty query and the view's result limit.")
            if method == "auto":
                method = "combined" if self.config.semantic and self.config.lexical else "semantic" if self.config.semantic else "lexical"
            if (method not in {"semantic", "lexical", "combined"}
                    or method == "semantic" and not self.config.semantic
                    or method == "lexical" and not (self.config.lexical or self.config.graph)
                    or method == "combined" and not (self.config.semantic and self.config.lexical)):
                raise ValueError("Search method is not enabled in this view.")
            snapshot, identity = self.snapshot, self.snapshot_id
            types = (MEMORY_TYPES if self.config.semantic or self.config.lexical else frozenset()) | (GRAPH_TYPES if self.config.graph else frozenset())
        task = asyncio.create_task(retrieve(snapshot, query, self.config.embeddings if self.config.semantic else None,
            limit=limit, cancel=self.config.cancel, record_types=types, method=method, reserve_context=False))
        try:
            while not task.done():
                # Closing a group must stop its requests without setting the
                # enclosing run's shared cancellation signal.
                if self._closed:
                    raise GhostCancelled("The memory read session is closed.")
                await asyncio.wait({task}, timeout=0.05)
            result = task.result()
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        with self._lock:
            self.check_cancelled()
            if identity != self.snapshot_id:
                raise GhostConflict("Memory snapshot changed during search. Repeat the query in the refreshed view.")
            for match in result["matches"]:
                key = match["knoxel_id"]
                match["text"] = match["text"][:self.grant.excerpt_limit]
                sources = self._sources(key)
                # Source guards include attribution and every derived dependency.
                for source in [key, *sources]:
                    self._record(source)
                match["source_ids"] = sources[:self.grant.result_limit]
                match["source_count"] = len(sources)
                match.update(self._metadata(key))
            return self._trace("search", result)

    def get(self, record_id: int, offset: int = 0, length: int = 2000) -> dict:
        """Return a bounded page of the centrally rendered record.

        Offsets count characters in indented JSON. Numbered lines retain the
        page's original line positions. Unavailable IDs use one generic error.
        """
        with self._lock:
            self.check_cancelled()
            if offset < 0 or not 1 <= length <= self.grant.page_size:
                raise ValueError("Use a nonnegative offset and the view's character limit.")
            from pm.context_render.records import render_record_file
            text = render_record_file(self._access_record(record_id))
            page = text[offset:offset + length]
            line = text[:offset].count("\n") + 1
            return self._trace("get", {**self._metadata(record_id), "offset": offset,
                "lines": [{"line": line + index, "text": value} for index, value in enumerate(page.splitlines())],
                "next_offset": offset + len(page) if offset + len(page) < len(text) else None,
                "format": "pm.rendered-record.v1"})

    def sources(self, record_id: int, offset: int = 0, limit: int = 8) -> dict:
        """Page through stored source links without expanding the workflow grant."""
        with self._lock:
            self.check_cancelled()
            self._record(record_id)
            if offset < 0 or not 1 <= limit <= self.grant.result_limit:
                raise ValueError("Use a nonnegative offset and the view's result limit.")
            ids = self._sources(record_id)
            page = ids[offset:offset + limit]
            return self._trace("sources", {"records": [self._metadata(key) for key in page],
                "next_offset": offset + len(page) if offset + len(page) < len(ids) else None})

    def neighbors(self, record_id: int, direction: str = "both", offset: int = 0, limit: int = 8) -> dict:
        """Page through one stored graph hop with permitted endpoint metadata."""
        with self._lock:
            self.check_cancelled()
            self._record(record_id)
            if not self.config.graph or direction not in {"both", "incoming", "outgoing"}:
                raise ValueError("Enable graph tools and use incoming, outgoing, or both.")
            if offset < 0 or not 1 <= limit <= self.grant.graph_page_size:
                raise ValueError("Use a nonnegative offset and the view's graph limit.")
            ids = [key for key in self.adjacency[record_id] if direction == "both" or
                   self.records[key]["data"]["source_id" if direction == "outgoing" else "target_id"] == record_id]
            page = ids[offset:offset + limit]
            relations = []
            for key in page:
                data = self._record(key)["data"]
                relations.append({"edge": self._metadata(key),
                    "endpoints": [self._metadata(data[field]) for field in ("source_id", "target_id")],
                    "source_ids": self._sources(key)[:self.grant.result_limit]})
            return self._trace("neighbors", {"relations": relations,
                "next_offset": offset + len(page) if offset + len(page) < len(ids) else None})

    def refresh(self) -> dict:
        """Replace the snapshot through the host selector and clear read guards."""
        with self._lock:
            self.check_cancelled()
            if self.config.refresh_view is None:
                raise ValueError("This workflow does not permit memory refresh.")
            self._install(self.config.refresh_view())
            self.accessed.clear()
            return self.view()

    @contextmanager
    def guard(self) -> Iterator[None]:
        """Hold the accepted owner lock through the caller's local commit.

        Changed inspected sources raise GhostConflict before the commit.
        Unrelated appends do not invalidate a decision. The caller must not
        perform model or network work inside this boundary.
        """
        with self._lock:
            self.check_cancelled()
            with guard_revisions(self.config.store, self.accessed, self.config.cancel,
                                 heads=self.head_ids.intersection(self.accessed)):
                yield

    def close(self) -> None:
        """Revoke this group's tools after native agent cleanup."""
        with self._lock:
            self._closed = True

    def __enter__(self) -> MemoryReadSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def tools(self) -> dict[str, Callable[..., object]]:
        """Return weak callbacks for configured read operations.

        The caller must retain this session until its requests finish.
        Closure or owner release makes later calls raise GhostCancelled.
        """
        tools = {"memory_view": self.view, "memory_search": self.search,
                 "memory_get": self.get, "memory_sources": self.sources}
        if self.config.graph:
            tools["memory_neighbors"] = self.neighbors
        if self.config.refresh_view is not None:
            tools["memory_refresh"] = self.refresh
        return {name: _weak_tool(method) for name, method in tools.items()}

    def server(self) -> FastMCP:
        """Register scoped tools while the caller retains this read session."""
        server = FastMCP("memory")
        for name, fn in self.tools().items():
            server.tool(name=name)(fn)
        return server


__all__ = ["MemoryView", "MemoryToolConfig", "MemoryReadSession", "guard_revisions"]
