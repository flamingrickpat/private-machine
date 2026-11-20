"""Read original evidence through a workflow grant and a stable snapshot.

The accepted memory store owns revisions. These tools never write records.
Construction and refresh perform no network work. Search reuses the embedding
service. Cancellation rejects results after an awaited provider request.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import asyncio
from collections.abc import Iterable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import TracebackType
import hashlib
from threading import Event, RLock
from typing import TYPE_CHECKING

from fastmcp import FastMCP
from pathlib import Path
from pydantic import BaseModel, ConfigDict, JsonValue

from pm.memory.references import referenced_ids
from pm.memory.retrieval import GRAPH_TYPES, MEMORY_TYPES, memory_kind, retrieve, utc_time
from pm.memory.snapshot import GhostCancelled, GhostConflict, Snapshot
from pm.records.base import RecordBase
from pm.records.graph_records import GraphEdge, GraphNode, ConceptNode
from pm.records.memory_records import Feature, Narrative, Stimulus
from pm.context_render.memory_tool import render_memory_record

if TYPE_CHECKING:
    from pm.memory.store import MemoryStore
    from pm.memory.embeddings import EmbeddingService


@contextmanager
def guard_revisions(store: MemoryStore, revisions: dict[int, int], cancel: Event | None = None,
                    *, heads: Iterable[int] = ()) -> Generator[None]:
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
                   or key in heads and key not in store.all_knoxels]
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


class MemoryTraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    operation: str
    snapshot_id: str
    accessed_revisions: dict[int, int]
    tool_response: JsonValue


@dataclass(frozen=True, slots=True)
class MemoryTraceFile:
    path: Path

    def record(self, event: MemoryTraceEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(event.model_dump_json() + "\n")


class MemoryRefreshScope(StrEnum):
    AcceptedInWorld = "accepted_in_world"


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
    refresh_scope: MemoryRefreshScope | None = None
    trace: MemoryTraceFile | None = None

    def __post_init__(self) -> None:
        if not (self.semantic or self.lexical or self.graph):
            raise ValueError("Enable at least one memory tool group.")


def _snapshot(config: MemoryToolConfig, grant: MemoryView) -> Snapshot:
    """Remove hidden references before any metadata, search, or graph read."""
    original = config.store.snapshot(grant.record_ids)
    allowed: dict[int, RecordBase] = {}
    for key, record in original.all_knoxels.items():
        if (record.scope.reality_id, record.scope.frame.value) not in grant.scopes:
            continue
        if grant.frontier is not None and (record.tick_id, record.sub_tick_id) > grant.frontier:
            continue
        if record.type.value in MEMORY_TYPES | GRAPH_TYPES:
            if grant.categories is not None and memory_kind(record) not in grant.categories:
                continue
            if grant.entity_ids is not None:
                entity_id: int | None = None
                if isinstance(record, (Feature, Narrative, Stimulus, ConceptNode)):
                    entity_id = record.entity_id
                if entity_id not in grant.entity_ids:
                    continue
            when = utc_time(record.timestamp_world_begin)
            if grant.start is not None and when < utc_time(grant.start):
                continue
            if grant.end is not None and when >= utc_time(grant.end):
                continue
        allowed[key] = record
    references = {key: referenced_ids(record) for key, record in allowed.items()}
    for key, record in allowed.items():
        if isinstance(record, GraphEdge):
            references[key].update((record.source_id, record.target_id, record.source_memory_id))
        elif isinstance(record, GraphNode):
            references[key].add(record.source_memory_id)
    while True:
        hidden = {key for key in allowed if not references[key].issubset(allowed)}
        if not hidden:
            break
        for key in hidden:
            del allowed[key]
    return Snapshot(allowed, {}, {key: original.revisions[key] for key in allowed}, original.embedding_index)




class MemoryReadSession:
    """Own one read snapshot, source trace, and cached graph adjacency.

    The host closes this session after native agent cleanup. A refresh replaces
    its snapshot explicitly. Old revisions stay in the trace. Concurrent
    refresh cannot publish a result from the previous snapshot.
    Registered tools hold weak references. The host owns the session until
    agent requests finish and its listeners stop.
    """

    def __init__(self, config: MemoryToolConfig) -> None:
        self.config = config
        self._lock = RLock()
        self._closed = False
        self.accessed: dict[int, int] = {}
        self._install(config.view)

    def _install(self, grant: MemoryView) -> None:
        """Freeze one grant into a snapshot, its adjacency, and its identity.

        Membership and revisions are read under the store lock; the payload
        serialization and the snapshot hash run outside it, so a long grant
        does not block commits.
        """
        with self.config.store.lock:
            snapshot = _snapshot(self.config, grant)
            self.head_ids = set(snapshot.revisions) & set(self.config.store.all_knoxels)
        records = snapshot.all_knoxels
        references = {key: referenced_ids(record) for key, record in records.items()}
        identity = (sorted(snapshot.revisions.items()), sorted(grant.record_ids), sorted(grant.scopes),
                    grant.frontier, sorted(grant.categories) if grant.categories is not None else None,
                    sorted(grant.entity_ids) if grant.entity_ids is not None else None,
                    grant.start.isoformat() if grant.start is not None else None,
                    grant.end.isoformat() if grant.end is not None else None,
                    grant.result_limit, grant.excerpt_limit, grant.page_size, grant.graph_page_size)
        snapshot_id = hashlib.sha256(repr(identity).encode()).hexdigest()
        adjacency: defaultdict[int, list[int]] = defaultdict(list)
        for key, record in records.items():
            if isinstance(record, GraphEdge):
                adjacency[record.source_id].append(key)
                if record.target_id != record.source_id:
                    adjacency[record.target_id].append(key)
        self.grant, self.snapshot, self.records = grant, snapshot, records
        self.references, self.snapshot_id, self.adjacency = references, snapshot_id, adjacency

    def check_cancelled(self) -> None:
        """Raise GhostCancelled when the owner revokes this read grant."""
        if self._closed or (self.config.cancel is not None and self.config.cancel.is_set()):
            raise GhostCancelled("The memory read session is closed or cancelled.")

    def _access_record(self, record_id: int) -> RecordBase:
        """Guard the frozen source grant without serializing arbitrary metadata."""
        if record_id not in self.records:
            raise ValueError(f"Record {record_id} is unavailable in memory view {self.snapshot_id}.")
        for key in [record_id, *self._sources(record_id)]:
            self.accessed[key] = self.snapshot.revisions[key]
        return self.snapshot.get(record_id)

    def _sources(self, record_id: int) -> list[int]:
        pending = [record_id]
        found: set[int] = set()
        while pending:
            key = pending.pop()
            if key in found:
                continue
            found.add(key)
            pending.extend(self.references[key] - found)
        return sorted(found - {record_id})

    def _trace(self, operation: str, text: str) -> str:
        self.check_cancelled()
        if self.config.trace is not None:
            self.config.trace.record(MemoryTraceEvent(operation=operation, snapshot_id=self.snapshot_id,
                accessed_revisions=dict(self.accessed), tool_response=text))
        return text

    def view(self) -> str:
        with self._lock:
            self.check_cancelled()
            counts = Counter(record.type.value for record in self.records.values())
            lines = ["Memory view: " + ", ".join(f"{kind}: {count}" for kind, count in sorted(counts.items())),
                     f"Search results per request: {self.grant.result_limit}; maximum read page: {self.grant.page_size} characters.",
                     f"Previously inspected sources: {len(self.accessed)}."]
            if self.grant.frontier is not None:
                lines.append(f"History ends at tick {self.grant.frontier[0]}.")
            lines.append(f"Search enabled: semantic={self.config.semantic}, lexical={self.config.lexical}; graph={self.config.graph}.")
            return self._trace("view", "\n".join(lines))

    async def search(self, query: str, method: str = "auto", limit: int = 8) -> str:
        with self._lock:
            self.check_cancelled()
            if not 1 <= limit <= self.grant.result_limit or not query.strip():
                raise ValueError("Use a nonempty query and the view's result limit.")
            if method == "auto":
                if self.config.semantic and self.config.lexical:
                    method = "combined"
                elif self.config.semantic:
                    method = "semantic"
                else:
                    method = "lexical"
            if (method not in {"semantic", "lexical", "combined"}
                    or method == "semantic" and not self.config.semantic
                    or method == "lexical" and not (self.config.lexical or self.config.graph)
                    or method == "combined" and not (self.config.semantic and self.config.lexical)):
                raise ValueError("Search method is not enabled in this view.")
            snapshot, identity = self.snapshot, self.snapshot_id
            types = (MEMORY_TYPES if self.config.semantic or self.config.lexical else frozenset[str]()) | (GRAPH_TYPES if self.config.graph else frozenset[str]())
        task = asyncio.create_task(retrieve(snapshot, query, self.config.embeddings if self.config.semantic else None,
            limit=limit, cancel=self.config.cancel, record_types=types, method=method, reserve_context=False))
        try:
            while not task.done():
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
            lines = [f"Search: {query}", f"Ranking: {result.trace.effective_method}."]
            excerpt_limit = min(self.grant.excerpt_limit, 480)
            for match in result.matches:
                record = self._access_record(match.knoxel_id)
                when = record.timestamp_world_begin.isoformat(timespec="minutes")
                excerpt = match.text[:excerpt_limit]
                lines.append(f"Source {record.id} | {record.type.value} | {when} | relevance {match.score:.3f}\n{excerpt}")
                if len(match.text) > excerpt_limit:
                    lines.append("Excerpt only; use memory_get for complete evidence.")
            if not result.matches:
                lines.append("No matching sources.")
            return self._trace("search", "\n\n".join(lines))

    def get(self, record_id: int, offset: int = 0, length: int = 2000) -> str:
        with self._lock:
            self.check_cancelled()
            if offset < 0 or not 1 <= length <= self.grant.page_size:
                raise ValueError("Use a nonnegative offset and the view's character limit.")
            text = render_memory_record(self._access_record(record_id))
            page = text[offset:offset + length]
            if offset:
                page = f"Source {record_id}, continued at character {offset}:\n" + page
            if offset + length < len(text):
                page += f"\nContinues: memory_get({record_id}, offset={offset + length}, length={length})."
            return self._trace("get", page)

    def sources(self, record_id: int, offset: int = 0, limit: int = 8) -> str:
        with self._lock:
            self.check_cancelled()
            self._access_record(record_id)
            if offset < 0 or not 1 <= limit <= self.grant.result_limit:
                raise ValueError("Use a nonnegative offset and the view's result limit.")
            ids = self._sources(record_id)
            lines = [f"Supporting sources for {record_id}:"]
            for key in ids[offset:offset + limit]:
                record = self._access_record(key)
                lines.append(f"Source {key} ({record.type.value}): {record.content[:240]}")
            if offset + limit < len(ids):
                lines.append(f"More sources: memory_sources({record_id}, offset={offset + limit}, limit={limit}).")
            return self._trace("sources", "\n".join(lines))

    def neighbors(self, record_id: int, direction: str = "both", offset: int = 0, limit: int = 8) -> str:
        with self._lock:
            self.check_cancelled()
            self._access_record(record_id)
            if not self.config.graph or direction not in {"both", "incoming", "outgoing"}:
                raise ValueError("Enable graph tools and use incoming, outgoing, or both.")
            if offset < 0 or not 1 <= limit <= self.grant.graph_page_size:
                raise ValueError("Use a nonnegative offset and the view's graph limit.")
            edges: list[GraphEdge] = []
            for key in self.adjacency[record_id]:
                edge = self.snapshot.get(key)
                assert isinstance(edge, GraphEdge)
                if direction == "outgoing" and edge.source_id != record_id:
                    continue
                if direction == "incoming" and edge.target_id != record_id:
                    continue
                edges.append(edge)
            lines = [f"Relations for source {record_id}:"]
            for edge in edges[offset:offset + limit]:
                self._access_record(edge.id)
                source = self._access_record(edge.source_id)
                target = self._access_record(edge.target_id)
                lines.append(f"Source {edge.id}: {source.id} --{edge.label}--> {target.id}\n{edge.fact_text}")
            if offset + limit < len(edges):
                lines.append(f"More relations: memory_neighbors({record_id}, direction={direction}, offset={offset + limit}, limit={limit}).")
            return self._trace("neighbors", "\n".join(lines))

    def refresh(self) -> str:
        with self._lock:
            self.check_cancelled()
            if self.config.refresh_scope is not MemoryRefreshScope.AcceptedInWorld:
                raise ValueError("This workflow does not permit memory refresh.")
            with self.config.store.lock:
                grant = MemoryView(record_ids=frozenset(self.config.store.revisions),
                    scopes=frozenset({(-1, "in_world")}), frontier=(self.config.store.current_tick_id, 2**63 - 1))
                self._install(grant)
            self.accessed.clear()
            return self.view()

    @contextmanager
    def guard(self) -> Generator[None]:
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

    def __exit__(self, exception_type: type[BaseException] | None, exception: BaseException | None,
                 traceback: TracebackType | None) -> None:
        self.close()

    def server(self) -> FastMCP:
        server = FastMCP("memory")
        server.tool(name="memory_view")(self.view)
        server.tool(name="memory_search")(self.search)
        server.tool(name="memory_get")(self.get)
        server.tool(name="memory_sources")(self.sources)
        if self.config.graph:
            server.tool(name="memory_neighbors")(self.neighbors)
        if self.config.refresh_scope is not None:
            server.tool(name="memory_refresh")(self.refresh)
        return server


__all__ = ["MemoryView", "MemoryToolConfig", "MemoryReadSession", "guard_revisions"]
