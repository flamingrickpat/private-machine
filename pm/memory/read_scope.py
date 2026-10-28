"""Capture memory views supplied to one workflow invocation.

Collection reads retain their full view, including absent query results.
Capture stops before host publication so validation cannot add unseen inputs.
"""
from __future__ import annotations

from collections.abc import Iterable
from contextvars import ContextVar
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pm.memory.store import MemoryStore
    from pm.memory.write_batch import WriteBatch


_current: ContextVar[ReadScope | None] = ContextVar("pm_memory_reads", default=None)


class ReadScope:
    """Collect conservative input manifests across synchronous and async calls.

    A store read that happens inside this scope becomes an input of the commit
    that later takes the lock. The scope over-collects on purpose: an extra
    dependency only rejects a commit that would otherwise race a writer.
    """

    def __init__(self) -> None:
        self.collecting = True
        self.views: dict[int, tuple[set[int], set[str]]] = {}
        self.completed_invocation_id: str | None = None

    def __enter__(self) -> ReadScope:
        self._token = _current.set(self)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None,
                 traceback: TracebackType | None) -> None:
        _current.reset(self._token)

    def observe(self, store: MemoryStore, record_ids: Iterable[int],
                commit_ids: Iterable[str] = ()) -> None:
        """Retain an input view under its owning store identity."""
        if self.collecting:
            ids, commits = self.views.setdefault(id(store), (set(), set()))
            ids.update(record_ids)
            commits.update(commit_ids)

    def apply(self, store: MemoryStore, batch: WriteBatch) -> None:
        """Attach this invocation's captured inputs at the atomic write boundary."""
        if id(store) in self.views:
            ids, commits = self.views[id(store)]
            batch.depend_on(ids)
            batch.depend_on_commits(commits)
            batch._manifest["memory_capture"] = "conservative_collection_views"


def current() -> ReadScope | None:
    """Return the current invocation scope, if present."""
    return _current.get()
