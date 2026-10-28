"""Capture memory views supplied to one workflow invocation.

Collection reads retain their full view, including absent query results.
Capture stops before host publication so validation cannot add unseen inputs.
"""
from __future__ import annotations

from contextvars import ContextVar


_current: ContextVar[ReadScope | None] = ContextVar("pm_memory_reads", default=None)


class ReadScope:
    """Collect conservative input manifests across synchronous and async calls."""

    def __init__(self):
        self.collecting = True
        self.views: dict[int, tuple[set[int], set[str]]] = {}
        self.completed_invocation_id: str | None = None

    def __enter__(self):
        self._token = _current.set(self)
        return self

    def __exit__(self, *errors):
        _current.reset(self._token)

    def observe(self, store, record_ids, commit_ids=()) -> None:
        """Retain an input view under its owning store identity."""
        if self.collecting:
            ids, commits = self.views.setdefault(id(store), (set(), set()))
            ids.update(record_ids)
            commits.update(commit_ids)

    def apply(self, store, batch) -> None:
        """Attach this invocation's captured inputs at the atomic write boundary."""
        if id(store) in self.views:
            ids, commits = self.views[id(store)]
            batch.depend_on(ids)
            batch.depend_on_commits(commits)
            batch._manifest["memory_capture"] = "conservative_collection_views"


def current() -> ReadScope | None:
    """Return the current invocation scope, if present."""
    return _current.get()
