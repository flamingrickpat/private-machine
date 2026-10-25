"""The JSON endpoints that the inspector page calls.

Each endpoint reads a prepared index. None of them runs cognition, calls a
model, or saves the store, so a page refresh is always safe on a character's
own checkpoint.

``InspectorApi.get`` returns a ``(status, payload)`` pair. The server turns
that pair into a response. Errors carry a short message that names what the
reader can do next. The full traceback stays in the terminal and in
:attr:`InspectorApi.errors`, because the traceback is for the person who
started the command, not for the page.
"""
from __future__ import annotations

import traceback

from pm.tools.inspector.model import MemoryIndex
from pm.tools.inspector.rendering import StoryPrompts, StoryRenderer

#: The most rows one page returns. The page asks for less, never more.
MAX_PAGE = 500


def _int(query: dict, name: str, default: int) -> int:
    """Return one integer query value. Raises ``ValueError`` when it is not a number."""
    values = query.get(name)
    if not values or values[0] == "":
        return default
    return int(values[0])


def _text(query: dict, name: str, default: str = "") -> str:
    """Return one optional text query value."""
    values = query.get(name)
    return values[0] if values else default


def _page(query: dict, default_limit: int) -> tuple[int, int]:
    """Return the ``(offset, limit)`` of one list request, with the limit capped."""
    return _int(query, "offset", 0), min(_int(query, "limit", default_limit), MAX_PAGE)


class InspectorApi:
    """Answer inspector requests from one loaded checkpoint.

    The index and the renderer are built once, before the server starts. Every
    later request reads those objects. That keeps a page load cheap on a
    checkpoint of thousands of records, and it keeps the store read-only.
    """

    def __init__(self, index: MemoryIndex, renderer: StoryRenderer | None = None) -> None:
        self.index = index
        self.prompts = StoryPrompts()
        self.renderer = renderer or StoryRenderer(index, self.prompts)
        self.errors: list[dict] = []

    def get(self, path: str, query: dict) -> tuple[int, object]:
        """Return the ``(status, payload)`` for one API path.

        ``path`` is the part after ``/api``. ``query`` is the parsed query
        string. A bad number or an unknown view gives ``400``, an unknown id
        gives ``404``, and a failing render gives ``500`` with its message.
        """
        parts = [part for part in path.split("/") if part]
        try:
            return self._route(parts, query)
        except (KeyError, ValueError, IndexError) as error:
            status = 404 if isinstance(error, KeyError) else 400
            return status, {"error": f"{type(error).__name__}: {error}"}
        except Exception as error:
            self.errors.append({"path": path, "error": f"{type(error).__name__}: {error}",
                                "trace": traceback.format_exc()})
            return 500, {"error": f"{type(error).__name__}: {error}"}

    def _route(self, parts: list[str], query: dict) -> tuple[int, object]:
        """Send one path to its reader."""
        if not parts:
            return 400, {"error": "no endpoint given"}
        name, rest = parts[0], parts[1:]
        if name == "summary" and not rest:
            return 200, self.index.summary()
        if name == "counts" and not rest:
            kinds, pairs = self.index.counts()
            return 200, {"kinds": [row.model_dump() for row in kinds],
                         "subtypes": [row.model_dump() for row in pairs[:120]]}
        if name == "tree" and not rest:
            return 200, {"tree": self.index.tree()}
        if name == "census" and not rest:
            return 200, {"edges": [row.model_dump() for row in self.index.relation_census()]}
        if name == "prompts" and not rest:
            return 200, self.prompts.describe()
        if name == "errors" and not rest:
            return 200, {"errors": self.errors[-20:]}
        if name == "records" and not rest:
            return 200, self._records(query)
        if name == "clusters" and not rest:
            offset, limit = _page(query, 300)
            rows = self.index.clusters()
            return 200, {"total": len(rows), "offset": offset, "items": rows[offset:offset + limit]}
        if name == "timeline" and not rest:
            offset, limit = _page(query, 120)
            rows = self.index.timeline(slot=_text(query, "slot"), limit=limit, offset=offset)
            return 200, {"total": len(self.index.tick_ids()), "offset": offset,
                         "items": [row.model_dump() for row in rows]}
        if name == "record" and rest:
            return self._record(int(rest[0]), rest[1:])
        if name == "ticks" and rest:
            return self._tick(int(rest[0]), rest[1:])
        return 400, {"error": f"unknown endpoint: /{'/'.join(parts)}"}

    def _records(self, query: dict) -> dict:
        """Return one page of the record list, filtered by text, kind, and subtype."""
        offset, limit = _page(query, 100)
        ids = self.index.matching(text=_text(query, "text"), kind=_text(query, "kind"),
                                  subtype=_text(query, "subtype"))
        return {"total": len(ids), "offset": offset,
                "items": [row.model_dump() for row in self.index.labels(ids[offset:offset + limit])]}

    def _record(self, record_id: int, rest: list[str]) -> tuple[int, object]:
        """Return one record: its fields, its render, or its shard history."""
        if not rest:
            return 200, self.index.detail(record_id).model_dump()
        if rest[0] == "render":
            return 200, self.renderer.record(record_id).model_dump()
        if rest[0] == "shards":
            return 200, {"hits": [hit.model_dump() for hit in self.index.shard_hits(record_id)[:MAX_PAGE]]}
        return 400, {"error": f"unknown record view: {rest[0]}"}

    def _tick(self, tick_id: int, rest: list[str]) -> tuple[int, object]:
        """Return one cycle: the full render, its GWT turn, its trace, or its story."""
        if not rest:
            return 200, self.renderer.tick(tick_id).model_dump()
        if rest[0] == "turn":
            return 200, self.index.turns[tick_id].model_dump(mode="json")
        if rest[0] == "trace":
            turn = self.index.turns.get(tick_id)
            trace = turn.trace if turn is not None else None
            return 200, trace.model_dump(mode="json") if trace is not None else {"trace": None}
        if rest[0] == "story":
            return 200, self.renderer.story(tick_id).model_dump()
        return 400, {"error": f"unknown cycle view: {rest[0]}"}


__all__ = ["MAX_PAGE", "InspectorApi"]
