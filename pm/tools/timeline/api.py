"""JSON endpoints for the read-only K003 execution timeline.

Every endpoint reads evidence that a completed simulation attempt already
wrote. The API keeps one ``AttemptView`` per selected run and attempt, so moving
between the timeline, calls, workflows, and cycles does not rebuild the same
indexes. No endpoint changes a file or starts a process.

Owns: HTTP-shaped projections of the attempt view.
Does not own: evidence parsing, timeline ordering, or record presentation.
See the package overview in ``pm.tools.timeline``.
"""
from __future__ import annotations

import json
import threading
import traceback
from typing import Literal

from pydantic import BaseModel, Field, JsonValue

from pm.tools.timeline.evidence import AttemptSummary, EvidenceError, EvidenceStore, RunSummary
from pm.tools.timeline.calls import CallSummary
from pm.tools.timeline.records import ConversationRow, CycleSummary
from pm.tools.timeline.timeline import AttemptView, ParticipantOverview, TimelineEvent

def _text(query: dict[str, list[str]], name: str, default: str = "") -> str:
    """Return one optional query value."""
    values = query.get(name)
    return values[0] if values else default


def _int(query: dict[str, list[str]], name: str, default: int) -> int:
    """Return one integer query value or raise ``ValueError``."""
    value = _text(query, name)
    if not value:
        return default
    return int(value)


class SearchHit(BaseModel):
    """One cross-evidence search result."""

    kind: Literal["speech", "call", "record", "workflow", "cycle", "agent"]
    participant: str = ""
    title: str
    preview: str = ""
    at: str = ""
    target: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])


class TimelinePage(BaseModel):
    """The filtered timeline and its unfiltered kind counts."""

    events: list[TimelineEvent] = Field(default_factory=list[TimelineEvent])
    counts: dict[str, int] = Field(default_factory=dict[str, int])
    total: int = 0


class AttemptPage(BaseModel):
    """Overview data for one attempt."""

    run: RunSummary
    attempt: AttemptSummary
    plan: JsonValue = None
    overviews: list[ParticipantOverview] = Field(default_factory=list[ParticipantOverview])
    report: JsonValue = None
    failures: list[dict[str, JsonValue]] = Field(default_factory=list[dict[str, JsonValue]])
    facts: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])


class TimelineApi:
    """Answer review requests from one evidence root."""

    def __init__(self, store: EvidenceStore) -> None:
        self.store: EvidenceStore = store
        self._lock: threading.RLock = threading.RLock()
        self._views: dict[tuple[str, int], AttemptView] = {}
        self.errors: list[dict[str, str]] = []

    def get(self, path: str, query: dict[str, list[str]]) -> tuple[int, object]:
        """Return ``(status, payload)`` for one API path."""
        parts = [part for part in path.split("/") if part]
        try:
            return self._route(parts, query)
        except (EvidenceError, KeyError) as error:
            return 404, {"error": f"{type(error).__name__}: {error}"}
        except (ValueError, FileNotFoundError) as error:
            return 400, {"error": f"{type(error).__name__}: {error}"}
        except Exception as error:  # A bug stays visible in the terminal and the page.
            self.errors.append({"path": path, "error": f"{type(error).__name__}: {error}",
                                "trace": traceback.format_exc()})
            return 500, {"error": f"{type(error).__name__}: {error}"}

    def _route(self, parts: list[str], query: dict[str, list[str]]) -> tuple[int, object]:
        if not parts:
            return 400, {"error": "no endpoint given"}
        name, rest = parts[0], parts[1:]
        if name == "runs" and not rest:
            return 200, [row.model_dump(mode="json") for row in self.store.list_runs()]
        if name == "run" and len(rest) == 1:
            run = self.store.run(rest[0])
            return 200, run.model_dump(mode="json")
        if name == "attempt" and not rest:
            return 200, self._attempt(query).model_dump(mode="json")
        if name == "timeline" and not rest:
            return 200, self._timeline(query).model_dump(mode="json")
        if name == "chat" and not rest:
            return 200, self._chat(query)
        if name == "calls" and not rest:
            return 200, self._calls(query)
        if name == "call" and not rest:
            return 200, self._call(query)
        if name == "workflows" and not rest:
            return 200, self._workflows(query)
        if name == "workflow" and not rest:
            return 200, self._workflow(query)
        if name == "cycles" and not rest:
            return 200, self._cycles(query)
        if name == "cycle" and not rest:
            return 200, self._cycle(query)
        if name == "records" and not rest:
            return 200, self._records(query)
        if name == "record" and not rest:
            return 200, self._record(query)
        if name == "artifact" and not rest:
            return 200, self._artifact(query)
        if name == "agent" and not rest:
            return 200, self._agent(query)
        if name == "search" and not rest:
            return 200, self._search(query)
        return 404, {"error": f"unknown endpoint: /{name}"}

    def _view(self, query: dict[str, list[str]]) -> AttemptView:
        """Return the selected attempt view, using the latest attempt by default."""
        run_id = _text(query, "run")
        if not run_id:
            raise ValueError("This request needs a run id.")
        run = self.store.run(run_id)
        attempt = _int(query, "attempt", run.summary.latest_attempt or 0)
        if attempt <= 0:
            raise ValueError(f"Run {run_id} has no attempt to review.")
        key = (run_id, attempt)
        with self._lock:
            view = self._views.get(key)
            if view is None:
                view = AttemptView.load(self.store, run_id, attempt)
                self._views[key] = view
            return view

    def _attempt(self, query: dict[str, list[str]]) -> AttemptPage:
        view = self._view(query)
        return AttemptPage(
            run=view.run.summary,
            attempt=view.attempt.summary,
            plan=view.run.plan.model_dump(mode="json"),
            overviews=view.overviews(),
            report=view.attempt.report,
            failures=view.attempt.run_failures,
            facts=view.attempt.facts,
        )

    def _timeline(self, query: dict[str, list[str]]) -> TimelinePage:
        view = self._view(query)
        participant = _text(query, "participant")
        kinds = {item for item in _text(query, "kinds").split(",") if item}
        events = view.timeline(participant=participant, kinds=kinds, query=_text(query, "q"))
        all_events = view.timeline(participant=participant)
        counts: dict[str, int] = {}
        for event in all_events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        return TimelinePage(events=events, counts=counts, total=len(all_events))

    def _chat(self, query: dict[str, list[str]]) -> list[dict[str, JsonValue]]:
        view = self._view(query)
        participant = _text(query, "participant")
        rows: list[ConversationRow] = []
        for pid in ([participant] if participant else sorted(view.participants)):
            rows.extend(view.participant(pid).records.conversation)
        rows.sort(key=lambda row: (row.at, row.wall_at, row.record_id))
        return [row.model_dump(mode="json") for row in rows]

    def _calls(self, query: dict[str, list[str]]) -> list[dict[str, JsonValue]]:
        view = self._view(query)
        participant = _text(query, "participant")
        route = _text(query, "route")
        needle = _text(query, "q").strip().lower()
        rows: list[CallSummary] = []
        for pid in ([participant] if participant else sorted(view.participants)):
            for call in view.participant(pid).calls.summaries():
                if route and call.route != route:
                    continue
                haystack = f"{call.call_id} {call.module} {call.model} {call.route} {call.preview}".lower()
                if needle and needle not in haystack:
                    continue
                rows.append(call)
        rows.sort(key=lambda row: (row.world_time, row.started_at, row.call_id))
        return [row.model_dump(mode="json") for row in rows]

    def _call(self, query: dict[str, list[str]]) -> dict[str, JsonValue]:
        view = self._view(query)
        participant = _text(query, "participant")
        if not participant:
            raise ValueError("A call request needs a participant.")
        call_id = _text(query, "call_id")
        if not call_id:
            raise ValueError("A call request needs a call_id.")
        return view.participant(participant).calls.get(call_id).model_dump(mode="json")

    def _workflows(self, query: dict[str, list[str]]) -> list[dict[str, JsonValue]]:
        view = self._view(query)
        participant = _text(query, "participant")
        rows: list[dict[str, JsonValue]] = []
        for pid in ([participant] if participant else sorted(view.participants)):
            rows.extend(row.model_dump(mode="json") for row in view.participant(pid).kernels.rows())
        return rows

    def _workflow(self, query: dict[str, list[str]]) -> dict[str, JsonValue]:
        view = self._view(query)
        participant = _text(query, "participant")
        if not participant:
            raise ValueError("A workflow request needs a participant.")
        invocation_id = _text(query, "invocation")
        if not invocation_id:
            raise ValueError("A workflow request needs an invocation id.")
        return view.participant(participant).kernels.get(invocation_id).model_dump(mode="json")

    def _cycles(self, query: dict[str, list[str]]) -> list[dict[str, JsonValue]]:
        view = self._view(query)
        participant = _text(query, "participant")
        rows: list[CycleSummary] = []
        for pid in ([participant] if participant else sorted(view.participants)):
            rows.extend(view.participant(pid).records.cycles())
        rows.sort(key=lambda row: (row.at, row.participant, row.tick))
        return [row.model_dump(mode="json") for row in rows]

    def _cycle(self, query: dict[str, list[str]]) -> dict[str, JsonValue]:
        view = self._view(query)
        participant = _text(query, "participant")
        if not participant:
            raise ValueError("A cycle request needs a participant.")
        tick = _int(query, "tick", 0)
        if tick <= 0:
            raise ValueError("A cycle request needs a tick.")
        return view.participant(participant).records.cycle(tick).model_dump(mode="json")

    def _records(self, query: dict[str, list[str]]) -> list[dict[str, JsonValue]]:
        view = self._view(query)
        participant = _text(query, "participant")
        if not participant:
            raise ValueError("A records request needs a participant.")
        records = view.participant(participant).records
        needle = _text(query, "q")
        role = _text(query, "role")
        rows = records.search_summaries(needle, limit=len(records.cards())) if needle else records.summaries()
        if role:
            rows = [row for row in rows if row.role == role]
        return [row.model_dump(mode="json") for row in rows]

    def _record(self, query: dict[str, list[str]]) -> dict[str, JsonValue]:
        view = self._view(query)
        participant = _text(query, "participant")
        if not participant:
            raise ValueError("A record request needs a participant.")
        record_id = _int(query, "id", 0)
        if record_id <= 0:
            raise ValueError("A record request needs a record id.")
        records = view.participant(participant).records
        return {"card": records.card(record_id).model_dump(mode="json"), "record": records.raw(record_id)}

    def _artifact(self, query: dict[str, list[str]]) -> dict[str, JsonValue]:
        view = self._view(query)
        participant = _text(query, "participant")
        if not participant:
            raise ValueError("An artifact request needs a participant.")
        path = _text(query, "path")
        if not path:
            raise ValueError("An artifact request needs a path.")
        participant_view = view.participant(participant)
        text, size = self.store.resolve_text(participant_view.paths.evidence_root, path)
        return {"path": path, "size": size, "text": text}

    def _agent(self, query: dict[str, list[str]]) -> dict[str, JsonValue]:
        view = self._view(query)
        if not view.participants:
            raise ValueError("This attempt has no participant evidence.")
        path = _text(query, "path")
        if not path:
            raise ValueError("An agent request needs a path.")
        root = next(iter(view.participants.values())).paths.attempt_root
        text, _ = self.store.resolve_text(root, path)
        return {"path": path, "receipt": json.loads(text)}

    def _search(self, query: dict[str, list[str]]) -> list[dict[str, JsonValue]]:
        view = self._view(query)
        needle = _text(query, "q").strip().lower()
        if not needle:
            return []
        hits: list[SearchHit] = []
        participant = _text(query, "participant")
        for pid in ([participant] if participant else sorted(view.participants)):
            member = view.participant(pid)
            for row in member.records.conversation:
                if needle in row.text.lower() or needle in row.speaker.lower():
                    hits.append(SearchHit(kind="speech", participant=pid, title=f"{row.speaker} · {pid}",
                                          preview=row.text, at=row.at,
                                          target={"kind": "record", "participant": pid, "record_id": row.record_id}))
            for call in member.calls.summaries():
                haystack = f"{call.call_id} {call.module} {call.model} {call.route} {call.preview}".lower()
                if needle in haystack:
                    hits.append(SearchHit(kind="call", participant=pid, title=call.route or "model call",
                                          preview=call.preview, at=call.world_time,
                                          target={"kind": "call", "participant": pid, "call_id": call.call_id}))
            for card in member.records.search(needle):
                hits.append(SearchHit(kind="record", participant=pid, title=f"{card.role} · {card.id}",
                                      preview=card.text, at="",
                                      target={"kind": "record", "participant": pid, "record_id": card.id}))
            for run in member.kernels.rows():
                haystack = f"{run.invocation_id} {run.workflow} {run.status}".lower()
                if needle in haystack:
                    hits.append(SearchHit(kind="workflow", participant=pid, title=f"{run.workflow} · {run.invocation_id[:8]}",
                                          preview=f"{run.status} · {run.phase_count} phases", at=run.world_time,
                                          target={"kind": "workflow", "participant": pid, "invocation_id": run.invocation_id}))
        return [row.model_dump(mode="json") for row in hits[:500]]


__all__ = ["AttemptPage", "SearchHit", "TimelineApi", "TimelinePage"]
