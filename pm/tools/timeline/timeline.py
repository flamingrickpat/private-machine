"""Merge conversation, model calls, workflow phases, and GWT cycles by time.

One review attempt has several evidence streams. This module builds named
participant views, then projects every stream into one timeline event list. The
event's ``at`` field is a UTC sort key. ``clock`` says whether the source
recorded virtual time or only wall time, so the page does not pretend the two
are the same clock.

Owns: the merged attempt view and timeline ordering.
Does not own: any evidence source or the browser API.
See the package overview in ``pm.tools.timeline``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, JsonValue

from pm.tools.timeline.calls import CallIndex, index_calls
from pm.tools.timeline.evidence import (
    AttemptBundle,
    EvidencePaths,
    EvidenceStore,
    RunBundle,
    normalize_time,
)
from pm.tools.timeline.records import RecordIndex, read_agent_receipts, index_records
from pm.tools.timeline.workflows import KernelIndex, index_kernels

EventKind = Literal["speech", "actor", "delivery", "call", "embedding", "workflow", "phase", "cycle", "fact", "failure", "agent"]
ClockKind = Literal["virtual", "wall", "unknown"]


class TimelineTarget(BaseModel):
    """What the browser opens when the owner selects an event."""

    kind: Literal["none", "call", "record", "cycle", "workflow", "artifact", "agent", "failure"] = "none"
    participant: str = ""
    call_id: str = ""
    record_id: int | None = None
    tick: int | None = None
    invocation_id: str = ""
    path: str = ""


class TimelineEvent(BaseModel):
    """One row in the merged execution timeline."""

    id: str
    kind: EventKind
    participant: str = ""
    title: str
    preview: str = ""
    at: str = ""
    wall_at: str = ""
    clock: ClockKind = "unknown"
    severity: Literal["info", "warning", "error"] = "info"
    target: TimelineTarget = Field(default_factory=TimelineTarget)
    details: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])


class ParticipantOverview(BaseModel):
    """Counts and current state for one participant in an attempt."""

    id: str
    records: int = 0
    conversation: int = 0
    calls: int = 0
    chat_calls: int = 0
    embedding_calls: int = 0
    workflows: int = 0
    cycles: int = 0
    latest_cycle: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    errors: int = 0


@dataclass
class ParticipantView:
    """All read-only indexes for one participant attempt."""

    participant: str
    paths: EvidencePaths
    records: RecordIndex
    calls: CallIndex
    kernels: KernelIndex


@dataclass
class AttemptView:
    """One attempt with every participant index loaded once."""

    run: RunBundle
    attempt: AttemptBundle
    participants: dict[str, ParticipantView]

    @classmethod
    def load(cls, store: EvidenceStore, run_id: str, attempt: int) -> "AttemptView":
        """Load one attempt and every participant evidence folder it names."""
        run = store.run(run_id)
        bundle = store.attempt(run_id, attempt)
        participants: dict[str, ParticipantView] = {}
        for participant in bundle.summary.evidence_participants:
            paths = store.paths(run_id, attempt, participant)
            calls = index_calls(participant, paths.prompts, paths.evidence_root)
            participants[participant] = ParticipantView(
                participant=participant,
                paths=paths,
                records=index_records(participant, paths.evidence_root),
                calls=calls,
                kernels=index_kernels(participant, paths.kernels, paths.evidence_root, calls),
            )
        return cls(run=run, attempt=bundle, participants=participants)

    def participant(self, participant_id: str) -> ParticipantView:
        """Return one participant view or raise ``FileNotFoundError``."""
        view = self.participants.get(participant_id)
        if view is None:
            raise FileNotFoundError(f"Participant {participant_id} has no evidence in this attempt.")
        return view

    def overviews(self) -> list[ParticipantOverview]:
        """Return one count summary for each captured participant."""
        rows: list[ParticipantOverview] = []
        for participant in sorted(self.participants):
            view = self.participants[participant]
            calls = view.calls.all()
            chat_calls = [call for call in calls if call.route == "chat/completions"]
            embedding_calls = [call for call in calls if call.route == "embeddings"]
            cycles = view.records.cycles()
            rows.append(ParticipantOverview(
                id=participant,
                records=len(view.records.cards()),
                conversation=len(view.records.conversation),
                calls=len(calls),
                chat_calls=len(chat_calls),
                embedding_calls=len(embedding_calls),
                workflows=len(view.kernels.all()),
                cycles=len(cycles),
                latest_cycle=max((cycle.tick for cycle in cycles), default=0),
                prompt_tokens=sum(call.prompt_tokens or 0 for call in calls),
                completion_tokens=sum(call.completion_tokens or 0 for call in calls),
                total_tokens=sum(call.total_tokens or 0 for call in calls),
                errors=sum(1 for call in calls if call.error),
            ))
        return rows

    def timeline(self, participant: str = "", kinds: set[str] | None = None,
                 query: str = "", limit: int | None = None) -> list[TimelineEvent]:
        """Return the merged event stream, filtered and ordered by UTC instant.

        ``kinds`` is a set of event kind names. An empty set means all kinds.
        ``limit`` optionally bounds a caller's projection. The browser receives
        all summaries so its filters can reach the end of a long attempt.
        """
        events = _build_events(self)
        if participant:
            events = [event for event in events if not event.participant or event.participant == participant]
        if kinds:
            events = [event for event in events if event.kind in kinds]
        needle = query.strip().lower()
        if needle:
            events = [event for event in events
                      if needle in event.title.lower() or needle in event.preview.lower()
                      or needle in event.participant.lower() or needle in event.kind.lower()]
        return events[:limit]


def _event(*, event_id: str, kind: EventKind, title: str, participant: str = "", preview: str = "",
           at: str = "", wall_at: str = "", clock: ClockKind = "unknown",
           severity: Literal["info", "warning", "error"] = "info",
           target: TimelineTarget | None = None, details: dict[str, JsonValue] | None = None) -> TimelineEvent:
    """Build one event with explicit clock and target fields."""
    return TimelineEvent(
        id=event_id,
        kind=kind,
        participant=participant,
        title=title,
        preview=preview[:600],
        at=at,
        wall_at=wall_at,
        clock=clock,
        severity=severity,
        target=target or TimelineTarget(),
        details=details or {},
    )


def _event_time(value: JsonValue, fallback: str = "") -> tuple[str, str, ClockKind]:
    """Return ``(at, wall_at, clock)`` for an evidence timestamp.

    A caller that has a virtual and a wall value passes the virtual one as
    ``value`` and the wall one as ``fallback``. A caller with only wall time
    passes it as ``value`` and the page marks it as wall time.
    """
    at = normalize_time(value)
    wall = normalize_time(fallback)
    if at and fallback:
        return at, wall, "virtual" if normalize_time(value) != normalize_time(fallback) else "wall"
    if at:
        return at, "", "virtual"
    if wall:
        return wall, wall, "wall"
    return "", "", "unknown"


def _build_events(view: AttemptView) -> list[TimelineEvent]:
    """Build every event for one attempt. Ordering happens after construction."""
    events: list[TimelineEvent] = []
    events.extend(_speech_events(view))
    events.extend(_actor_events(view))
    events.extend(_delivery_events(view))
    events.extend(_call_events(view))
    events.extend(_workflow_events(view))
    events.extend(_cycle_events(view))
    events.extend(_fact_events(view))
    events.extend(_failure_events(view))
    events.extend(_agent_events(view))
    rank = {"actor": 0, "speech": 1, "delivery": 2, "call": 3, "embedding": 3, "workflow": 4, "phase": 5,
            "cycle": 6, "fact": 7, "failure": 8, "agent": 9}
    return sorted(events, key=lambda event: (event.at or "9999", event.wall_at or "0000", rank[event.kind], event.id))


def _speech_events(view: AttemptView) -> list[TimelineEvent]:
    """Public dialogue records, with their captured wall and virtual time."""
    rows: list[TimelineEvent] = []
    for participant in sorted(view.participants):
        records = view.participants[participant].records
        for row in records.conversation:
            at, wall, clock = _event_time(row.at, row.wall_at)
            rows.append(_event(
                event_id=f"speech:{participant}:{row.record_id}",
                kind="speech",
                participant=participant,
                title=f"{row.speaker} · {participant}",
                preview=row.text,
                at=at,
                wall_at=wall,
                clock=clock,
                target=TimelineTarget(kind="record", participant=participant, record_id=row.record_id),
                details={"record_id": row.record_id, "delivery_id": row.delivery_id,
                         "speaker": row.speaker, "feature_type": row.feature_type},
            ))
    return rows


def _actor_events(view: AttemptView) -> list[TimelineEvent]:
    """Scenario actor deliveries, including rows from failed attempts."""
    rows: list[TimelineEvent] = []
    for index, row in enumerate(view.attempt.run_timeline):
        kind = row.get("kind")
        if kind == "actor":
            at = normalize_time(row.get("timestamp"))
            target = str(row.get("target", ""))
            speaker = str(row.get("speaker", "Actor"))
            rows.append(_event(
                event_id=f"actor:{index}:{row.get('delivery_id', '')}",
                kind="actor",
                participant=target,
                title=f"{speaker} → {target or 'participant'}",
                preview=str(row.get("content", "")),
                at=at,
                wall_at="",
                clock="virtual" if at else "unknown",
                target=TimelineTarget(kind="none", participant=target),
                details={key: value for key, value in row.items()},
            ))
    return rows


def _delivery_events(view: AttemptView) -> list[TimelineEvent]:
    """Routing receipts from the controller transit ledger."""
    by_output: dict[str, str] = {}
    for participant in view.participants.values():
        for row in participant.records.conversation:
            by_output[str(row.record_id)] = row.at
    rows: list[TimelineEvent] = []
    state = view.run.state
    for index, row in enumerate(state.transit):
        output_id = str(row.get("output_id", ""))
        at = normalize_time(by_output.get(output_id, ""))
        source = str(row.get("source", ""))
        target = str(row.get("target", ""))
        status = str(row.get("status", ""))
        rows.append(_event(
            event_id=f"delivery:{index}:{row.get('delivery_id', '')}",
            kind="delivery",
            participant=target,
            title=f"{source} → {target} · {status}",
            preview=str(row.get("content", "")),
            at=at,
            wall_at="",
            clock="virtual" if at else "unknown",
            severity="warning" if status == "pending" else "info",
            target=TimelineTarget(kind="none", participant=target),
            details={key: value for key, value in row.items()},
        ))
    return rows


def _call_events(view: AttemptView) -> list[TimelineEvent]:
    """Every prompt-log call, with its invocation and usage."""
    rows: list[TimelineEvent] = []
    for participant in sorted(view.participants):
        calls = view.participants[participant].calls.all()
        for call in calls:
            at = normalize_time(call.world_time)
            wall = normalize_time(call.started_at)
            if at:
                clock: ClockKind = "virtual"
            elif wall:
                at, clock = wall, "wall"
            else:
                clock = "unknown"
            usage: list[str] = []
            if call.total_tokens is not None:
                usage.append(f"{call.total_tokens} tok")
            if call.elapsed_seconds is not None:
                usage.append(f"{call.elapsed_seconds:.1f}s")
            title = f"{call.module or call.route or 'model'} · {call.model or 'unknown model'}"
            preview = " · ".join([part for part in [call.route, *usage, call.preview] if part])
            rows.append(_event(
                event_id=f"call:{participant}:{call.call_id}",
                kind="embedding" if call.route == "embeddings" else "call",
                participant=participant,
                title=title,
                preview=preview,
                at=at,
                wall_at=wall,
                clock=clock,
                severity="error" if call.error else "info",
                target=TimelineTarget(kind="call", participant=participant, call_id=call.call_id),
                details={"call_id": call.call_id, "route": call.route, "module": call.module,
                         "invocation_id": call.invocation_id, "error": call.error},
            ))
    return rows


def _workflow_events(view: AttemptView) -> list[TimelineEvent]:
    """One start row and one row per kernel phase for every invocation."""
    rows: list[TimelineEvent] = []
    for participant in sorted(view.participants):
        for run in view.participants[participant].kernels.all():
            at = normalize_time(run.world_time)
            wall = normalize_time(run.started_at)
            clock: ClockKind = "virtual" if at else ("wall" if wall else "unknown")
            if not at:
                at = wall
            start = _event(
                event_id=f"workflow:{participant}:{run.invocation_id}",
                kind="workflow",
                participant=participant,
                title=f"{run.workflow} · {len(run.phases)} phases",
                preview=f"{run.status or 'unknown status'} · {run.call_count} calls · {run.total_tokens} tok",
                at=at,
                wall_at=wall,
                clock=clock,
                severity="warning" if any(phase.ok is False for phase in run.phases) else "info",
                target=TimelineTarget(kind="workflow", participant=participant, invocation_id=run.invocation_id),
                details={"invocation_id": run.invocation_id, "workflow": run.workflow,
                         "status": run.status, "phase_count": len(run.phases)},
            )
            rows.append(start)
            for index, phase in enumerate(run.phases):
                rows.append(_event(
                    event_id=f"phase:{participant}:{run.invocation_id}:{index}",
                    kind="phase",
                    participant=participant,
                    title=f"{phase.phase} · {phase.role or phase.kind or 'phase'}",
                    preview=phase.summary or phase.verdict or phase.status,
                    at=at,
                    wall_at=wall,
                    clock=clock,
                    severity="warning" if phase.ok is False else "info",
                    target=TimelineTarget(kind="workflow", participant=participant,
                                          invocation_id=run.invocation_id),
                    details={"phase": phase.phase, "kind": phase.kind, "role": phase.role,
                             "verdict": phase.verdict, "status": phase.status},
                ))
    return rows


def _cycle_events(view: AttemptView) -> list[TimelineEvent]:
    """One row per captured GWT cycle."""
    rows: list[TimelineEvent] = []
    for participant in sorted(view.participants):
        records = view.participants[participant].records
        for cycle in records.cycles():
            rows.append(_event(
                event_id=f"cycle:{participant}:{cycle.tick}",
                kind="cycle",
                participant=participant,
                title=f"Cycle {cycle.tick}",
                preview=cycle.summary,
                at=normalize_time(cycle.at),
                wall_at=normalize_time(cycle.wall_at),
                clock="virtual" if cycle.at else "unknown",
                target=TimelineTarget(kind="cycle", participant=participant, tick=cycle.tick),
                details={"tick": cycle.tick, "slot_id": cycle.slot_id,
                         "members": cycle.member_count, "stimuli": cycle.stimulus_count,
                         "codelet_runs": cycle.codelet_runs},
            ))
    return rows


def _fact_events(view: AttemptView) -> list[TimelineEvent]:
    """Mandatory facts with their recorded delivery reason."""
    step_times: dict[str, str] = {}
    for step in view.run.plan.steps:
        for fact_id in step.fact_ids:
            step_times[fact_id] = step.at
    rows: list[TimelineEvent] = []
    for fact_id, value in sorted(view.attempt.facts.items()):
        data = value if isinstance(value, dict) else {}
        at = normalize_time(step_times.get(fact_id, "")) or normalize_time(view.run.state.clock)
        rows.append(_event(
            event_id=f"fact:{fact_id}",
            kind="fact",
            title=f"Fact {fact_id}",
            preview=str(data.get("reason", "")),
            at=at,
            clock="virtual" if at else "unknown",
            severity="info",
            details={"fact_id": fact_id, "delivery_ids": data.get("delivery_ids", [])},
        ))
    return rows


def _failure_events(view: AttemptView) -> list[TimelineEvent]:
    """Controller and attempt failures, kept visible instead of summarized away."""
    rows: list[TimelineEvent] = []
    seen: set[str] = set()
    for index, failure in enumerate(view.attempt.run_failures):
        text = str(failure.get("error") or failure.get("cleanup_error") or failure)
        if text in seen:
            continue
        seen.add(text)
        at = normalize_time(view.run.state.clock)
        rows.append(_event(
            event_id=f"failure:{index}",
            kind="failure",
            title=f"Attempt {failure.get('attempt', '?')} failure",
            preview=text,
            at=at,
            clock="virtual" if at else "unknown",
            severity="error",
            details={key: value for key, value in failure.items()},
        ))
    return rows


def _agent_events(view: AttemptView) -> list[TimelineEvent]:
    """Harness actor, reviewer, and director calls, labelled separately."""
    rows: list[TimelineEvent] = []
    if not view.attempt.summary.has_agents or not view.participants:
        return rows
    first_participant = next(iter(view.participants.values()))
    receipts = read_agent_receipts(first_participant.paths.attempt_root)
    for index, receipt in enumerate(receipts):
        raw_request = receipt.get("request")
        request = raw_request if isinstance(raw_request, dict) else {}
        role = str(request.get("role", "agent"))
        status = str(receipt.get("status", ""))
        output = str(receipt.get("output", ""))
        result = receipt.get("result")
        preview = output or (str(result) if result is not None else "")
        mtime = receipt.get("_mtime")
        wall = ""
        if isinstance(mtime, (int, float)):
            wall = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(timespec="seconds")
        rows.append(_event(
            event_id=f"agent:{index}:{receipt.get('id', '')}",
            kind="agent",
            title=f"Harness {role} · {status}",
            preview=preview,
            at=wall,
            wall_at=wall,
            clock="wall" if wall else "unknown",
            severity="warning" if status != "completed" else "info",
            target=TimelineTarget(kind="agent", path=str(receipt.get("_path", ""))),
            details={"status": status, "role": role, "path": receipt.get("_path", "")},
        ))
    return rows


__all__ = ["AttemptView", "ParticipantOverview", "ParticipantView", "TimelineEvent", "TimelineTarget"]
