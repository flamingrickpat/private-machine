"""Read process spans and existing model receipts on a shared wall clock."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from pm.host.cognition_profiler import ProfileEvent, SpanEvidence
from pm.tools.timeline.timeline import ParticipantView


class ProfileBar(BaseModel):
    """A selectable interval and its evidence identity on one participant lane."""

    participant: str
    lane: str
    operation: str
    start: datetime
    end: datetime | None
    outcome: str
    kind: Literal["workflow", "call"]
    target_id: str
    invocation_id: str
    inputs: list[str]
    prompt_hash: str = ""
    evidence: SpanEvidence | None = None
    span_id: str = ""
    elapsed_seconds: float | None = None
    parent_span_id: str = ""


def profile_bars(view: ParticipantView) -> list[ProfileBar]:
    bars: list[ProfileBar] = []
    root = view.paths.evidence_root / "runtime" / "profile"
    pid = view.paths.process_id
    pattern = f"spans-{pid}.jsonl" if pid is not None else "spans-*.jsonl"
    for path in sorted(root.glob(pattern)):
        bars.extend(read_workflow_spans(view.participant, path))
    by_invocation: dict[str, list[ProfileBar]] = {}  # pm-lint: allow PM004 -- spans grouped by kernel invocation
    for bar in bars:
        by_invocation.setdefault(bar.invocation_id, []).append(bar)
    for call in view.calls.all():
        if not call.started_at:
            continue
        start = datetime.fromisoformat(call.started_at)
        end = None
        if call.elapsed_seconds is not None:
            end = start + timedelta(seconds=call.elapsed_seconds)
        lane = f"{call.process_id}/{call.owner_thread}/{call.thread_id}"
        # An interval on the same native thread can be linked to its innermost
        # enclosing operation. Other threads retain only the known invocation.
        parent: ProfileBar | None = None
        for span in by_invocation.get(call.invocation_id, []):
            if span.lane != lane or span.end is None or end is None:
                continue
            if span.start <= start and end <= span.end:
                if parent is None or span.start > parent.start:
                    parent = span
        bars.append(ProfileBar(
            participant=view.participant, lane=lane, parent_span_id=parent.span_id if parent else "",
            operation=call.module or call.route, start=start, end=end, kind="call",
            target_id=call.call_id, invocation_id=call.invocation_id, inputs=[], prompt_hash=call.prompt_hash,
            elapsed_seconds=call.elapsed_seconds,
            outcome=call.error or ("finished" if end is not None else "end unavailable"),
        ))
    return bars


def read_workflow_spans(participant: str, path: Path) -> list[ProfileBar]:
    process_id = path.stem.removeprefix("spans-")
    spans: dict[str, ProfileBar] = {}  # pm-lint: allow PM004 -- spans indexed by recorded span ID
    start_ns: dict[str, int] = {}  # pm-lint: allow PM004 -- monotonic time by span ID
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break  # An active writer can still be appending this event.
            event = ProfileEvent.model_validate_json(line)
            if event.outcome == "started":
                start_ns[event.span_id] = event.monotonic_ns
                parent = spans.get(event.evidence.parent)
                invocation = parent.invocation_id if parent is not None else event.invocation_id
                spans[event.span_id] = ProfileBar(
                    participant=participant, lane=f"{process_id}/{event.thread_name}/{event.thread_id}",
                    operation=event.operation, start=event.wall_at, end=None, outcome="open",
                    kind="workflow", target_id=invocation, invocation_id=invocation, parent_span_id=event.evidence.parent,
                    inputs=event.input_envelope_ids, evidence=event.evidence, span_id=event.span_id,
                )
            else:
                span = spans[event.span_id]
                span.evidence = event.evidence
                span.end = event.wall_at
                span.elapsed_seconds = (event.monotonic_ns - start_ns.pop(event.span_id)) / 1_000_000_000
                span.outcome = event.outcome
    return list(spans.values())
