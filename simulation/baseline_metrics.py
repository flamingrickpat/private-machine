"""Measure visible waiting time and provider-reported usage from saved run receipts."""
from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path

from pydantic import BaseModel
from pm.host.cognition_profiler import ProfileEvent, SpanKind


class PublicEvent(BaseModel):
    kind: str
    target: str = ""
    source: str = ""
    wall_at: datetime | None = None


class ConversationReceipt(BaseModel):
    timeline: list[PublicEvent]


class EndpointReceipt(BaseModel):
    pid: int


class ProviderUsage(BaseModel):
    cost: float | None = None


class UsageReceipt(BaseModel):
    call_id: str
    event: str
    usage: ProviderUsage | None = None


class BaselineMetrics(BaseModel):
    """A wait starts with the first actor input since that participant last spoke."""

    visible_exchanges: int
    pending_participants: int
    untimed_public_events: int
    p95_wait_seconds: float | None
    worst_wait_seconds: float | None
    participant_model_calls: int
    participant_calls_without_cost: int
    reported_participant_cost_usd: float
    minecraft_tool_results: int
    minecraft_tool_failures: int
    minecraft_open_spans: int
    p95_minecraft_tool_seconds: float | None
    worst_minecraft_tool_seconds: float | None


def measure_run(root: Path) -> BaselineMetrics:
    receipt = ConversationReceipt.model_validate_json((root / "state.json").read_bytes())
    waiting: dict[str, datetime] = {}  # pm-lint: allow PM004 -- pending wait by participant ID
    waits: list[float] = []
    untimed = 0
    for event in receipt.timeline:
        if event.kind not in {"actor", "speech"}:
            continue
        if event.wall_at is None:
            untimed += 1
            waiting.pop(event.target if event.kind == "actor" else event.source, None)
        elif event.kind == "actor":
            waiting.setdefault(event.target, event.wall_at)
        elif event.source in waiting:
            waits.append((event.wall_at - waiting.pop(event.source)).total_seconds())
    waits.sort()
    costs: dict[str, float | None] = {}  # pm-lint: allow PM004 -- provider cost by call ID
    tool_starts: dict[str, int] = {}  # pm-lint: allow PM004 -- start times by Minecraft span ID
    tool_times: list[float] = []
    tool_failures = 0
    processed: set[Path] = set()
    for endpoint in sorted(root.glob("attempts/*/*-endpoint.json")):
        pid = EndpointReceipt.model_validate_json(endpoint.read_bytes()).pid
        participant = endpoint.name.removesuffix("-endpoint.json")
        filename = f"usage-{pid}.jsonl"
        home = root / "participants" / participant
        ledger = home / "logs/prompts" / filename
        if ledger in processed:
            continue
        processed.add(ledger)
        trace = home / "runtime/profile" / f"spans-{pid}.jsonl"
        if trace.is_file():
            with trace.open("rb") as stream:
                for line in stream:
                    if not line.endswith(b"\n"):
                        break
                    span = ProfileEvent.model_validate_json(line)
                    if span.evidence.kind != SpanKind.Minecraft:
                        continue
                    if span.outcome == "started":
                        tool_starts[span.span_id] = span.monotonic_ns
                    else:
                        elapsed_ns = span.monotonic_ns - tool_starts.pop(span.span_id)
                        tool_times.append(elapsed_ns / 1_000_000_000)
                        if span.outcome != "finished" or span.evidence.tool_succeeded is False:
                            tool_failures += 1
        if not ledger.is_file():
            continue
        with ledger.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                usage = UsageReceipt.model_validate_json(line)
                costs.setdefault(usage.call_id, None)
                if usage.usage is not None and usage.usage.cost is not None:
                    costs[usage.call_id] = usage.usage.cost
    tool_times.sort()
    return BaselineMetrics(
        visible_exchanges=len(waits), pending_participants=len(waiting.keys()), untimed_public_events=untimed,
        p95_wait_seconds=waits[math.ceil(len(waits) * 0.95) - 1] if waits else None,
        worst_wait_seconds=waits[-1] if waits else None, participant_model_calls=len(costs.keys()),
        participant_calls_without_cost=sum(value is None for value in costs.values()),
        reported_participant_cost_usd=math.fsum(value for value in costs.values() if value is not None),
        minecraft_tool_results=len(tool_times), minecraft_tool_failures=tool_failures,
        minecraft_open_spans=len(tool_starts.keys()),
        p95_minecraft_tool_seconds=tool_times[math.ceil(len(tool_times) * 0.95) - 1] if tool_times else None,
        worst_minecraft_tool_seconds=tool_times[-1] if tool_times else None,
    )
