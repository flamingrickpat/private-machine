"""Persist workflow timing and causal input identities beside a participant's runtime."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
import os
import threading
import time
from uuid import uuid4

from pydantic import BaseModel, Field

from pm.bus.envelopes import Envelope


class SpanKind(StrEnum):
    Workflow = "workflow"
    Phase = "phase"
    Codelet = "codelet"
    Minecraft = "minecraft"
    Commit = "commit"


class SpanEvidence(BaseModel):
    """Evidence selected by the operation owner for the debugger."""
    kind: SpanKind = SpanKind.Workflow
    parent: str = ""
    knoxel_ids: list[int] = Field(default_factory=list)
    wave: int | None = None
    score: float | None = None
    outputs: list[int] = Field(default_factory=list)
    retrieved_record_ids: list[int] = Field(default_factory=list)
    journal_sequence: int | None = None
    instruction_revision: int | None = None
    tool_succeeded: bool | None = None
    commit_id: str | None = None
    control_generation: int | None = None
    private_only: bool | None = None
    tick: int | None = None


class ProfileEvent(BaseModel):
    """One process-local event. The invocation ID joins it to kernel and model receipts."""

    evidence: SpanEvidence = Field(default_factory=SpanEvidence)
    span_id: str
    invocation_id: str
    operation: str
    wall_at: datetime
    monotonic_ns: int
    thread_id: int
    thread_name: str
    outcome: str
    input_envelope_ids: list[str]
    input_topics: list[str]
    correlation_ids: list[str]


class CognitionProfiler:
    """Own the append lock and trace file for one runtime in one process."""

    def __init__(self, runtime_root: Path) -> None:
        self.path: Path = runtime_root / "profile" / f"spans-{os.getpid()}.jsonl"
        self.lock: threading.Lock = threading.Lock()

    def begin_workflow(self, operation_name: str, invocation_id: str, batch: list[Envelope]) -> ProfileEvent:
        """Commit a start event before acquiring services or entering the kernel."""
        event = ProfileEvent(
            span_id=uuid4().hex, invocation_id=invocation_id, operation=operation_name,
            wall_at=datetime.now(timezone.utc), monotonic_ns=time.perf_counter_ns(),
            thread_id=threading.get_native_id(), thread_name=threading.current_thread().name,
            outcome="started", input_envelope_ids=[item.id for item in batch],
            input_topics=[item.topic for item in batch],
            correlation_ids=[item.correlation_id for item in batch if item.correlation_id],
        )
        self.append(event)
        return event

    @contextmanager
    def operation(self, operation_name: str, invocation_id: str, evidence: SpanEvidence) -> Iterator[ProfileEvent]:
        """Retain input before dispatch and let the owner attach output record IDs before exit."""
        event = ProfileEvent(
            span_id=uuid4().hex, invocation_id=invocation_id, operation=operation_name, evidence=evidence,
            wall_at=datetime.now(timezone.utc), monotonic_ns=time.perf_counter_ns(),
            thread_id=threading.get_native_id(), thread_name=threading.current_thread().name,
            outcome="started", input_envelope_ids=[], input_topics=[], correlation_ids=[],
        )
        self.append(event)
        try:
            yield event
        except BaseException as error:
            try:
                self.finish_workflow(event, type(error).__name__)
            except OSError as trace_error:
                error.add_note(f"Profile write failed for {event.span_id}: {trace_error}")
            raise
        else:
            self.finish_workflow(event, "finished")

    def finish_workflow(self, started: ProfileEvent, outcome: str) -> None:
        """Write the observed outcome with the same identity as the start."""
        finished = started.model_copy()
        finished.wall_at = datetime.now(timezone.utc)
        finished.monotonic_ns = time.perf_counter_ns()
        finished.outcome = outcome
        self.append(finished)

    def append(self, event: ProfileEvent) -> None:
        line = event.model_dump_json()  # pm-lint: allow PM007 -- trace file serialization
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
