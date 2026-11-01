"""Read workflow kernel journals and their local artifacts.

A kernel folder is named by its invocation ID. ``journal.jsonl`` records each
phase, routing decision, verdict, timestamp, and trace path. ``state.md`` names
the workflow and its accepted phase history. The other files are the role
traces, model sessions, and request or result artifacts that the workflow wrote.

Owns: the read-only kernel index for one participant attempt.
Does not own: workflow execution or the simulation report.
See the package overview in ``pm.tools.timeline``.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, JsonValue

from pm.tools.timeline.calls import CallIndex
from pm.tools.timeline.evidence import normalize_time


class KernelPhase(BaseModel):
    """One journal row: a role result, route decision, or failure."""

    phase: str
    kind: str = ""
    role: str = ""
    attempt: int = 0
    ok: bool | None = None
    verdict: str = ""
    status: str = ""
    summary: str = ""
    timestamp: str = ""
    trace_path: str = ""
    session_ref: str = ""


class ArtifactInfo(BaseModel):
    """One file below a kernel invocation folder."""

    name: str
    relative: str
    size: int
    kind: str


class WorkflowRow(BaseModel):
    """One workflow invocation as a list row."""

    invocation_id: str
    participant: str
    workflow: str = ""
    status: str = ""
    world_time: str = ""
    started_at: str = ""
    ended_at: str = ""
    phase_count: int = 0
    artifact_count: int = 0
    call_count: int = 0
    total_tokens: int = 0


class KernelRun(BaseModel):
    """One workflow invocation and the evidence it left behind."""

    invocation_id: str
    participant: str
    workflow: str = ""
    status: str = ""
    started_at: str = ""
    ended_at: str = ""
    world_time: str = ""
    phases: list[KernelPhase] = Field(default_factory=list[KernelPhase])
    artifacts: list[ArtifactInfo] = Field(default_factory=list[ArtifactInfo])
    call_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _read_journal(path: Path) -> list[dict[str, JsonValue]]:
    """Read a journal, keeping malformed rows as an explicit error phase."""
    rows: list[dict[str, JsonValue]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value: JsonValue = json.loads(line)
        except json.JSONDecodeError:
            rows.append({"phase": "unreadable-journal-row", "kind": "error", "ok": False,
                         "verdict": "unreadable", "result": {"summary": line[:300]}})
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _phase(row: dict[str, JsonValue]) -> KernelPhase:
    """Project one journal row to the fields the timeline shows."""
    result = row.get("result")
    summary = ""
    if isinstance(result, dict):
        value = result.get("summary")
        if isinstance(value, str):
            summary = value
    if not summary:
        value = row.get("verdict")
        if isinstance(value, str):
            summary = value
    attempt = row.get("attempt", 0)
    ok = row.get("ok")
    return KernelPhase(
        phase=str(row.get("phase", "")),
        kind=str(row.get("kind", "")),
        role=str(row.get("role", "")),
        attempt=attempt if isinstance(attempt, int) else 0,
        ok=ok if isinstance(ok, bool) else None,
        verdict=str(row.get("verdict", "")),
        status=str(row.get("status", "")),
        summary=summary,
        timestamp=str(row.get("timestamp", "")),
        trace_path=str(row.get("trace_path", "")),
        session_ref=str(row.get("session_ref", "")),
    )


def _artifact_kind(path: Path) -> str:
    """Name the artifact by what a reader can do with it."""
    if path.name == "journal.jsonl":
        return "journal"
    if path.name == "state.md":
        return "state"
    if path.suffix == ".json":
        return "json"
    if path.suffix == ".jsonl":
        return "jsonl"
    if path.suffix in {".md", ".txt"}:
        return "text"
    if path.suffix in {".yaml", ".yml"}:
        return "yaml"
    return "binary"


def _workflow_name(state_markdown: str, invocation_id: str) -> str:
    """Read the workflow name from the kernel state header."""
    for line in state_markdown.splitlines():
        if line.startswith("- workflow:"):
            value = line.partition(":")[2].strip()
            if value:
                return value
    return invocation_id[:8]


class KernelIndex:
    """Every workflow kernel invocation in one participant attempt."""

    def __init__(self, participant: str, kernel_root: Path, evidence_root: Path, calls: CallIndex) -> None:
        self.participant: str = participant
        self.kernel_root: Path = kernel_root
        self.evidence_root: Path = evidence_root
        self.calls: CallIndex = calls
        self._runs: dict[str, KernelRun] = {}
        self._load()

    def _load(self) -> None:
        if not self.kernel_root.is_dir():
            return
        for path in sorted(self.kernel_root.iterdir()):
            if not path.is_dir():
                continue
            journal = _read_journal(path / "journal.jsonl")
            phases = [_phase(row) for row in journal]
            state_markdown = (path / "state.md").read_text(encoding="utf-8") if (path / "state.md").is_file() else ""
            started = min((phase.timestamp for phase in phases if phase.timestamp), default="")
            ended = max((phase.timestamp for phase in phases if phase.timestamp), default="")
            invocation_calls = self.calls.for_invocation(path.name)
            status = ""
            for phase in reversed(phases):
                if phase.status:
                    status = phase.status
                    break
            if not status and phases:
                status = phases[-1].verdict
            artifacts: list[ArtifactInfo] = []
            for item in sorted(path.rglob("*")):
                if not item.is_file():
                    continue
                try:
                    relative = item.relative_to(self.evidence_root).as_posix()
                except ValueError:
                    relative = ""
                artifacts.append(ArtifactInfo(
                    name=item.relative_to(path).as_posix(),
                    relative=relative,
                    size=item.stat().st_size,
                    kind=_artifact_kind(item),
                ))
            self._runs[path.name] = KernelRun(
                invocation_id=path.name,
                participant=self.participant,
                workflow=_workflow_name(state_markdown, path.name),
                status=status,
                started_at=started,
                ended_at=ended,
                world_time=self.calls.world_time_for_invocation(path.name),
                phases=phases,
                artifacts=artifacts,
                call_count=len(invocation_calls),
                prompt_tokens=sum(call.prompt_tokens or 0 for call in invocation_calls),
                completion_tokens=sum(call.completion_tokens or 0 for call in invocation_calls),
                total_tokens=sum(call.total_tokens or 0 for call in invocation_calls),
            )

    def all(self) -> list[KernelRun]:
        """Return every kernel run, ordered by virtual time and then start time."""
        return sorted(self._runs.values(), key=lambda run: (normalize_time(run.world_time) or "9999",
                                                            normalize_time(run.started_at) or "9999",
                                                            run.invocation_id))

    def rows(self) -> list[WorkflowRow]:
        """Return every kernel run as a table row."""
        return [WorkflowRow(
            invocation_id=run.invocation_id, participant=run.participant, workflow=run.workflow,
            status=run.status, world_time=run.world_time, started_at=run.started_at,
            ended_at=run.ended_at, phase_count=len(run.phases), artifact_count=len(run.artifacts),
            call_count=run.call_count, total_tokens=run.total_tokens) for run in self.all()]

    def get(self, invocation_id: str) -> KernelRun:
        """Return one kernel run or raise ``FileNotFoundError``."""
        run = self._runs.get(invocation_id)
        if run is None:
            raise FileNotFoundError(f"No kernel invocation {invocation_id}.")
        return run


def index_kernels(participant: str, kernel_root: Path, evidence_root: Path, calls: CallIndex) -> KernelIndex:
    """Build the kernel index for one participant attempt."""
    return KernelIndex(participant, kernel_root, evidence_root, calls)


__all__ = ["ArtifactInfo", "KernelIndex", "KernelPhase", "KernelRun", "WorkflowRow", "index_kernels"]
