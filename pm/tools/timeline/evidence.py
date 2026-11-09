"""Discover simulation run folders and read their durable attempt evidence.

The simulation controller owns the run format. This module is a read-only
adapter: it validates the few fields the review page needs, keeps the raw plan
and report available, and refuses paths that leave the selected run or attempt.

Owns: run and attempt discovery, identity checks, and safe evidence paths.
Does not own: run execution, scenario planning, or any mutation. See the package
overview in ``pm.tools.timeline``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import threading
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from pm.paths import REPO_ROOT

#: The simulation controller writes one folder for each run below this root.
DEFAULT_RUNS_ROOT = REPO_ROOT / ".runtime" / "simulation" / "runs"

#: Run and participant identities are simple folder names. A review target is
#: never an arbitrary path from the browser.
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
PARTICIPANT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class EvidenceError(RuntimeError):
    """The selected evidence is absent, malformed, or outside its root."""


class RunState(BaseModel):
    """The controller document that the simulation server writes.

    The model keeps only the fields this page presents. Unknown fields stay in
    the raw file, which the page shows on demand. Rows inside ``timeline``,
    ``transit`` and ``failures`` are persisted by the controller and have more
    than one shape, so they remain JSON values here.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    title: str = ""
    status: str = ""
    created_at: str = ""
    config: str = ""
    participants: list[dict[str, JsonValue]] = Field(default_factory=list[dict[str, JsonValue]])
    edges: list[dict[str, JsonValue]] = Field(default_factory=list[dict[str, JsonValue]])
    step_index: int = 0
    turn: int = 0
    attempt: int = 0
    clock: str = ""
    timeline: list[dict[str, JsonValue]] = Field(default_factory=list[dict[str, JsonValue]])
    transit: list[dict[str, JsonValue]] = Field(default_factory=list[dict[str, JsonValue]])
    facts: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])
    pending_input: JsonValue = None
    reports: list[str] = Field(default_factory=list[str])
    failures: list[dict[str, JsonValue]] = Field(default_factory=list[dict[str, JsonValue]])


class PlanFact(BaseModel):
    """One mandatory scenario fact, projected from ``plan.json``."""

    id: str
    text: str
    at: str | None = None


class PlanStep(BaseModel):
    """One scenario stage, projected from ``plan.json``."""

    id: str
    at: str
    instruction: str
    fact_ids: list[str] = Field(default_factory=list[str])
    completion: str = ""
    target: str = ""
    speaker: str = ""


class PlanView(BaseModel):
    """Read-only projection of the plan. The simulation package owns the contract."""

    title: str = ""
    brief: str = ""
    start_time: str = ""
    facts: list[PlanFact] = Field(default_factory=list[PlanFact])
    steps: list[PlanStep] = Field(default_factory=list[PlanStep])


class RunSummary(BaseModel):
    """One run row in the run picker."""

    id: str
    title: str = ""
    status: str = ""
    created_at: str = ""
    attempt_count: int = 0
    latest_attempt: int = 0
    participant_count: int = 0
    timeline_count: int = 0
    failure_count: int = 0
    reports: list[str] = Field(default_factory=list[str])


class AttemptSummary(BaseModel):
    """One attempt row for a selected run."""

    number: int
    status: str = ""
    failures: int = 0
    evidence_participants: list[str] = Field(default_factory=list[str])
    has_report: bool = False
    has_agents: bool = False
    has_source: bool = False
    report_path: str = ""


class AttemptBundle(BaseModel):
    """An attempt and the controller report that describes its final state."""

    summary: AttemptSummary
    report: JsonValue = None
    run_timeline: list[dict[str, JsonValue]] = Field(default_factory=list[dict[str, JsonValue]])
    run_failures: list[dict[str, JsonValue]] = Field(default_factory=list[dict[str, JsonValue]])
    facts: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])


class RunBundle(BaseModel):
    """One run, its plan, and its attempt summaries."""

    summary: RunSummary
    state: RunState
    plan: PlanView
    attempts: list[AttemptSummary] = Field(default_factory=list[AttemptSummary])


@dataclass(frozen=True)
class EvidencePaths:
    """Paths that belong to one participant attempt."""

    run_root: Path
    attempt_root: Path
    evidence_root: Path

    @property
    def prompts(self) -> Path:
        """The prompt-log folder for this participant attempt."""
        return self.evidence_root / "logs" / "prompts"

    @property
    def kernels(self) -> Path:
        """The workflow kernel folder for this participant attempt."""
        return self.evidence_root / "runtime" / "kernel"

    @property
    def cycles(self) -> Path:
        """The captured GWT cycle packages for this participant attempt."""
        return self.evidence_root / "cycles"


def require_identity(value: str, pattern: re.Pattern[str], label: str) -> str:
    """Return a safe folder identity or raise ``EvidenceError``.

    The browser supplies these values. Rejecting anything that is not one plain
    path component stops a request from reading a sibling run or a parent folder.
    """
    if not pattern.fullmatch(value):
        raise EvidenceError(f"{label} is not a valid evidence folder name: {value!r}")
    return value


def normalize_time(value: JsonValue) -> str:
    """Return a sortable UTC timestamp for an ISO value.

    A value without an offset is treated as UTC. Simulation rows mix local
    offsets and naive world time, so a raw string sort would put 21:00+01:00
    after 20:00Z even though both describe the same instant.
    """
    if not isinstance(value, str) or not value:
        return ""
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> JsonValue:
    """Read one UTF-8 JSON file. A missing or invalid file raises."""
    if not path.is_file():
        raise FileNotFoundError(f"Evidence file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_object(path: Path) -> dict[str, JsonValue]:
    """Read one JSON object and reject a different top-level shape."""
    value = read_json(path)
    if not isinstance(value, dict):
        raise EvidenceError(f"Expected a JSON object in {path}")
    return value


def _attempt_dirs(run_root: Path) -> list[int]:
    """Return the attempt numbers that exist for a run, in order."""
    root = run_root / "attempts"
    if not root.is_dir():
        return []
    numbers: list[int] = []
    for path in root.iterdir():
        if path.is_dir() and path.name.isdigit():
            numbers.append(int(path.name))
    return sorted(numbers)


def _evidence_participants(attempt_root: Path) -> list[str]:
    """Return participant ids that have a captured evidence folder."""
    root = attempt_root / "evidence"
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


class EvidenceStore:
    """Read simulation run evidence from one root without changing it.

    The store caches parsed runs and attempts. A review browser asks for the
    same run many times while the owner moves between tabs, and the controller
    files do not change during a completed review.
    """

    def __init__(self, runs_root: Path = DEFAULT_RUNS_ROOT) -> None:
        self.runs_root: Path = runs_root.resolve()
        self._lock: threading.RLock = threading.RLock()
        self._runs: dict[str, RunBundle] = {}
        self._attempts: dict[tuple[str, int], AttemptBundle] = {}

    def refresh(self) -> None:
        """Discard parsed views after a new attempt captures evidence."""
        with self._lock:
            self._runs.clear()
            self._attempts.clear()

    def run_root(self, run_id: str) -> Path:
        """Return the checked run folder for ``run_id``."""
        safe_id = require_identity(run_id, RUN_ID, "Run id")
        root = (self.runs_root / safe_id).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Run folder is missing: {root}")
        if not root.is_relative_to(self.runs_root):
            raise EvidenceError("The run folder is outside the evidence root.")
        return root

    def list_runs(self) -> list[RunSummary]:
        """Return every run that has a readable controller document."""
        if not self.runs_root.is_dir():
            return []
        rows: list[RunSummary] = []
        for path in sorted(self.runs_root.iterdir()):
            if not path.is_dir() or not (path / "state.json").is_file():
                continue
            try:
                rows.append(self._summary(path))
            except (EvidenceError, OSError, ValueError, json.JSONDecodeError):
                # A run whose controller document is unreadable still deserves a
                # row, because the owner needs to see that it is broken.
                rows.append(RunSummary(id=path.name, title="(unreadable state.json)", status="unreadable"))
        return rows

    def run(self, run_id: str) -> RunBundle:
        """Return one parsed run and its attempt rows."""
        with self._lock:
            cached = self._runs.get(run_id)
            if cached is not None:
                return cached
            root = self.run_root(run_id)
            state = RunState.model_validate(read_json_object(root / "state.json"))
            plan_path = root / "plan.json"
            plan = PlanView.model_validate(read_json_object(plan_path)) if plan_path.is_file() else PlanView()
            attempts = [self._attempt_summary(run_id, number) for number in _attempt_dirs(root)]
            summary = self._summary(root, state)
            bundle = RunBundle(summary=summary, state=state, plan=plan, attempts=attempts)
            self._runs[run_id] = bundle
            return bundle

    def attempt(self, run_id: str, number: int) -> AttemptBundle:
        """Return one attempt report and the controller timeline at that attempt."""
        key = (run_id, number)
        with self._lock:
            cached = self._attempts.get(key)
            if cached is not None:
                return cached
            run = self.run(run_id)
            root = self.run_root(run_id) / "attempts" / str(number)
            if not root.is_dir() or number not in _attempt_dirs(self.run_root(run_id)):
                raise FileNotFoundError(f"Attempt {number} is missing for run {run_id}.")
            summary = self._attempt_summary(run_id, number)
            report_path = root / "report.json"
            report = read_json(report_path) if report_path.is_file() else None
            reported: JsonValue = report.get("run", {}) if isinstance(report, dict) else {}
            timeline = reported.get("timeline") if isinstance(reported, dict) else None
            failures = reported.get("failures") if isinstance(reported, dict) else None
            facts = reported.get("facts") if isinstance(reported, dict) else None
            reported_timeline = TypeAdapter(list[dict[str, JsonValue]]).validate_python(timeline) if timeline is not None else []
            bundle = AttemptBundle(
                summary=summary,
                report=report,
                run_timeline=reported_timeline or run.state.timeline,
                run_failures=TypeAdapter(list[dict[str, JsonValue]]).validate_python(failures) if failures is not None else run.state.failures,
                facts=dict(facts) if isinstance(facts, dict) else run.state.facts,
            )
            self._attempts[key] = bundle
            return bundle

    def paths(self, run_id: str, attempt: int, participant: str) -> EvidencePaths:
        """Return safe evidence paths for one participant attempt."""
        safe_participant = require_identity(participant, PARTICIPANT_ID, "Participant id")
        run_root = self.run_root(run_id)
        attempt_root = (run_root / "attempts" / str(attempt)).resolve()
        if not attempt_root.is_dir() or not attempt_root.is_relative_to(run_root):
            raise FileNotFoundError(f"Attempt {attempt} is missing for run {run_id}.")
        evidence_root = (attempt_root / "evidence" / safe_participant).resolve()
        if not evidence_root.is_dir() or not evidence_root.is_relative_to(attempt_root):
            raise FileNotFoundError(f"Participant {safe_participant} has no evidence in attempt {attempt}.")
        return EvidencePaths(run_root=run_root, attempt_root=attempt_root, evidence_root=evidence_root)

    def resolve_text(self, root: Path, relative: str, *, limit: int = 2_000_000) -> tuple[str, int]:
        """Read one text artifact inside ``root`` and return its text and size.

        The browser sends a relative path from an artifact list. The resolved
        path must stay below the participant evidence root. A binary file is
        refused, and a file larger than ``limit`` is refused rather than
        truncated silently.
        """
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()):
            raise EvidenceError("The artifact path leaves the participant evidence root.")
        if not target.is_file():
            raise FileNotFoundError(f"Artifact is missing: {relative}")
        size = target.stat().st_size
        if size > limit:
            raise EvidenceError(f"Artifact is too large to open here ({size} bytes): {relative}")
        try:
            return target.read_text(encoding="utf-8"), size
        except UnicodeDecodeError as error:
            raise EvidenceError(f"Artifact is not UTF-8 text: {relative}") from error

    def _summary(self, root: Path, state: RunState | None = None) -> RunSummary:
        """Build one run row, reading the controller document when needed."""
        if state is None:
            state = RunState.model_validate(read_json_object(root / "state.json"))
        numbers = _attempt_dirs(root)
        return RunSummary(
            id=state.id or root.name,
            title=state.title,
            status=state.status,
            created_at=state.created_at,
            attempt_count=len(numbers),
            latest_attempt=max(numbers, default=0),
            participant_count=len(state.participants),
            timeline_count=len(state.timeline),
            failure_count=len(state.failures),
            reports=list(state.reports),
        )

    def _attempt_summary(self, run_id: str, number: int) -> AttemptSummary:
        """Build one attempt row from its report and evidence folders."""
        root = self.run_root(run_id) / "attempts" / str(number)
        report_path = root / "report.json"
        report = read_json_object(report_path) if report_path.is_file() else {}
        reported = report.get("run", {})
        failures = reported.get("failures") if isinstance(reported, dict) else None
        return AttemptSummary(
            number=number,
            status=str(reported.get("status", "")) if isinstance(reported, dict) else "",
            failures=len(failures) if isinstance(failures, list) else 0,
            evidence_participants=_evidence_participants(root),
            has_report=report_path.is_file(),
            has_agents=(root / "agents").is_dir(),
            has_source=(root / "source").is_dir(),
            report_path=report_path.relative_to(root).as_posix() if report_path.is_file() else "",
        )


__all__ = [
    "AttemptBundle",
    "AttemptSummary",
    "DEFAULT_RUNS_ROOT",
    "EvidenceError",
    "EvidencePaths",
    "EvidenceStore",
    "PlanFact",
    "PlanStep",
    "PlanView",
    "RunBundle",
    "RunState",
    "RunSummary",
    "normalize_time",
    "read_json",
    "read_json_object",
    "require_identity",
]
