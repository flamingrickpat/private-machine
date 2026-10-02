"""One synchronous kernel call for direct tests and module invocations.

Python errors reach the caller. Invalid role results return a failed outcome.
Cancelled and waiting runs retain their artifacts but never return a report.
Only a successful terminal route permits report loading. A suspended run can
be resumed by writing a durable answer receipt and invoking again.
"""
from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
import time
from typing import Any

import yaml
from pm_workflows.kernel import Kernel
from pm_workflows.runtime import WorkflowRuntime


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class WorkflowStatus(StrEnum):
    Completed = "completed"
    Failed = "failed"
    Cancelled = "cancelled"
    WaitingForInput = "waiting_for_input"


@dataclass(frozen=True)
class WorkflowRequest:
    """The caller supplies a unique run ID and a YAML-compatible payload."""

    run_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class WorkflowResult:
    """Normalized outcome with the raw kernel summary and retained artifact path."""

    status: WorkflowStatus
    run_id: str
    report: dict[str, Any] | None
    kernel_summary: dict[str, Any]
    artifact_dir: Path
    question: str | None = None
    question_id: str | None = None
    resume_phase: str | None = None


class WorkflowRunner:
    """Reuse the real kernel without a bus, module thread, or Ghost dependency.

    Construction starts no work. Each invocation uses the supplied environment
    and cancellation signal. The caller owns their lifetime and unique run ID.
    """

    def __init__(
        self,
        workflow: Path,
        runtime_root: Path,
        *,
        namespace: str = "standalone",
        base_dir: Path = PROJECT_ROOT,
        success_status: str = "done",
        state_lock_attempts: int = 5,
    ) -> None:
        self.workflow = Path(workflow)
        self.runtime_root = Path(runtime_root)
        self.namespace = namespace
        self.base_dir = Path(base_dir)
        self.success_status = success_status
        self.state_lock_attempts = max(1, state_lock_attempts)

    def invoke(
        self, request: WorkflowRequest, runtime: WorkflowRuntime, *, resume: bool = False,
    ) -> WorkflowResult:
        """Run inline and normalize the terminal outcome once.

        A cancelled run does not read its report. Invalid YAML or a non-mapping
        success report raises an error. Python role errors propagate when the
        supplied runtime enables ``propagate_python_errors``. ``resume``
        continues a suspended run from its retained journal.

        The kernel projects its journal into a read-only ``state.md`` through an
        atomic replace. Windows can hold a transient lock on that file (indexer,
        backup, antivirus), and ``os.replace`` then raises ``PermissionError``.
        The journal is the source of truth and is written before the projection,
        so a bounded retry resumes from the journal instead of repeating a
        completed phase or an external effect.
        """
        environment = runtime.environment
        if not runtime.stop_event.is_set():
            environment.write_text(
                "inbox/request.yaml",
                yaml.safe_dump(request.payload, sort_keys=False, allow_unicode=True),
            )
        allowed_mcp = (
            set(runtime.mcp_servers) if runtime.mcp_servers is not None else None
        )
        summary = None
        kernel = None
        for attempt in range(self.state_lock_attempts):
            try:
                kernel = Kernel(
                    manifest_path=self.workflow,
                    workspace=self.runtime_root / self.namespace / request.run_id / "workspace",
                    task_id=request.run_id,
                    task_text=f"Read inbox/request.yaml from the supplied working target ({request.run_id}). "
                              "Write the final YAML report to outbox/report.yaml in that target. "
                              "Agent tools already use the target. Use relative paths.",
                    base_dir=self.base_dir,
                    coding_agent="pm-coder" if runtime.agent_model is not None else "python",
                    kernel_data_root=self.runtime_root / "kernel",
                    run_id=request.run_id,
                    resume=resume or attempt > 0,
                    allowed_mcp=allowed_mcp,
                    runtime=runtime,
                )
                summary = kernel.run()
                break
            except PermissionError:
                if attempt == self.state_lock_attempts - 1:
                    raise
                time.sleep(0.25 * (attempt + 1))
        status = _status(summary, self.success_status, runtime.stop_event.is_set())
        report = None
        question = question_id = resume_phase = None
        if status is WorkflowStatus.Completed:
            report = yaml.safe_load(environment.read_text("outbox/report.yaml"))
            if not isinstance(report, dict):
                raise ValueError("The workflow report must be a YAML mapping.")
            if runtime.stop_event.is_set():
                status, report = WorkflowStatus.Cancelled, None
        elif status is WorkflowStatus.WaitingForInput:
            question, resume_phase = self._waiting_question(kernel.kernel_data)
            question_id = f"{request.run_id}:{resume_phase}"
        return WorkflowResult(
            status, request.run_id, report, summary, kernel.kernel_data,
            question, question_id, resume_phase,
        )

    def _answers_root(self, run_id: str) -> Path:
        return self.runtime_root / "kernel" / run_id / "answers"

    def _waiting_question(self, kernel_data: Path) -> tuple[str | None, str | None]:
        """Return the last durable human question and the phase it suspends."""
        journal = kernel_data / "journal.jsonl"
        if not journal.is_file():
            return None, None
        for line in reversed(journal.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("verdict") == "waiting_user":
                return entry.get("answer"), entry.get("phase")
        return None, None

    def write_answer(
        self, run_id: str, answer: str, *, question_id: str, resume_phase: str,
    ) -> Path:
        """Write one durable ``pm.answer-receipt.v1`` for a suspended run."""
        if not answer.strip():
            raise ValueError("An answer receipt needs nonempty answer text.")
        root = self._answers_root(run_id)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{question_id.replace(':', '_')}.yaml"
        path.write_text(yaml.safe_dump({
            "schema": "pm.answer-receipt.v1",
            "id": question_id,
            "resume_phase": resume_phase,
            "answer": answer,
        }, sort_keys=False), encoding="utf-8")
        return path


def _status(summary: dict, success_status: str, cancelled: bool) -> WorkflowStatus:
    if cancelled or summary["terminal_status"] == "terminated" or summary["exit_reason"] == "terminated":
        return WorkflowStatus.Cancelled
    if summary["exit_reason"] == "suspended:user":
        return WorkflowStatus.WaitingForInput
    # An explicit successful stop route sets ok=False in the existing kernel.
    expected_exits = {"completed", f"{summary['terminal_phase']}: stop"}
    if summary["terminal_status"] == success_status and summary["exit_reason"] in expected_exits:
        return WorkflowStatus.Completed
    return WorkflowStatus.Failed
