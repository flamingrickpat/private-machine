"""One synchronous kernel call for direct tests and module invocations.

Python errors reach the caller. Invalid role results return a failed outcome.
Cancelled and waiting runs retain their artifacts but never return a report.
Only a successful terminal route permits report loading.
"""
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
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
    ) -> None:
        self.workflow = Path(workflow)
        self.runtime_root = Path(runtime_root)
        self.namespace = namespace
        self.base_dir = Path(base_dir)
        self.success_status = success_status

    def invoke(self, request: WorkflowRequest, runtime: WorkflowRuntime) -> WorkflowResult:
        """Run inline and normalize the terminal outcome once.

        A cancelled run does not read its report. Invalid YAML or a non-mapping
        success report raises an error. Python role errors propagate when the
        supplied runtime enables ``propagate_python_errors``.
        """
        environment = runtime.environment
        if not runtime.stop_event.is_set():
            environment.write_text(
                "inbox/request.yaml",
                yaml.safe_dump(request.payload, sort_keys=False, allow_unicode=True),
            )
        kernel = Kernel(
            manifest_path=self.workflow,
            workspace=self.runtime_root / self.namespace / request.run_id / "workspace",
            task_id=request.run_id,
            task_text=f"Read inbox/request.yaml from runtime.environment ({request.run_id}).",
            base_dir=self.base_dir,
            # Keep the existing Python-only path until agent target binding is complete.
            coding_agent="python",
            kernel_data_root=self.runtime_root / "kernel",
            run_id=request.run_id,
            resume=False,
            runtime=runtime,
        )
        summary = kernel.run()
        status = _status(summary, self.success_status, runtime.stop_event.is_set())
        report = None
        if status is WorkflowStatus.Completed:
            report = yaml.safe_load(environment.read_text("outbox/report.yaml"))
            if not isinstance(report, dict):
                raise ValueError("The workflow report must be a YAML mapping.")
            if runtime.stop_event.is_set():
                status, report = WorkflowStatus.Cancelled, None
        return WorkflowResult(status, request.run_id, report, summary, kernel.kernel_data)


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
