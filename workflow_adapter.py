"""Adapts one bus module to one real pm-workflows invocation.

The shared runner owns the kernel call; subclasses own only routing
differences (which messages they accept and how they turn a role report into
bus messages). Each accepted batch starts one invocation against the module's
working target, with the module's own stop event as the cancellation signal.
"""
from __future__ import annotations

import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Mapping

from pm_workflows.runtime import WorkflowRuntime

from bus.envelopes import Envelope
from runtime.shell import ShellModule
from runtime.workflow_runner import (
    PROJECT_ROOT, WorkflowRequest, WorkflowResult, WorkflowRunner, WorkflowStatus,
)


class ShellModuleWorkflow(ShellModule):
    """One module that turns data batches into real workflow invocations."""

    def __init__(
        self,
        module_id: str,
        bus,
        on_error: Callable[[BaseException], None],
        clock: Callable[[], float],
        tick_seconds: float | None,
        *,
        working_target,
        workflow: Path,
        tools_factory: Callable[[Any], Mapping[str, Callable]],
        runtime_root: Path,
        success_status: str = "done",
    ) -> None:
        super().__init__(module_id, bus, on_error, clock, tick_seconds)
        self.working_target = working_target
        self.workflow = Path(workflow)
        self.tools_factory = tools_factory
        self.runtime_root = Path(runtime_root)
        self.success_status = success_status
        self.last_environment: Any | None = None
        self.last_kernel_summary: dict | None = None
        self.last_result: WorkflowResult | None = None

    @abstractmethod
    def prepare_input(self, batch: list) -> dict:
        """Convert bus input into this workflow's request format."""

    @abstractmethod
    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        """Convert a parsed role report into bus messages."""

    # ------------------------------------------------------------ execution

    def handle_batch(self, batch: list[Envelope]) -> None:
        invocation_id = uuid.uuid4().hex
        self.invocation_id = invocation_id
        self.last_kernel_summary = None
        self.last_result = None
        environment = self.working_target.open(invocation_id)
        self.last_environment = environment

        request = self.prepare_input(batch)
        runtime = WorkflowRuntime(
            environment=environment,
            stop_event=self.stop_event,
            tools=self.tools_factory(environment),
            propagate_python_errors=True,
        )

        runner = WorkflowRunner(
            self.workflow, self.runtime_root, namespace=self.module_id,
            success_status=self.success_status,
        )
        result = runner.invoke(WorkflowRequest(invocation_id, request), runtime)
        self.last_result = result
        self.last_kernel_summary = result.kernel_summary

        if result.status in {WorkflowStatus.Cancelled, WorkflowStatus.WaitingForInput}:
            return
        if result.status is WorkflowStatus.Failed:
            raise RuntimeError(
                f"workflow {self.module_id} ended unsuccessfully: {result.kernel_summary}"
            )
        if self.stop_event.is_set():
            return
        self.publish_output(result.report, batch)


__all__ = ["ShellModuleWorkflow", "PROJECT_ROOT"]
