"""Adapts one bus module to one real pm-workflows invocation.

The shared adapter owns the kernel call; subclasses own only routing
differences (which messages they accept and how they turn a role report into
bus messages). Each accepted batch starts one invocation against the module's
working target, with the module's own stop event as the cancellation signal.
"""
from __future__ import annotations

import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from pm_workflows.kernel import Kernel
from pm_workflows.runtime import WorkflowRuntime

from ghost import serialize_knoxel
from shell import Envelope, ShellModule

PROJECT_ROOT = Path(__file__).resolve().parent


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
        ghost,
    ) -> None:
        super().__init__(module_id, bus, on_error, clock, tick_seconds)
        self.working_target = working_target
        self.workflow = Path(workflow)
        self.tools_factory = tools_factory
        self.runtime_root = Path(runtime_root)
        self.ghost = ghost
        self.last_environment: Any | None = None
        self.last_kernel_summary: dict | None = None

    # ------------------------------------------------------------ shared writes

    def _serialize_reference(self, payload: dict, key: str) -> dict | None:
        value = payload.get(key)
        if value is None:
            return None
        knoxel = self.ghost.get_knoxel_by_id(value)
        if knoxel is None:
            return None
        return serialize_knoxel(knoxel)

    def prepare_input(self, batch: list) -> dict:
        messages: list[dict] = []
        for envelope in batch:
            message = {
                "id": envelope.id,
                "source": envelope.source,
                "kind": envelope.kind,
                "topic": envelope.topic,
                "correlation_id": envelope.correlation_id,
                "payload": envelope.payload,
            }
            knoxel = self._serialize_reference(envelope.payload, "knoxel_id")
            if knoxel is not None:
                message["knoxel"] = knoxel
            tick = envelope.payload.get("gwt_tick")
            if tick is not None:
                record = self.ghost.gwt_turns.get(tick)
                if record is not None:
                    message["gwt"] = {
                        "__type__": "GwtTurnRecord",
                        "data": record.model_dump(mode="json"),
                    }
            messages.append(message)
        return {
            "invocation_id": self.invocation_id,
            "module_id": self.module_id,
            "messages": messages,
        }

    @abstractmethod
    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        """Convert a parsed role report into bus messages."""

    # ------------------------------------------------------------ execution

    def handle_batch(self, batch: list[Envelope]) -> None:
        invocation_id = uuid.uuid4().hex
        self.invocation_id = invocation_id
        environment = self.working_target.open(invocation_id)
        self.last_environment = environment

        request = self.prepare_input(batch)
        environment.write_text(
            "inbox/request.yaml",
            yaml.safe_dump(request, sort_keys=False, allow_unicode=True),
        )

        runtime = WorkflowRuntime(
            environment=environment,
            stop_event=self.stop_event,
            tools=self.tools_factory(environment),
            propagate_python_errors=True,
        )

        kernel = Kernel(
            manifest_path=self.workflow,
            workspace=self.runtime_root / self.module_id / invocation_id / "workspace",
            task_id=invocation_id,
            task_text=f"Read inbox/request.yaml from runtime.environment ({invocation_id}).",
            base_dir=PROJECT_ROOT,
            coding_agent="python",
            kernel_data_root=self.runtime_root / "kernel",
            run_id=invocation_id,
            resume=False,
            runtime=runtime,
        )
        summary = kernel.run()
        self.last_kernel_summary = summary

        if summary.get("terminal_status") == "terminated" or summary.get("exit_reason") == "terminated":
            return
        if summary.get("terminal_status") != "done":
            raise RuntimeError(
                f"workflow {self.module_id} ended unsuccessfully: {summary}"
            )
        if self.stop_event.is_set():
            return

        report = yaml.safe_load(environment.read_text("outbox/report.yaml"))
        self.publish_output(report, batch)


__all__ = ["ShellModuleWorkflow", "PROJECT_ROOT"]