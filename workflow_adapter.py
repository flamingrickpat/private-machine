"""Adapts one bus module to one real pm-workflows invocation.

The shared runner owns the kernel call; subclasses own only routing
differences (which messages they accept and how they turn a role report into
bus messages). Each accepted batch starts one invocation against the module's
working target.

A run has its own cancellation event, separate from the module's shutdown
event. Stopping the module also cancels its active run; cancelling one run
does not stop the module or any other run. A run that suspends for user input
publishes a question and resumes when a correlated answer arrives.
"""
from __future__ import annotations

import threading
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Mapping

from fastmcp import FastMCP
from pm_workflows.runtime import AgentModel, WorkflowRuntime

from bus.envelopes import Envelope, _new_id
from runtime.mcp_services import serve_services
from runtime.shell import ShellModule
from runtime.workflow_models import WorkflowModels
from runtime.workflow_runner import (
    PROJECT_ROOT, WorkflowRequest, WorkflowResult, WorkflowRunner, WorkflowStatus,
)


class ShellModuleWorkflow(ShellModule):
    """One module that turns data batches into real workflow invocations."""

    QUESTION_TOPIC = "workflow.question"
    ANSWER_TOPIC = "workflow.answer"

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
        tools_factory: Callable[[Any, Any], Mapping[str, Callable]],
        runtime_root: Path,
        success_status: str = "done",
        agent_model: AgentModel | None = None,
        batch_provider: Callable[[], Any] | None = None,
        mcp_services_factory: Callable[[Any], Mapping[str, FastMCP]] | None = None,
        child_factory: Callable[[str], WorkflowRuntime] | None = None,
        workflow_models: WorkflowModels | None = None,
    ) -> None:
        super().__init__(module_id, bus, on_error, clock, tick_seconds)
        self.working_target = working_target
        self.workflow = Path(workflow)
        self.tools_factory = tools_factory
        self.runtime_root = Path(runtime_root)
        self.success_status = success_status
        self.agent_model = agent_model
        self.batch_provider = batch_provider
        self.mcp_services_factory = mcp_services_factory
        self.child_factory = child_factory
        self.workflow_models = workflow_models
        # The active run's cancellation signal. Distinct from ``stop_event``
        # so a single run can be cancelled without stopping the module.
        self.run_stop = threading.Event()
        self._pending: dict[str, Any] | None = None
        self.last_environment: Any | None = None
        self.last_kernel_summary: dict | None = None
        self.last_result: WorkflowResult | None = None
        self.last_proposal_batch: Any | None = None

    @abstractmethod
    def prepare_input(self, batch: list) -> dict:
        """Convert bus input into this workflow's request format."""

    @abstractmethod
    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        """Convert a parsed role report into bus messages."""

    def publish_question(self, result: WorkflowResult, batch: list[Envelope]) -> None:
        """Announce a suspended run's question for routing by MainCogLoop."""
        self.bus.put(Envelope(
            id=_new_id(),
            source=self.module_id,
            kind="telemetry",
            topic=self.QUESTION_TOPIC,
            correlation_id=result.run_id,
            payload={
                "module_id": self.module_id,
                "run_id": result.run_id,
                "question_id": result.question_id,
                "resume_phase": result.resume_phase,
                "question": result.question,
            },
        ))

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == self.ANSWER_TOPIC and envelope.kind == "data":
            return envelope.target in {None, self.module_id}
        return super().accepts(envelope)

    # ------------------------------------------------------------ lifecycle

    def stop(self) -> None:
        self.run_stop.set()
        super().stop()

    def deliver(self, envelope: Envelope) -> None:
        # A terminate control reaches the module through the bus without
        # calling ``stop()``; cancel its active run as well.
        if envelope.kind == "control":
            self.run_stop.set()
        super().deliver(envelope)

    def cancel_run(self) -> None:
        """Cancel the active run without stopping the module."""
        self.run_stop.set()
        self.mailbox.wake()

    # ------------------------------------------------------------ execution

    def handle_batch(self, batch: list[Envelope]) -> None:
        if self._pending is not None:
            answer = self._answer_for(batch)
            if answer is None:
                return  # unrelated input while a question is outstanding
            self._resume(answer)
            return
        self._start(batch)

    def _start(self, batch: list[Envelope]) -> None:
        invocation_id = uuid.uuid4().hex
        self.invocation_id = invocation_id
        self.run_stop = threading.Event()
        self.last_kernel_summary = None
        self.last_result = None
        self.last_proposal_batch = None
        environment = self.working_target.open(invocation_id)
        self.last_environment = environment
        request = WorkflowRequest(invocation_id, self.prepare_input(batch))
        self._execute(request, environment, batch, resume=False)

    def _resume(self, answer: str) -> None:
        pending = self._pending
        self._pending = None
        runner = self._runner()
        runner.write_answer(
            pending["request"].run_id,
            answer,
            question_id=pending["question_id"],
            resume_phase=pending["resume_phase"],
        )
        self._execute(
            pending["request"], pending["environment"], pending["envelopes"],
            resume=True,
        )

    def _execute(
        self,
        request: WorkflowRequest,
        environment: Any,
        batch: list[Envelope],
        *,
        resume: bool,
    ) -> None:
        proposal_batch = self.batch_provider() if self.batch_provider else None
        tools = self.tools_factory(environment, proposal_batch)
        servers = self.mcp_services_factory(environment) if self.mcp_services_factory else {}
        with serve_services(servers) as descriptors:
            runtime = WorkflowRuntime(
                environment=environment,
                stop_event=self.run_stop,
                tools=tools,
                propagate_python_errors=True,
                agent_model=self.agent_model,
                mcp_servers=descriptors or None,
                child_factory=self.child_factory,
            )
            result = self._runner().invoke(request, runtime, resume=resume)
        self.last_result = result
        self.last_kernel_summary = result.kernel_summary
        self.last_proposal_batch = proposal_batch

        if result.status is WorkflowStatus.Cancelled:
            self._discard(proposal_batch)
            self._pending = None
            return
        if result.status is WorkflowStatus.Failed:
            self._discard(proposal_batch)
            self._pending = None
            raise RuntimeError(
                f"workflow {self.module_id} ended unsuccessfully: {result.kernel_summary}"
            )
        if result.status is WorkflowStatus.WaitingForInput:
            self._pending = {
                "request": request,
                "environment": environment,
                "envelopes": batch,
                "question_id": result.question_id,
                "resume_phase": result.resume_phase,
            }
            self.publish_question(result, batch)
            return
        if self.run_stop.is_set():
            self._discard(proposal_batch)
            return
        self.publish_output(result.report, batch)

    def _runner(self) -> WorkflowRunner:
        return WorkflowRunner(
            self.workflow, self.runtime_root, namespace=self.module_id,
            success_status=self.success_status,
            models=self.workflow_models,
        )

    @staticmethod
    def _discard(proposal_batch: Any) -> None:
        if proposal_batch is not None:
            proposal_batch.discard()

    def _answer_for(self, batch: list[Envelope]) -> str | None:
        pending = self._pending or {}
        for envelope in batch:
            if envelope.topic != self.ANSWER_TOPIC:
                continue
            if envelope.payload.get("question_id") not in {None, pending.get("question_id")}:
                continue
            answer = envelope.payload.get("answer")
            if isinstance(answer, str) and answer.strip():
                return answer
        return None


__all__ = ["ShellModuleWorkflow", "PROJECT_ROOT"]
