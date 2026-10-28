"""Adapts one bus module to one real pm-workflows invocation.

The shared runner owns the kernel call; subclasses own only routing
differences (which messages they accept and how they turn a role report into
bus messages). Each accepted batch starts one invocation against the module's
working target.

A run has its own cancellation event, separate from the module's shutdown
event. Stopping the module also cancels its active run; cancelling one run
does not stop the module or any other run. A run that suspends for user input
publishes a question and resumes when a correlated answer arrives.

A parked run is retained in two places: the accepted ``ModuleOutput`` record,
and a per-module receipt file for the case where accepted memory has no causal
history. Restoration installs the recorded environment before the kernel
resumes, so a resumed run reads the files it stopped with.
"""
from __future__ import annotations

import threading
import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
import uuid
from abc import abstractmethod
from pathlib import Path
import json
from typing import Any, Callable, Mapping, TypedDict

from fastmcp import FastMCP
from pm_workflows.runtime import AgentModel, WorkflowRuntime
from pm_workflows.working_target import WorkingEnvironment, WorkingTarget

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, new_id
from pm.host.mcp_listeners import serve_services
from pm.host.application import BusModule
from pm.llm.workflow_sessions import WorkflowModels
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowResult, WorkflowRunner, WorkflowStatus
from pm.bus.envelopes import Envelope
from pm.memory.store_tools import build_tools
from pm.memory.read_scope import ReadScope, current as current_reads
from pm.memory.store import MemoryStore


class PendingInvocation(TypedDict):
    """A run that suspended for outside input and waits for its answer.

    ``request`` and ``envelopes`` are the original run input, so the resumed
    invocation continues the same run instead of starting a new one. The
    environment is the live working target of that invocation.
    """

    request: WorkflowRequest
    environment: WorkingEnvironment
    envelopes: list[Envelope]
    question_id: str | None
    resume_phase: str | None


class WorkflowModule(BusModule):
    """One module that turns data batches into real workflow invocations."""

    QUESTION_TOPIC = "workflow.question"
    ANSWER_TOPIC = "workflow.answer"

    def __init__(
        self,
        module_id: str,
        bus: BusSystem,
        on_error: Callable[[BaseException], None],
        clock: Callable[[], float],
        tick_seconds: float | None,
        *,
        working_target: WorkingTarget,
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
        self._pending: PendingInvocation | None = None
        # The memory store of a cognitive module; None for a module that only
        # routes files. ``CognitiveWorkflow`` supplies it.
        self.ghost: MemoryStore | None = None
        self.last_environment: Any | None = None
        self.last_kernel_summary: dict | None = None
        self.last_result: WorkflowResult | None = None
        self.last_proposal_batch: Any | None = None

    @abstractmethod
    def prepare_input(self, batch: list[Envelope]) -> dict[str, Any]:
        """Convert bus input into this workflow's request format."""

    @abstractmethod
    def publish_output(self, report: dict[str, Any], batch: list[Envelope]) -> None:
        """Convert a parsed role report into bus messages."""

    def publish_question(self, result: WorkflowResult, batch: list[Envelope]) -> None:
        """Announce a suspended run's question for routing by MainCogLoop."""
        self.bus.put(Envelope(
            id=new_id(),
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

    @property
    def parked_path(self) -> Path:
        """Return this module's durable wait receipt path."""
        return self.runtime_root / "parked" / f"{self.module_id}.json"

    def save_parked(self) -> None:
        """Persist a parked request after its kernel journal reaches the wait.

        No running model call is serialized. The kernel journal owns resume.
        Missing or excluded input records prevent restoration of that branch.
        """
        from pm.host.durable_files import atomic_json
        pending = self._pending
        if pending is None:
            self.parked_path.unlink(missing_ok=True)
            return
        view = getattr(pending["environment"], "ghost_view", None)
        # The parked receipt is a JSON record whose parts are read back by
        # ``restore_parked``: the run input, the question, and later the kernel
        # state and file bytes.
        state: dict[str, Any] = {
            "run_id": pending["request"].run_id, "payload": pending["request"].payload,
            "envelopes": [item.to_json() for item in pending["envelopes"]],
            "question_id": pending["question_id"], "resume_phase": pending["resume_phase"],
            "visible_ids": list(view.revisions) if view is not None else None}
        atomic_json(self.parked_path, state)
        ghost = getattr(self.working_target, "ghost", None)
        if ghost is not None:
            from pm.records.codelet_records import ModuleOutput
            from pm.runs.parked_state import capture_kernel
            state["kernel_state"] = capture_kernel(self.runtime_root, self.module_id, state["run_id"])
            environment = pending["environment"]
            state["files"] = {name: environment.read_text(name)
                              for directory in ("inbox", "outbox", "scratch") for name in environment.list_files(directory)}
            changes = ghost.begin_batch(f"parked:{self.module_id}:{state['run_id']}:{state['question_id']}")
            if view is not None:
                changes.depend_on(view.revisions)
                changes.depend_on_commits(view.commit_ids)
            action_id = next((item.payload.get("knoxel_id") for item in pending["envelopes"]
                              if item.topic == "action.request"), None)
            if action_id is not None:
                changes.depend_on([action_id], authority=action_id)
            changes.create(ModuleOutput(content="The workflow is parked for outside input.", source_module=self.module_id,
                status="waiting_for_input", metadata={"parked_module": state, "memory_kind": "runtime_receipt"}))
            ghost.commit(changes)

    def restore_parked(self) -> None:
        """Restore an accepted parked invocation without calling its model."""
        ghost = getattr(self.working_target, "ghost", None)
        raw = None
        if ghost is not None:
            from pm.records.codelet_records import ModuleOutput
            finished = {item.metadata.get("finished_invocation_id") for item in ghost.all_knoxels.values()
                        if isinstance(item, ModuleOutput)}
            candidates = [item for item in ghost.all_knoxels.values() if isinstance(item, ModuleOutput)
                          and item.source_module == self.module_id and item.metadata.get("parked_module")
                          and item.metadata["parked_module"]["run_id"] not in finished]
            if candidates:
                raw = candidates[-1].metadata["parked_module"]
        if raw is None:
            if not self.parked_path.exists() or ghost is not None and ghost.causal_history is not None:
                return
            raw = json.loads(self.parked_path.read_text(encoding="utf-8"))
        visible_ids = raw["visible_ids"]
        if ghost is not None and visible_ids is not None:
            if any(ghost.get_knoxel_by_id(key) is None for key in visible_ids):
                return
            from pm.memory.bash_view import MemoryBashEnvironment
            view = ghost.snapshot(visible_ids)
            environment = MemoryBashEnvironment(view.all_knoxels, view._gwt, "read", ghost_view=view)
        else:
            environment = self.working_target.open(raw["run_id"])
        if raw.get("kernel_state"):
            from pm.runs.parked_state import restore_kernel
            restore_kernel(self.runtime_root, self.module_id, raw["run_id"], raw["kernel_state"])
            for name, content in raw.get("files", {}).items():
                import posixpath
                import shlex
                parent = posixpath.dirname(name)
                result = environment.exec("mkdir -p -- " + shlex.quote(parent))
                if result.exit_code != 0:
                    raise RuntimeError(result.stderr)
                environment.write_text(name, content)
        self.invocation_id = raw["run_id"]
        self.last_environment = environment
        envelopes = [Envelope.from_json(item) for item in raw["envelopes"]]
        self.prepare_input(envelopes)
        self._pending = {"request": WorkflowRequest(raw["run_id"], raw["payload"]),
                         "environment": environment, "envelopes": envelopes,
                         "question_id": raw["question_id"], "resume_phase": raw["resume_phase"]}

    # ------------------------------------------------------------ execution

    def handle_batch(self, batch: list[Envelope]) -> None:
        from pm.memory.snapshot import GhostCancelled
        from pm.llm.prompt_log import invocation_scope
        try:
            with ReadScope(), invocation_scope(self.module_id, lambda: self.invocation_id):
                self._handle_batch(batch)
        except (asyncio.CancelledError, FutureCancelledError, GhostCancelled):
            # Only an explicit cancellation signal makes cancellation expected.
            # Logic and transport errors still reach the application during stop.
            if not (self.stop_event.is_set() or self.run_stop.is_set()):
                raise

    def _handle_batch(self, batch: list[Envelope]) -> None:
        """Start or resume a run unless the module already stopped."""
        if self.stop_event.is_set():
            return
        if self._pending is not None:
            answer = self._answer_for(batch)
            if answer is None:
                return  # unrelated input while a question is outstanding
            self._resume(answer)
            return
        self._start(batch)

    def _start(self, batch: list[Envelope]) -> None:
        """Begin one invocation with a fresh run signal and working target."""
        invocation_id = uuid.uuid4().hex
        self.invocation_id = invocation_id
        self.run_stop = threading.Event()
        # A stop can race with replacement of the previous run's signal.
        # Recheck the module signal after installing the new run signal.
        if self.stop_event.is_set():
            self.run_stop.set()
            return
        self.last_kernel_summary = None
        self.last_result = None
        self.last_proposal_batch = None
        environment = self.working_target.open(invocation_id)
        self.last_environment = environment
        request = WorkflowRequest(invocation_id, self.prepare_input(batch))
        self._execute(request, environment, batch, resume=False)

    def _resume(self, answer: str) -> None:
        """Write the durable answer, then continue the same run."""
        pending = self._pending
        self._pending = None
        ghost = getattr(self.working_target, "ghost", None)
        if ghost is not None:
            ghost.snapshot()
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
        environment: WorkingEnvironment,
        batch: list[Envelope],
        *,
        resume: bool,
    ) -> None:
        """Run one kernel invocation and route exactly its terminal outcome."""
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
            self._record_failure(request, batch, result)
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
            self.save_parked()
            self.publish_question(result, batch)
            return
        if self.run_stop.is_set():
            self._discard(proposal_batch)
            return
        reads = current_reads()
        if reads is not None:
            reads.collecting = False
            reads.completed_invocation_id = request.run_id
        self.publish_output(result.report, batch)
        self.parked_path.unlink(missing_ok=True)

    def _runner(self) -> WorkflowRunner:
        """Reuse one runner configuration for this module's kernel calls."""
        return WorkflowRunner(
            self.workflow, self.runtime_root, namespace=self.module_id,
            success_status=self.success_status,
            models=self.workflow_models,
        )

    def _record_failure(self, request: WorkflowRequest, envelopes: list[Envelope], result: WorkflowResult) -> None:
        """Finish an approved action only for a declared terminal kernel failure.

        Python and transport exceptions bypass this method. Their selected
        actions remain pending because those exceptions prove no clean outcome.
        """
        ghost = self.ghost
        if ghost is None or not envelopes or envelopes[0].topic != "action.request":
            return
        action_id = envelopes[0].payload.get("knoxel_id")
        if action_id is None:
            return
        reads = current_reads()
        if reads is not None:
            reads.collecting = False
            reads.completed_invocation_id = request.run_id
        changes = ghost.begin_batch(f"workflow-failure:{request.run_id}")
        changes.depend_on([action_id], authority=action_id,
                          manifest={"kind": "workflow_failure", "summary": result.kernel_summary})
        changes.set_lifecycle(action_id, dispatch_status="admitted", execution_status="failed")
        ghost.commit(changes)

    @staticmethod
    def _discard(proposal_batch: Any) -> None:
        """Discard a proposal batch whose run produced no accepted output."""
        if proposal_batch is not None:
            proposal_batch.discard()

    def _answer_for(self, batch: list[Envelope]) -> str | None:
        """Return the answer that belongs to the outstanding question, if any."""
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


__all__ = ["WorkflowModule", "PROJECT_ROOT"]


class CognitiveWorkflow(WorkflowModule):
    """Knoxel routing shared by cognitive workflows, outside the generic runner."""

    def __init__(
        self,
        module_id: str,
        bus: BusSystem,
        on_error: Callable[[BaseException], None],
        clock: Callable[[], float],
        tick_seconds: float | None,
        *,
        ghost: MemoryStore,
        working_target: WorkingTarget,
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
        """Forward the module configuration and retain the memory store."""
        super().__init__(
            module_id,
            bus,
            on_error,
            clock,
            tick_seconds,
            working_target=working_target,
            workflow=workflow,
            tools_factory=tools_factory,
            runtime_root=runtime_root,
            success_status=success_status,
            agent_model=agent_model,
            batch_provider=batch_provider,
            mcp_services_factory=mcp_services_factory,
            child_factory=child_factory,
            workflow_models=workflow_models,
        )
        self.ghost = ghost

    def prepare_input(self, batch: list[Envelope]) -> dict[str, Any]:
        # Resolve input through the invocation scope, never the unrestricted Ghost.
        tools = build_tools(self.last_environment)
        messages = []
        for envelope in batch:
            message = {
                "id": envelope.id,
                "source": envelope.source,
                "kind": envelope.kind,
                "topic": envelope.topic,
                "correlation_id": envelope.correlation_id,
                "causation_id": envelope.causation_id,
                "target": envelope.target,
                "payload": envelope.payload,
                "knoxel": tools["ghost_get"](envelope.payload["knoxel_id"]),
            }
            if "gwt_tick" in envelope.payload:
                message["gwt"] = {
                    "__type__": "GwtTurnRecord",
                    "data": tools["ghost_gwt"](envelope.payload["gwt_tick"]),
                }
            messages.append(message)
        return {
            "invocation_id": self.invocation_id,
            "module_id": self.module_id,
            "messages": messages,
        }
