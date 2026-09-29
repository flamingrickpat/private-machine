"""One durable managed-run table and its start/status/update/terminate domain.

The table is stored in Ghost's ``managed_runs`` mapping so it survives
checkpoints and an interrupted process. The same domain functions serve the
SelfManagement Python steps, the Assistant MCP tools, and the steward. No
module owns a second copy.

A run receives an accepted receipt before its worker starts. A repeated start
with the same action ID returns the original receipt instead of launching a
second worker. Unavailable workflows produce an explicit failed receipt.
Termination sets a code-owned cancellation signal; it never edits files.
Artifacts are written to a real task folder and preserved after cancellation.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from pm_workflows.runtime import WorkflowRuntime

from runtime.workflow_runner import (
    PROJECT_ROOT,
    WorkflowRequest,
    WorkflowResult,
    WorkflowRunner,
    WorkflowStatus,
)
from working_target import FolderEnvironment

STATES = frozenset({
    "accepted", "running", "waiting_for_input", "completed", "failed",
    "cancelled", "interrupted",
})
ACTIVE_STATES = frozenset({"accepted", "running", "waiting_for_input"})


class ManagedRunError(RuntimeError):
    """A management operation cannot complete as requested."""


class UnavailableWorkflow(ManagedRunError):
    """The run named a workflow the application does not manage."""


@dataclass(frozen=True)
class WorkflowSpec:
    """One managed workflow: its manifest and optional environment builder."""

    manifest: Path
    environment_factory: Callable[[str], Any] | None = None
    namespace: str = ""


@dataclass
class ManagedRunReceipt:
    """One row of the managed-run table. Serialized into Ghost verbatim."""

    run_id: str
    action_id: str
    workflow_name: str
    parent_correlation: str
    state: str
    latest_event: str
    artifact_path: str
    instructions: str = ""
    revision: int = 0
    question_id: str = ""
    answer: str = ""
    request_ids: list[str] = field(default_factory=list)
    arguments: dict = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError(f"Unknown managed-run state: {self.state!r}")

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "action_id": self.action_id,
            "workflow_name": self.workflow_name,
            "parent_correlation": self.parent_correlation,
            "state": self.state,
            "latest_event": self.latest_event,
            "artifact_path": self.artifact_path,
            "instructions": self.instructions,
            "revision": self.revision,
            "question_id": self.question_id,
            "answer": self.answer,
            "request_ids": list(self.request_ids),
            "arguments": dict(self.arguments),
            "evidence": list(self.evidence),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ManagedRunReceipt":
        known = {name: data[name] for name in cls.__dataclass_fields__ if name in data}
        return cls(**known)


class ManagedRunSupervisor:
    """Own the managed-run table, workers, and code-owned cancellation."""

    def __init__(
        self,
        ghost,
        *,
        runtime_root: Path,
        workflows: Mapping[str, WorkflowSpec],
        clock: Callable[[], float],
        environment_factory: Callable[[str], Any] | None = None,
        task_root: Path | None = None,
        base_dir: Path = PROJECT_ROOT,
        event_sink: Callable[[dict], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_terminal: Callable[[dict], None] | None = None,
        worker_tools: Callable[[str], Mapping[str, Callable]] | None = None,
    ) -> None:
        self.ghost = ghost
        self.runtime_root = Path(runtime_root)
        self.workflows = dict(workflows)
        self.clock = clock
        self.environment_factory = environment_factory
        self.task_root = Path(task_root) if task_root is not None else self.runtime_root / "managed"
        self.base_dir = Path(base_dir)
        self.event_sink = event_sink
        self.on_error = on_error
        self.on_terminal = on_terminal
        self.worker_tools = worker_tools
        self._lock = threading.RLock()
        self._cancels: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._pending: dict[str, dict] = {}

    # ------------------------------------------------------------- table

    def receipts(self) -> dict[str, dict]:
        with self.ghost.lock:
            return {
                run_id: dict(receipt)
                for run_id, receipt in self.ghost.managed_runs.items()
            }

    def get(self, run_id: str) -> dict:
        with self.ghost.lock:
            receipt = self.ghost.managed_runs.get(run_id)
            if receipt is None:
                raise ManagedRunError(f"No managed run {run_id!r}.")
            return dict(receipt)

    def _find_by_action(self, action_id: str) -> dict | None:
        if not action_id:
            return None
        with self.ghost.lock:
            for receipt in self.ghost.managed_runs.values():
                if receipt.get("action_id") == action_id:
                    return dict(receipt)
        return None

    def _persist(self, receipt: ManagedRunReceipt) -> dict:
        """Serialize one table write. Retries a stale batch once."""
        with self._lock:
            for _attempt in range(2):
                batch = self.ghost.begin_batch(
                    f"managed-run:{receipt.run_id}:{uuid.uuid4().hex}",
                )
                batch.set_run_receipt(receipt.run_id, receipt.to_dict())
                try:
                    self.ghost.commit(batch)
                    return receipt.to_dict()
                except Exception:
                    if _attempt == 1:
                        raise
            raise ManagedRunError("The managed-run table did not accept the receipt.")

    def _transition(
        self,
        run_id: str,
        *,
        state: str,
        latest_event: str,
        **changes: Any,
    ) -> dict:
        current = ManagedRunReceipt.from_dict(self.get(run_id))
        current.state = state
        current.latest_event = latest_event
        current.updated_at = self.clock()
        for name, value in changes.items():
            setattr(current, name, value)
        return self._persist(current)

    # ------------------------------------------------------------ start

    def start(
        self,
        *,
        action_id: str,
        workflow_name: str,
        parent_correlation: str = "",
        instructions: str = "",
        artifact_path: str = "",
        run_id: str | None = None,
        arguments: dict | None = None,
    ) -> dict:
        """Persist the accepted identity, then launch one worker.

        A repeated action ID returns the original receipt. An unknown workflow
        produces an explicit failed receipt rather than a silent no-op.
        """
        existing = self._find_by_action(action_id)
        if existing is not None:
            return existing
        if workflow_name not in self.workflows:
            receipt = ManagedRunReceipt(
                run_id=run_id or uuid.uuid4().hex,
                action_id=action_id,
                workflow_name=workflow_name,
                parent_correlation=parent_correlation,
                state="failed",
                latest_event=f"workflow {workflow_name!r} is not managed",
                artifact_path=artifact_path,
                instructions=instructions,
                arguments=dict(arguments or {}),
                created_at=self.clock(),
                updated_at=self.clock(),
            )
            return self._persist(receipt)

        run_id = run_id or uuid.uuid4().hex
        if not artifact_path:
            artifact_path = str(self.task_root / run_id)
        receipt = ManagedRunReceipt(
            run_id=run_id,
            action_id=action_id,
            workflow_name=workflow_name,
            parent_correlation=parent_correlation,
            state="accepted",
            latest_event="accepted",
            artifact_path=artifact_path,
            instructions=instructions,
            arguments=dict(arguments or {}),
            created_at=self.clock(),
            updated_at=self.clock(),
            evidence=[artifact_path],
        )
        persisted = self._persist(receipt)
        self._launch(receipt, resume=False, payload=None)
        return persisted

    def _launch(self, receipt: ManagedRunReceipt, *, resume: bool, payload: dict | None) -> None:
        cancel = self._cancels.setdefault(receipt.run_id, threading.Event())
        thread = threading.Thread(
            target=self._worker,
            args=(receipt, resume, payload),
            name=f"managed-{receipt.run_id[:8]}",
            daemon=True,
        )
        with self._lock:
            self._threads[receipt.run_id] = thread
        thread.start()

    # ----------------------------------------------------------- worker

    def _worker(self, receipt: ManagedRunReceipt, resume: bool, payload: dict | None) -> None:
        cancel = self._cancels[receipt.run_id]
        try:
            if not resume:
                self._transition(
                    receipt.run_id, state="running", latest_event="workflow started",
                    question_id="",
                )
                self._emit(receipt.run_id, "started", receipt=receipt.to_dict())
            spec = self.workflows[receipt.workflow_name]
            environment = self._environment(receipt)
            if payload is None:
                payload = self._request_payload(receipt)
            tools = dict(self.worker_tools(receipt.run_id)) if self.worker_tools else {}
            runtime = WorkflowRuntime(
                environment=environment,
                stop_event=cancel,
                tools=tools,
                propagate_python_errors=True,
            )
            runner = WorkflowRunner(
                spec.manifest,
                self.runtime_root,
                namespace=spec.namespace or receipt.workflow_name,
                base_dir=self.base_dir,
            )
            if resume:
                runner.write_answer(
                    receipt.run_id,
                    str(payload.get("answer", "")),
                    question_id=str(payload.get("question_id", "")),
                    resume_phase=str(payload.get("resume_phase", "")),
                )
            result = runner.invoke(
                WorkflowRequest(receipt.run_id, payload), runtime, resume=resume,
            )
            self._finish(receipt.run_id, result, environment)
        except BaseException as exc:  # a worker crash must become a failed receipt
            try:
                self._transition(
                    receipt.run_id, state="failed",
                    latest_event=f"worker error: {type(exc).__name__}: {exc}"[-500:],
                )
                self._emit(receipt.run_id, "failed", error=str(exc))
            except BaseException:
                pass
            if self.on_error is not None:
                self.on_error(exc)
            self._notify_terminal(receipt.run_id)

    # ----------------------------------------------------------- terminal

    def _notify_terminal(self, run_id: str) -> None:
        """Offer one terminal receipt to the optional consolidation hook."""
        if self.on_terminal is None:
            return
        try:
            self.on_terminal(self.get(run_id))
        except Exception:
            # Consolidation is not allowed to corrupt the run receipt.
            pass

    def _environment(self, receipt: ManagedRunReceipt) -> Any:
        if self.environment_factory is not None:
            return self.environment_factory(receipt.run_id)
        return FolderEnvironment(Path(receipt.artifact_path), "bash")

    def _request_payload(self, receipt: ManagedRunReceipt) -> dict:
        return {
            "run_id": receipt.run_id,
            "instruction": receipt.instructions,
            "artifact_path": receipt.artifact_path,
            "answer": "",
            **dict(receipt.arguments),
        }

    def _finish(self, run_id: str, result: WorkflowResult, environment: Any) -> None:
        if result.status is WorkflowStatus.Cancelled:
            self._transition(
                run_id, state="cancelled", latest_event="cancelled",
                evidence=self._artifact_evidence(run_id, environment),
            )
            self._emit(run_id, "cancelled")
            self._notify_terminal(run_id)
            return
        if result.status is WorkflowStatus.WaitingForInput:
            with self._lock:
                self._pending[run_id] = {
                    "question_id": result.question_id,
                    "resume_phase": result.resume_phase,
                    "environment": environment,
                }
            self._transition(
                run_id, state="waiting_for_input",
                latest_event=f"waiting for answer: {result.question}",
                question_id=str(result.question_id or ""),
            )
            self._emit(run_id, "waiting_for_input", question_id=result.question_id,
                       question=result.question)
            return
        if result.status is WorkflowStatus.Completed:
            self._transition(run_id, state="completed", latest_event="completed",
                             evidence=self._artifact_evidence(run_id, environment))
            self._emit(run_id, "completed")
            self._notify_terminal(run_id)
            return
        self._transition(run_id, state="failed",
                         latest_event=f"workflow failed: {result.kernel_summary}"[-500:],
                         evidence=self._artifact_evidence(run_id, environment))
        self._emit(run_id, "failed", summary=result.kernel_summary)
        self._notify_terminal(run_id)

    def _artifact_evidence(self, run_id: str, environment: Any) -> list[str]:
        """List real files produced by the run without reading their content."""
        paths: list[str] = []
        for directory in ("outbox", "scratch"):
            try:
                paths.extend(environment.list_files(directory))
            except Exception:
                continue
        return paths

    def _emit(self, run_id: str, event: str, **fields: Any) -> None:
        if self.event_sink is None:
            return
        self.event_sink({"run_id": run_id, "event": event, **fields})

    # -------------------------------------------------------- operations

    def status(self, run_id: str) -> dict:
        return self.get(run_id)

    def update(self, run_id: str, instructions: str, *, request_id: str = "") -> dict:
        """Queue one instruction revision. Duplicate request IDs are idempotent."""
        current = ManagedRunReceipt.from_dict(self.get(run_id))
        if request_id and request_id in current.request_ids:
            return current.to_dict()
        if current.state not in ACTIVE_STATES:
            raise ManagedRunError(
                f"Run {run_id!r} is {current.state} and cannot accept an update."
            )
        if not instructions.strip():
            raise ManagedRunError("An update needs nonempty instruction text.")
        current.instructions = instructions
        current.revision += 1
        current.latest_event = f"instructions updated to revision {current.revision}"
        current.updated_at = self.clock()
        if request_id:
            current.request_ids.append(request_id)
        receipt = self._persist(current)
        self._emit(run_id, "updated", revision=current.revision)
        return receipt

    def answer(self, run_id: str, question_id: str, answer: str) -> dict:
        """Deliver a correlated answer and resume the suspended run."""
        current = ManagedRunReceipt.from_dict(self.get(run_id))
        with self._lock:
            pending = self._pending.get(run_id)
        if current.state != "waiting_for_input" or pending is None:
            raise ManagedRunError(f"Run {run_id!r} is not waiting for input.")
        if question_id and question_id != current.question_id:
            raise ManagedRunError(
                f"Question {question_id!r} does not match {current.question_id!r}."
            )
        if not answer.strip():
            raise ManagedRunError("An answer needs nonempty text.")
        current.answer = answer
        current.state = "running"
        current.latest_event = "answer received; resuming"
        current.updated_at = self.clock()
        receipt = self._persist(current)
        payload = {
            "run_id": run_id,
            "instruction": current.instructions,
            "artifact_path": current.artifact_path,
            "answer": answer,
            "question_id": current.question_id,
            "resume_phase": pending["resume_phase"],
            **dict(current.arguments),
        }
        self._launch(current, resume=True, payload=payload)
        self._emit(run_id, "answered", question_id=current.question_id)
        return receipt

    def terminate(self, run_id: str) -> dict:
        """Cancel a run through its code-owned signal. Files are untouched."""
        current = ManagedRunReceipt.from_dict(self.get(run_id))
        if current.state not in ACTIVE_STATES:
            return current.to_dict()
        with self._lock:
            cancel = self._cancels.get(run_id)
        if cancel is not None:
            cancel.set()
        receipt = self._transition(
            run_id, state="cancelled", latest_event="terminated by request",
        )
        self._emit(run_id, "terminated")
        return receipt

    def join(self, run_id: str, timeout: float | None = None) -> None:
        with self._lock:
            thread = self._threads.get(run_id)
        if thread is not None:
            thread.join(timeout)

    def join_all(self, timeout: float | None = None) -> None:
        for run_id in list(self._threads):
            self.join(run_id, timeout)


__all__ = [
    "ACTIVE_STATES",
    "STATES",
    "ManagedRunError",
    "ManagedRunReceipt",
    "ManagedRunSupervisor",
    "UnavailableWorkflow",
    "WorkflowSpec",
]
