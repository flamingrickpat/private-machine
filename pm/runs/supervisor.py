"""One durable managed-run table and its start/status/update/terminate domain.

The table is stored in MemoryStore's ``managed_runs`` mapping so it survives
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
import asyncio
from concurrent.futures import CancelledError as FutureCancelledError
from copy import deepcopy
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from pm_workflows.runtime import WorkflowRuntime

from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowResult, WorkflowRunner, WorkflowStatus
from pm.host.folder_target import FolderEnvironment
from pm.llm.workflow_sessions import WorkflowModels
from pm.host.durable_files import ProcessLock, BodyBusy, atomic_json, read_json
from pm.runs.run_receipt import ACTIVE_STATES, ManagedRunError, ManagedRunReceipt
from pm.runs.workflow_spec import WorkflowSpec, scoped_children
from pm.memory.agent_tools import guard_revisions


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
        workflow_models: WorkflowModels | None = None,
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
        self.workflow_models = workflow_models
        self._lock = threading.RLock()
        self._stopping = threading.Event()
        self._updates = threading.Condition(self._lock)
        self._cancels: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._pending: dict[str, dict] = {}
        self._worker_locks: dict[str, ProcessLock] = {}
        self._checkpoint_stamp: int | None = None
        self._checkpoint_runs: dict[str, dict] = {}

    # ------------------------------------------------------------- table

    def receipts(self) -> dict[str, dict]:
        with self.ghost.lock:
            ids = list(self.ghost.managed_runs)
        return {run_id: self.get(run_id) for run_id in ids}

    def get(self, run_id: str) -> dict:
        with self._lock:
            return self._read_control(run_id)

    def _read_control(self, run_id: str) -> dict:
        """Read the newer accepted receipt under the supervisor mutex."""
        with self.ghost.lock:
            receipt = self.ghost.managed_runs.get(run_id)
            if receipt is None:
                raise ManagedRunError(f"No managed run {run_id!r}.")
            # Control edits must enter a Ghost batch. Nested event acknowledgments
            # and request IDs cannot mutate the accepted in-memory row first.
            receipt = deepcopy(receipt)
        if receipt.get("artifact_path"):
            path = Path(receipt["artifact_path"]) / "control.json"
            if path.is_file():
                durable = read_json(path)
                if durable.get("control_version", 0) > receipt.get("control_version", 0):
                    receipt = durable
        # Ghost commits before its mirror. A different process can die between
        # those writes. Read a changed checkpoint before authorizing old work.
        # Cache only its managed rows; unchanged calls require one stat.
        checkpoint = self.ghost.checkpoint
        if checkpoint is not None and checkpoint.is_file():
            stamp = checkpoint.stat().st_mtime_ns
            if stamp != self._checkpoint_stamp:
                self._checkpoint_runs = read_json(checkpoint).get("managed_runs", {})
                self._checkpoint_stamp = stamp
            committed = self._checkpoint_runs.get(run_id)
            if committed and committed.get("control_version", 0) > receipt.get("control_version", 0):
                receipt = deepcopy(committed)
        return receipt

    def _find_by_action(self, action_id: str) -> dict | None:
        if not action_id:
            return None
        with self.ghost.lock:
            for receipt in self.ghost.managed_runs.values():
                if receipt.get("action_id") == action_id:
                    return self.get(receipt["run_id"])
        return None

    def _persist(self, receipt: ManagedRunReceipt) -> dict:
        """Commit MemoryStore, then mirror control for other processes.

        The file mutex rejects a stale control write. A crash between commits
        leaves the accepted MemoryStore receipt as the restart source.
        """
        with self._lock:
            if receipt.artifact_path:
                path = Path(receipt.artifact_path) / "control.json"
                lock = ProcessLock(path.with_suffix(".lock")).acquire(wait=True)
            else:
                path, lock = None, None
            try:
                with self.ghost.lock:
                    known = receipt.run_id in self.ghost.managed_runs
                if known and self._read_control(receipt.run_id).get("control_version", 0) > receipt.control_version:
                    raise ManagedRunError("The accepted Ghost control receipt changed. Read it again before writing.")
                if path is not None and path.exists():
                    durable = read_json(path)
                    if durable.get("control_version", 0) > receipt.control_version:
                        raise ManagedRunError("The managed control receipt changed. Read it again before writing.")
                receipt.control_version += 1
                result = self._commit_receipt(receipt)
                if path is not None:
                    atomic_json(path, result)
                return result
            finally:
                if lock is not None:
                    lock.close()

    def _commit_receipt(self, receipt: ManagedRunReceipt) -> dict:
        """Retry a MemoryStore revision conflict once without launching external work."""
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
        with self._lock:
            current = ManagedRunReceipt.from_dict(self.get(run_id))
            if current.state not in ACTIVE_STATES and state != current.state:
                return current.to_dict()
            current.state = state
            current.latest_event = latest_event
            current.updated_at = self.clock()
            for name, value in changes.items():
                setattr(current, name, value)
            return self._persist(current)

    # ------------------------------------------------------------ start

    def _singleton_key(self, workflow_name: str) -> str:
        """Return the configured admission key, or an empty key for parallel work."""
        spec = self.workflows.get(workflow_name)
        return (spec.singleton_key or workflow_name) if spec is not None and spec.singleton else ""

    def _singleton_owner(self, workflow_name: str, *, exclude: str = "") -> dict | None:
        """Read the owner under the supervisor mutex, including worker cleanup.

        The accepted receipt reserves the slot before thread launch. Cancellation
        changes the receipt before cleanup ends, so the worker lock also counts.
        This admission policy belongs to one supervisor. External body leases
        retain their independent operating-system fence across processes.
        """
        key = self._singleton_key(workflow_name)
        if not key:
            return None
        for run_id, run in self.receipts().items():
            if run_id == exclude:
                continue
            owner_key = run.get("singleton_key") or self._singleton_key(run["workflow_name"])
            if owner_key == key and (run["state"] in ACTIVE_STATES or run_id in self._worker_locks):
                return run
        return None

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
        A busy singleton retains a failed request receipt and publishes its
        current owner to cognition. It launches no worker and cancels no owner.
        """
        with self._lock:
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
            singleton_key = self._singleton_key(workflow_name)
            owner = self._singleton_owner(workflow_name)
            if owner is not None:
                detail = (f"Workflow {workflow_name} is already in use by run {owner['run_id']} "
                          f"({owner['workflow_name']}). This start did not launch a worker. "
                          "Choose whether to wait, update that run, or terminate it.")
                receipt = ManagedRunReceipt(run_id=run_id, action_id=action_id,
                    workflow_name=workflow_name, parent_correlation=parent_correlation,
                    state="failed", latest_event=detail, artifact_path="",
                    instructions=instructions, arguments=dict(arguments or {}),
                    created_at=self.clock(), updated_at=self.clock(),
                    singleton_key=singleton_key, blocked_by_run_id=owner["run_id"])
                self._persist(receipt)
                self._emit(run_id, "start_rejected", reason="singleton_busy", detail=detail,
                           singleton_key=singleton_key, blocked_by_run_id=owner["run_id"],
                           owner_workflow=owner["workflow_name"], owner_state=owner["state"])
                return self.get(run_id)
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
                singleton_key=singleton_key,
            )
            persisted = self._persist(receipt)
            self._launch(receipt, resume=False, payload=None)
            return persisted

    def _launch(self, receipt: ManagedRunReceipt, *, resume: bool, payload: dict | None) -> None:
        if self._stopping.is_set():
            self._transition(receipt.run_id, state="cancelled", latest_event="application stopping")
            return
        if receipt.run_id not in self._worker_locks:
            try:
                self._worker_locks[receipt.run_id] = ProcessLock(Path(receipt.artifact_path) / "worker.lock").acquire()
            except BodyBusy as error:
                raise ManagedRunError("Another process owns this managed worker.") from error
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
                self._emit(receipt.run_id, "started", receipt={key: value for key, value in receipt.to_dict().items() if key != "events"})
            spec = self.workflows[receipt.workflow_name]
            environment = self._environment(receipt)
            if payload is None:
                payload = self._request_payload(receipt)
            if spec.tools_factory is not None:
                tools = dict(spec.tools_factory(receipt.run_id, environment))
            elif self.worker_tools is not None:
                tools = dict(self.worker_tools(receipt.run_id))
            else:
                tools = {}
            tools["managed_instructions"] = lambda wait=False, close=False: self.instructions_at_boundary(
                receipt.run_id, wait=wait, close=close, epoch=receipt.execution_epoch,
            )
            tools["managed_authorize"] = lambda revision, commit: self.authorize_dispatch(
                receipt.run_id, receipt.execution_epoch, revision, commit)
            tools["managed_instructions"].authorize = tools["managed_authorize"]
            tools["managed_instructions"].cancelled = lambda: self.cancelled_at_boundary(
                receipt.run_id, receipt.execution_epoch)
            tools["managed_milestone"] = lambda key, summary, evidence: self.milestone(
                receipt.run_id, receipt.execution_epoch, key, summary, evidence)
            if spec.memory_config_factory is not None:
                tools["memory_config"] = lambda: spec.memory_config_factory(self.ghost, cancel)
                tools["memory_guard"] = lambda revisions: guard_revisions(self.ghost, revisions, cancel)
            runtime = WorkflowRuntime(
                environment=environment,
                stop_event=cancel,
                tools=tools,
                propagate_python_errors=True,
                agent_model=spec.agent_model,
                mcp_servers=spec.mcp_servers,
            )
            scoped_children(runtime, Path(receipt.artifact_path))
            runner = WorkflowRunner(
                spec.manifest,
                self.runtime_root if receipt.execution_epoch == 1 else
                self.runtime_root / "recovery" / receipt.run_id / f"epoch-{receipt.execution_epoch:04d}",
                namespace=spec.namespace or receipt.workflow_name,
                base_dir=self.base_dir,
                models=self.workflow_models,
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
            from pm.memory.snapshot import GhostCancelled
            cancelled = cancel.is_set() and isinstance(exc, (GhostCancelled, asyncio.CancelledError, FutureCancelledError))
            recording_failed = False
            try:
                current = self.get(receipt.run_id)
                if current["state"] in ACTIVE_STATES:
                    self._transition(
                        receipt.run_id, state="cancelled" if cancelled else "failed",
                        latest_event=f"worker error: {type(exc).__name__}: {exc}"[-500:],
                    )
                    self._emit(receipt.run_id, "failed", error=str(exc), detail=str(exc),
                               reason="resource_busy" if isinstance(exc, BodyBusy) else "worker_error")
                else:
                    # Publication can fail after a terminal receipt commits.
                    # Keep that actual outcome and identify the delivery error.
                    self._emit(receipt.run_id, "publication_interrupted", error=str(exc))
            except BaseException as recording_error:
                recording_failed = True
                exc.add_note(f"Recording the worker failure also failed: {type(recording_error).__name__}: {recording_error}")
            from pm.runs.errors import TaskServiceError
            # Service loss or resource contention ends only this task. Keep the
            # fence intact and let cognition choose the next action. Logic faults
            # still reach the application's error handler.
            if recording_failed or (not cancelled and not isinstance(exc, (TaskServiceError, BodyBusy))):
                if self.on_error is None:
                    raise
                self.on_error(exc)
            self._notify_terminal(receipt.run_id)
        finally:
            with self._lock:
                lock = self._worker_locks.pop(receipt.run_id, None)
            if lock is not None:
                lock.close()

    # ----------------------------------------------------------- terminal

    def _notify_terminal(self, run_id: str) -> None:
        """Offer one terminal receipt to the optional consolidation hook."""
        if self.on_terminal is None:
            return
        try:
            self.on_terminal(self.get(run_id))
        except Exception as error:
            # Keep the task outcome intact and expose the separate hook failure.
            if self.on_error is None:
                raise
            self.on_error(error)

    def _environment(self, receipt: ManagedRunReceipt) -> Any:
        spec = self.workflows[receipt.workflow_name]
        if spec.environment_factory is not None:
            return spec.environment_factory(receipt.run_id)
        if self.environment_factory is not None:
            return self.environment_factory(receipt.run_id)
        return FolderEnvironment(Path(receipt.artifact_path), "bash")

    def _request_payload(self, receipt: ManagedRunReceipt) -> dict:
        payload = {
            **dict(receipt.arguments),
            "run_id": receipt.run_id,
            "instruction": receipt.instructions,
            "artifact_path": receipt.artifact_path,
            "answer": "",
        }
        spec = self.workflows.get(receipt.workflow_name)
        if spec is not None and spec.request_context is not None:
            with self.ghost.lock:
                payload.update(spec.request_context(self.ghost))
        return payload

    def _finish(self, run_id: str, result: WorkflowResult, environment: Any) -> None:
        with self._lock:
            if result.status is WorkflowStatus.Cancelled or self.get(run_id)["state"] == "cancelled":
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
                if result.report.get("accepted") is False:
                    self._transition(
                        run_id, state="failed",
                        latest_event=str(result.report.get("rejection_reason") or "workflow rejected its result"),
                        evidence=self._artifact_evidence(run_id, environment),
                    )
                    self._emit(run_id, "failed")
                    self._notify_terminal(run_id)
                    return
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
        """Commit an event before publication. Consumers acknowledge its identity."""
        with self._lock:
            current = ManagedRunReceipt.from_dict(self.get(run_id))
            identity = f"{run_id}:{current.execution_epoch}:{len(current.events) + 1}"
            message = {"run_id": run_id, "event": event, "event_id": identity,
                       "execution_epoch": current.execution_epoch, **fields}
            current.events.append({"message": message, "published": False})
            self._persist(current)
        if self.event_sink is not None:
            self.event_sink(message)

    def acknowledge_event(self, run_id: str, event_id: str) -> None:
        """Mark a consumed event. A repeated acknowledgment changes no identity."""
        with self._lock:
            current = ManagedRunReceipt.from_dict(self.get(run_id))
            row = next((row for row in current.events if row["message"]["event_id"] == event_id), None)
            if row is not None and not row["published"]:
                row["published"] = True
                self._persist(current)

    def milestone(self, run_id: str, epoch: int, key: str, summary: str, evidence: str) -> None:
        """Publish one retained attempt boundary to cognition with stable identity.

        Repeated delivery keeps the original event. Stale workers cannot publish
        new progress. A milestone reports evidence, never verified completion.
        """
        with self._lock:
            current = ManagedRunReceipt.from_dict(self.get(run_id))
            if current.execution_epoch != epoch or current.state != "running":
                raise ManagedRunError("This worker no longer owns progress publication.")
            if any(row["message"].get("milestone_key") == key for row in current.events):
                return
            identity = f"{run_id}:{epoch}:{len(current.events) + 1}"
            message = {"run_id": run_id, "event": "gameplay_progress", "event_id": identity,
                       "execution_epoch": epoch, "milestone_key": key, "detail": summary, "evidence": evidence}
            current.events.append({"message": message, "published": False})
            self._persist(current)
        # The sink can commit Ghost memory. Keep that commit outside this lock
        # so another cognition thread can inspect the managed receipt.
        if self.event_sink is not None:
            self.event_sink(message)

    def replay_events(self) -> None:
        """Publish pending progress with the original event IDs, without a worker."""
        if self.event_sink is not None:
            for receipt in self.receipts().values():
                for row in receipt.get("events", []):
                    if not row["published"]:
                        self.event_sink(row["message"])

    # -------------------------------------------------------- operations

    def status(self, run_id: str) -> dict:
        return self.get(run_id)

    def update(self, run_id: str, instructions: str, *, request_id: str = "") -> dict:
        """Queue one instruction revision. Duplicate request IDs are idempotent."""
        with self._lock:
            current = ManagedRunReceipt.from_dict(self.get(run_id))
            if request_id and request_id in current.request_ids:
                return current.to_dict()
            if current.state not in ACTIVE_STATES:
                raise ManagedRunError(
                    f"Run {run_id!r} is {current.state} and cannot accept an update."
                )
            if not current.updates_open:
                raise ManagedRunError(f"Run {run_id!r} passed its final instruction boundary.")
            if not instructions.strip():
                raise ManagedRunError("An update needs nonempty instruction text.")
            current.instructions = instructions
            current.revision += 1
            current.latest_event = f"instruction revision {current.revision} queued"
            current.updated_at = self.clock()
            if request_id:
                current.request_ids.append(request_id)
            receipt = self._persist(current)
            self._updates.notify_all()
            self._emit(run_id, "updated", revision=current.revision)
            return receipt

    def instructions_at_boundary(self, run_id: str, *, wait: bool = False, close: bool = False,
                                 epoch: int | None = None) -> dict:
        """Consume the latest revision on the worker thread, or wait for one.

        Closing the final boundary rejects subsequent updates. The receipt
        distinguishes a queued revision from a revision consumed by the role.
        """
        from pm_workflows.runtime import WorkflowTerminated

        with self._updates:
            if wait:
                while (not self._cancels[run_id].is_set()
                       and self.get(run_id)["state"] != "cancelled"
                       and self.get(run_id)["revision"] <= self.get(run_id)["applied_revision"]):
                    self._updates.wait(0.1)
            if self._cancels[run_id].is_set():
                raise WorkflowTerminated()
            current = ManagedRunReceipt.from_dict(self.get(run_id))
            if current.state == "cancelled" or (epoch is not None and current.execution_epoch != epoch):
                self._cancels[run_id].set()
                raise WorkflowTerminated()
            updates_open = current.updates_open and not close
            if current.applied_revision == current.revision and current.updates_open == updates_open:
                # Checking an unchanged boundary must not rewrite the full
                # companion checkpoint before every Minecraft tool call.
                return {"instruction": current.instructions, "revision": current.revision}
            current.applied_revision = current.revision
            current.updates_open = updates_open
            current.latest_event = f"instruction revision {current.revision} consumed at boundary"
            current.updated_at = self.clock()
            self._persist(current)
            return {"instruction": current.instructions, "revision": current.revision}

    def authorize_dispatch(self, run_id: str, epoch: int, revision: int, commit: Callable) -> bool:
        """Order a durable dispatch marker against accepted control changes.

        The marker defines the start of an in-flight call. Later cancellation
        retains its result. A superseded revision cannot obtain a new marker.
        """
        with self._lock:
            root = Path(self.get(run_id)["artifact_path"])
            lock = ProcessLock(root / "control.lock").acquire(wait=True)
            try:
                current = self.get(run_id)
                if current["state"] == "cancelled" or current.get("execution_epoch", 1) != epoch:
                    from pm_workflows.runtime import WorkflowTerminated
                    raise WorkflowTerminated()
                if current["revision"] != revision:
                    return False
                commit()
                return True
            finally:
                lock.close()

    def cancelled_at_boundary(self, run_id: str, epoch: int) -> bool:
        """Poll durable cancellation during a long external call.

        This read consumes no instruction revision. A control accepted by a
        different process still reaches the active worker's stop watcher.
        """
        with self._lock:
            current = self.get(run_id)
            cancelled = current["state"] == "cancelled" or current.get("execution_epoch", 1) != epoch
            if cancelled:
                self._cancels[run_id].set()
            return cancelled

    def resume(self, run_id: str, *, request_id: str = "") -> dict:
        """Explicitly resume retained body intent with a new worker epoch.

        Cancelled, failed, and interrupted body runs retain their identity.
        No old kernel cursor or uncertain action is replayed. The workflow
        starts at its observation boundary with the existing task evidence.
        """
        with self._lock:
            current = ManagedRunReceipt.from_dict(self.get(run_id))
            if request_id and request_id in current.resume_requests:
                return current.to_dict()
            spec = self.workflows.get(current.workflow_name)
            if spec is None or not spec.resumable:
                raise ManagedRunError("Explicit recovery is available only for body workflows.")
            if current.state not in {"interrupted", "cancelled", "failed"}:
                raise ManagedRunError(f"Run {run_id!r} is {current.state} and cannot resume.")
            thread = self._threads.get(run_id)
            if thread is not None and thread.is_alive():
                raise ManagedRunError("The previous worker still needs external cleanup.")
            owner = self._singleton_owner(current.workflow_name, exclude=run_id)
            if owner is not None:
                detail = (f"Workflow {current.workflow_name} is already in use by run {owner['run_id']}. "
                          "Recovery did not launch a worker. Wait, update that run, or terminate it.")
                self._emit(run_id, "resume_rejected", reason="singleton_busy", detail=detail,
                           blocked_by_run_id=owner["run_id"], singleton_key=self._singleton_key(current.workflow_name))
                raise ManagedRunError(detail)
            try:
                lock = ProcessLock(Path(current.artifact_path) / "worker.lock").acquire()
            except BodyBusy as error:
                raise ManagedRunError("Another process owns this managed worker.") from error
            try:
                current.interrupted_from = current.state
                current.singleton_key = self._singleton_key(current.workflow_name)
                current.state, current.updates_open = "accepted", True
                current.execution_epoch += 1
                current.latest_event = "explicit resume accepted; fresh observation required"
                current.updated_at = self.clock()
                if request_id:
                    current.resume_requests.append(request_id)
                receipt = self._persist(current)
                self._cancels[run_id] = threading.Event()
                self._worker_locks[run_id] = lock
                self._launch(current, resume=False, payload=None)
                return receipt
            except BaseException:
                lock.close()
                self._worker_locks.pop(run_id, None)
                raise

    def answer(self, run_id: str, question_id: str, answer: str) -> dict:
        """Deliver a correlated answer and resume the suspended run."""
        with self._lock:
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
                **dict(current.arguments),
                "run_id": run_id,
                "instruction": current.instructions,
                "artifact_path": current.artifact_path,
                "answer": answer,
                "question_id": current.question_id,
                "resume_phase": pending["resume_phase"],
            }
            self._launch(current, resume=True, payload=payload)
            self._emit(run_id, "answered", question_id=current.question_id)
            return receipt

    def terminate(self, run_id: str, *, request_id: str = "") -> dict:
        """Cancel through the owned signal and retain a stable request ID.

        Replaying an acknowledged cancel must not cancel a later resumed epoch.
        Direct callers can omit the ID for a new, unconditional cancellation.
        """
        with self._lock:
            current = ManagedRunReceipt.from_dict(self.get(run_id))
            if request_id and request_id in current.request_ids:
                return current.to_dict()
            if request_id:
                current.request_ids.append(request_id)
            if current.state not in ACTIVE_STATES:
                return self._persist(current) if request_id else current.to_dict()
            with self._lock:
                cancel = self._cancels.get(run_id)
            if cancel is not None:
                cancel.set()
            self._updates.notify_all()
            receipt = self._transition(
                run_id, state="cancelled", latest_event="terminated by request",
                request_ids=current.request_ids,
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

    def stop(self) -> None:
        """Fence new workers and signal cancellation without waiting for I/O.

        The application can call this method from its error thread. Shutdown
        joins module threads first, then records waiting-run cancellation and
        joins managed workers through shutdown().
        """
        self._stopping.set()
        with self._lock:
            for cancel in self._cancels.values():
                cancel.set()
            self._updates.notify_all()

    def shutdown(self) -> None:
        """Cancel active runs and wait for worker cleanup before services stop.

        Call after the application joins its bus modules. Waiting runs also
        receive durable cancellation. External effects remain in their journals.
        Repeated calls retain terminal outcomes and do not restart work.
        """
        self.stop()
        for run_id, receipt in self.receipts().items():
            if receipt["state"] in ACTIVE_STATES:
                self.terminate(run_id)
        self.join_all()

