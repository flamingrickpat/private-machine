"""Run one instance. Exchange JSON commands and events with the console.

The worker is the child process of one console instance. It builds the
companion assembly from the deployment settings, runs the application, and
speaks a line-oriented JSON protocol on stdout: worker events go out, commands
come in, and every command gets one ``ack`` or ``rejected`` line carrying its
request ID. Dependency output is redirected to stderr so it cannot corrupt the
stream.

The worker also instruments three seams for the console page: workflow
invocations, model completions, and module batches. That telemetry is
observation only; it never changes a workflow result or a reply. Diagnostics
go to ``root/logs``, and the process stops only through a command, a signal,
or the application's first error.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from threading import Event
import json
import asyncio
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING
from pm.utils import yaml_util
from pathlib import Path
from uuid import uuid4

from pm.bus.envelopes import Envelope
from pm.console.assembly import DEFAULT_ASSEMBLY, DEFAULT_OPERATOR_HOOKS, load
from pm.host.application import BusModule
from pm.memory.yaml_codec import RecordPayload
from pm.runs.workflow_runner import WorkflowRequest, WorkflowResult, WorkflowRunner
from pm.llm.activation import ChatClient, ChatResult
from pm.llm.prompt_encoding import TurnInput

if TYPE_CHECKING:
    from pm_workflows.runtime import WorkflowRuntime

    from pm.console.server import InstanceSettings


def run(settings: InstanceSettings, root: Path) -> None:
    """Own one runtime, with serialized chat submission and durable edits.

    The command loop is the only writer of chat submissions: one pending
    exchange at a time, because the reply path and the user name are shared.
    A command that the runtime cannot honour produces a ``rejected`` line and
    keeps the worker running; a crash reaches the application error path.
    """
    from pm.host.diagnostics import configure_logging
    from pm.host.processes import ensure_process_cleanup
    configure_logging(root)
    ensure_process_cleanup()
    from pm.llm.prompt_log import backfill_cache_logs
    backfill_cache_logs()
    output = sys.stdout
    sys.stdout = sys.stderr  # Dependency output must not corrupt the event stream.
    lock = threading.Lock()
    context = threading.local()

    def emit(kind: str, **fields: object) -> None:
        """Write one protocol line. Values that are not JSON become their text."""
        with lock:
            output.write(json.dumps({"kind": kind, "time": time.time(), **fields}, default=str) + "\n")
            output.flush()

    original_invoke = WorkflowRunner.invoke

    def invoke(runner: WorkflowRunner, request: WorkflowRequest, runtime: "WorkflowRuntime",
               *, resume: bool = False) -> WorkflowResult:
        """Publish the start, end, and error of one workflow invocation.

        The wrapper repeats the runner's own signature, so a caller keeps the
        ``resume`` contract and the console sees the same fields either way.
        """
        started = time.perf_counter()
        previous = getattr(context, "workflow", None)
        fields = {"node": runner.namespace, "run_id": request.run_id,
                  "workflow": str(runner.workflow), "input": request.payload,
                  "owner_module": threading.current_thread().name,
                  "parent_run": previous["run_id"] if previous else None,
                  "parent_node": previous["node"] if previous else None,
                  "correlation_id": request.payload.get("correlation_id") or getattr(context, "correlation", "")}
        manifest = runner.workflow.read_text(encoding="utf-8").split("---", 2)
        definition = yaml_util.safe_load(manifest[1]) if len(manifest) == 3 else {}
        context.workflow = fields
        emit("workflow.start", **fields, definition=definition)
        try:
            result = original_invoke(runner, request, runtime, resume=resume)
            emit("workflow.end", **fields, seconds=time.perf_counter() - started,
                 status=str(result.status), report=result.report, artifacts=str(result.artifact_dir))
            return result
        except BaseException as exc:
            emit("workflow.error", **fields, seconds=time.perf_counter() - started, error=str(exc))
            raise
        finally:
            context.workflow = previous

    WorkflowRunner.invoke = invoke
    original_complete = ChatClient.complete

    async def complete(client: ChatClient, messages: Sequence[TurnInput], *,
                       cancel: Event | None = None, **generation: object) -> ChatResult:
        """Publish one model call without its message bodies at the end.

        A cancellation flag travels with the generation fields, so the console
        shows the same request the model received.
        """
        started = time.perf_counter()
        call_id = uuid4().hex
        workflow = getattr(context, "workflow", None)
        generation_fields: dict[str, object] = dict(generation)
        if cancel is not None:
            generation_fields["cancel"] = cancel
        fields = {"call_id": call_id, "node": workflow["node"] if workflow else threading.current_thread().name,
                  "owner_module": threading.current_thread().name,
                  "run_id": workflow["run_id"] if workflow else None,
                  "correlation_id": workflow["correlation_id"] if workflow else getattr(context, "correlation", ""),
                  "messages": messages, "generation": generation_fields}
        emit("model.start", **fields)
        try:
            result = await original_complete(client, messages, cancel=cancel, **generation)
            emit("model.end", **{k: v for k, v in fields.items() if k not in {"messages", "generation"}},
                 seconds=time.perf_counter() - started, result=result.model_dump(mode="json"))
            return result
        except BaseException as exc:
            emit("model.error", **{k: v for k, v in fields.items() if k not in {"messages", "generation"}},
                 seconds=time.perf_counter() - started, error=str(exc))
            raise

    ChatClient.complete = complete
    from pm.app.operator_config import operator_config
    model_config, environment, _ = operator_config(settings.get("config") or None)
    load(settings.get("operator_hooks") or DEFAULT_OPERATOR_HOOKS).prepare_environment(environment)
    build_app = load(settings.get("assembly") or DEFAULT_ASSEMBLY).build_app
    app = build_app(settings["seed"], root, checkpoint_path=root / "ghost.pickle",
                    cognition=settings["cognition"], config_path=settings.get("config") or None,
                    emit_reply=lambda text: None)
    pending: dict[str, str] = {
        item.metadata["exchange_correlation"]: item.metadata.get("session", "recovered")
        for item in app.ghost.all_features if item.metadata.get("exchange_correlation")
        and (item.metadata.get("exchange_status") == "pending"
             or app.ghost.get_knoxel_by_id(item.metadata.get("public_reply_id")) is not None
             and app.ghost.get_knoxel_by_id(item.metadata["public_reply_id"]).metadata.get("publication_status") == "pending")
    }
    search_cancel = threading.Event()
    submission_lock = threading.Lock()

    def observe(envelope: Envelope, recipient: str) -> None:
        """Mirror one bus envelope, including a public reply, into the event stream."""
        raw = json.loads(envelope.to_json())
        raw["envelope_kind"] = raw.pop("kind")
        emit("bus", recipient=recipient, **raw)
        if recipient == "user_input" and envelope.topic == "dialogue.reply":
            feature = app.ghost.get_knoxel_by_id(envelope.payload["knoxel_id"])
            if feature is not None:
                with submission_lock:
                    emit("chat.reply", session=pending.get(envelope.correlation_id, "unsolicited"),
                         correlation_id=envelope.correlation_id, knoxel_id=feature.id,
                         recovery_outcome=envelope.payload.get("recovery_outcome", ""), text=feature.content)
                    pending.pop(envelope.correlation_id, None)
                emit("history", history=app.ghost.dialogue_history())

    app.bus.observer = observe
    for module in app.modules:
        original = module.handle_batch

        def handle(batch: list[Envelope], module: BusModule = module,
                   original: Callable[[list[Envelope]], None] = original) -> None:
            if not batch and module.module_id == "scheduler":
                return original(batch)
            started = time.perf_counter()
            context.correlation = batch[0].correlation_id if batch else ""
            emit("module.start", node=module.module_id, inputs=[item.id for item in batch],
                 correlation_id=context.correlation)
            try:
                return original(batch)
            finally:
                emit("module.end", node=module.module_id, seconds=time.perf_counter() - started,
                     invocation_id=module.invocation_id, correlation_id=context.correlation)

        module.handle_batch = handle
    def monitor() -> None:
        """Publish the first application error once shutdown begins."""
        while not app.shutdown_event.wait(.2):
            pass
        if app.error is not None:
            search_cancel.set()
            emit("application.error", error=str(app.error))


    def snapshot() -> None:
        """Publish the accepted records, revisions, and history for the page."""
        view = app.ghost.snapshot()
        records = view.serialize_knoxels()
        for record in records.values():
            record["data"]["embedding"] = []
        emit("snapshot", records=records, revisions=dict(view.revisions),
             gwt=view.serialize_gwt(), runs=app.supervisor.receipts(),
             history=app.ghost.dialogue_history(), user_name=app.ghost.get_user_name(),
             companion_name=app.ghost.get_companion_name(),
             cycle=app.ghost.states[-1].model_dump(mode="json") if app.ghost.states else None)

    search_active = threading.Event()
    search_thread = None

    def search(command: dict) -> None:
        """Run one memory search in this thread and publish its indexed progress."""
        try:
            from pm.memory.embeddings import EmbeddingService
            result = asyncio.run(app.ghost.search_memory(
                command["query"], EmbeddingService(model_config.embedding, live_test=True),
                prepare=bool(command.get("prepare")), cancel=search_cancel,
                record_type=command.get("record_type", ""), subtype=command.get("subtype", ""),
                progress=lambda done, total: emit("memory.index", indexed=done, total=total)))
            emit("memory.search", result=result)
        except BaseException as exc:
            emit("memory.search", error=str(exc) or type(exc).__name__)
        finally:
            search_active.clear()

    try:
        app.run()
        emit("ready", nodes=[{"id": m.module_id, "topics": sorted(m.DATA_TOPICS),
                              "workflow": str(getattr(m, "workflow", ""))} for m in app.modules])
        threading.Thread(target=monitor, name="console-monitor", daemon=True).start()
        snapshot()
        for line in sys.stdin:
            command = json.loads(line)
            request_id = command["request_id"]
            try:
                action = command["action"]
                if action == "chat":
                    if app.shutdown_event.is_set():
                        raise ValueError(f"The application stopped: {app.error or 'shutdown requested'}. Restart the instance.")
                    with submission_lock:
                        if pending:
                            raise ValueError("Wait for the current reply before changing the user or sending another message.")
                        with app.ghost.lock:
                            app.ghost.ghost_config.user_name = command["user"]
                        correlation = app.user_input.submit(command["text"], session=command["session"])
                        pending[correlation] = command["session"]
                    emit("chat.input", session=command["session"], user=command["user"],
                         text=command["text"], correlation_id=correlation)
                elif action == "snapshot":
                    snapshot()
                elif action == "search":
                    if search_active.is_set():
                        raise ValueError("A memory search is already active.")
                    if not command.get("query", "").strip():
                        raise ValueError("Enter a search query.")
                    search_active.set()
                    search_cancel.clear()
                    search_thread = threading.Thread(target=search, args=(command,), name="memory-search", daemon=False)
                    search_thread.start()
                elif action == "cancel_search":
                    search_cancel.set()
                elif action == "memory":
                    batch = app.ghost.begin_batch("console-" + request_id)
                    record = RecordPayload.model_validate(command["record"]).root
                    if command.get("create"):
                        record.id = -1
                        record.tick_id = -1
                        batch.create(record)
                    else:
                        if batch.view().revisions[record.id] != command["revision"]:
                            raise ValueError("The memory changed. Reload the record before you save it.")
                        batch.replace(record)
                    receipt = app.ghost.commit(batch)
                    emit("memory.commit", receipt=receipt.model_dump(mode="json"))
                    snapshot()
                elif action == "run_control":
                    operation = command["operation"]
                    if operation == "cancel":
                        result = app.supervisor.terminate(command["run_id"], request_id=request_id)
                    elif operation == "update":
                        result = app.supervisor.update(command["run_id"], command["text"], request_id=request_id)
                    elif operation == "answer":
                        result = app.supervisor.answer(command["run_id"], command["question_id"], command["text"])
                    elif operation == "resume":
                        result = app.supervisor.resume(command["run_id"], request_id=command.get("request_id", ""))
                    else:
                        raise ValueError("Unknown managed-run operation.")
                    emit("run.control", result=result)
                elif action == "stop":
                    search_cancel.set()
                    emit("ack", request_id=request_id)
                    break
                else:
                    raise ValueError("Unknown command.")
                emit("ack", request_id=request_id)
            except Exception as exc:
                emit("rejected", request_id=request_id, error=str(exc))
    finally:
        search_cancel.set()
        app.shutdown()
        for run_id, receipt in app.supervisor.receipts().items():
            if receipt["state"] in {"accepted", "running", "waiting_for_input"}:
                app.supervisor.terminate(run_id)
        app.join()
        app.supervisor.join_all()
        if search_thread is not None:
            from pm.host.diagnostics import join_thread
            join_thread(search_thread)
        app.raise_if_error()


if __name__ == "__main__":
    try:
        run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")), Path(sys.argv[1]).parent)
    except BaseException:
        import logging
        logging.getLogger(__name__).exception("Console worker failed")
        traceback.print_exc()
        raise
    finally:
        from pm.host.diagnostics import close_logging
        close_logging()
