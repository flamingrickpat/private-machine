"""Run one instance. Exchange JSON commands and events with the console."""
from __future__ import annotations

import json
import sys
import threading
import time
import traceback
import yaml
from pathlib import Path
from uuid import uuid4

from main import build_app
from pm_core.persist.dataprovider_yaml import _deserialize_knoxel
from runtime.workflow_runner import WorkflowRunner
from runtime.model_activation import ChatClient


def run(settings: dict, root: Path) -> None:
    """Own one runtime, with serialized chat submission and durable edits."""
    output = sys.stdout
    sys.stdout = sys.stderr  # Dependency output must not corrupt the event stream.
    lock = threading.Lock()
    context = threading.local()

    def emit(kind: str, **fields) -> None:
        with lock:
            output.write(json.dumps({"kind": kind, "time": time.time(), **fields}, default=str) + "\n")
            output.flush()

    original_invoke = WorkflowRunner.invoke

    def invoke(runner, request, runtime, **kwargs):
        started = time.perf_counter()
        previous = getattr(context, "workflow", None)
        fields = {"node": runner.namespace, "run_id": request.run_id,
                  "workflow": str(runner.workflow), "input": request.payload,
                  "owner_module": threading.current_thread().name,
                  "parent_run": previous["run_id"] if previous else None,
                  "parent_node": previous["node"] if previous else None,
                  "correlation_id": request.payload.get("correlation_id") or getattr(context, "correlation", "")}
        manifest = runner.workflow.read_text(encoding="utf-8").split("---", 2)
        definition = yaml.safe_load(manifest[1]) if len(manifest) == 3 else {}
        context.workflow = fields
        emit("workflow.start", **fields, definition=definition)
        try:
            result = original_invoke(runner, request, runtime, **kwargs)
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

    async def complete(client, messages, **generation):
        started = time.perf_counter()
        call_id = uuid4().hex
        workflow = getattr(context, "workflow", None)
        fields = {"call_id": call_id, "node": workflow["node"] if workflow else threading.current_thread().name,
                  "owner_module": threading.current_thread().name,
                  "run_id": workflow["run_id"] if workflow else None,
                  "correlation_id": workflow["correlation_id"] if workflow else getattr(context, "correlation", ""),
                  "messages": messages, "generation": generation}
        emit("model.start", **fields)
        try:
            result = await original_complete(client, messages, **generation)
            emit("model.end", **{k: v for k, v in fields.items() if k not in {"messages", "generation"}},
                 seconds=time.perf_counter() - started, result=result.model_dump(mode="json"))
            return result
        except BaseException as exc:
            emit("model.error", **{k: v for k, v in fields.items() if k not in {"messages", "generation"}},
                 seconds=time.perf_counter() - started, error=str(exc))
            raise

    ChatClient.complete = complete
    app = build_app(settings["seed"], root, checkpoint_path=root / "ghost.json",
                    cognition=settings["cognition"], config_path=settings.get("config") or None,
                    emit_reply=lambda text: None)
    pending: dict[str, str] = {}
    submission_lock = threading.Lock()

    def observe(envelope, recipient):
        raw = json.loads(envelope.to_json())
        raw["envelope_kind"] = raw.pop("kind")
        emit("bus", recipient=recipient, **raw)
        if recipient == "user_input" and envelope.topic == "dialogue.reply":
            feature = app.ghost.get_knoxel_by_id(envelope.payload["knoxel_id"])
            if feature is not None:
                with submission_lock:
                    emit("chat.reply", session=pending.get(envelope.correlation_id, "unsolicited"),
                         correlation_id=envelope.correlation_id, text=feature.content)
                    pending.pop(envelope.correlation_id, None)

    app.bus.observer = observe
    for module in app.modules:
        original = module.handle_batch

        def handle(batch, module=module, original=original):
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
    app.run()
    emit("ready", nodes=[{"id": m.module_id, "topics": sorted(m.DATA_TOPICS),
                          "workflow": str(getattr(m, "workflow", ""))} for m in app.modules])

    def monitor():
        while not app.shutdown_event.wait(.2):
            pass
        if app.error is not None:
            emit("application.error", error=str(app.error))

    threading.Thread(target=monitor, daemon=True).start()

    def snapshot():
        view = app.ghost.snapshot()
        records = view.serialize_knoxels()
        for record in records.values():
            record["data"]["embedding"] = []
        emit("snapshot", records=records, revisions=dict(view.revisions),
             gwt=view.serialize_gwt(), runs=app.supervisor.receipts())

    try:
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
                        correlation = app.user_input.submit(command["text"])
                        pending[correlation] = command["session"]
                    emit("chat.input", session=command["session"], user=command["user"],
                         text=command["text"], correlation_id=correlation)
                elif action == "snapshot":
                    snapshot()
                elif action == "memory":
                    batch = app.ghost.begin_batch("console-" + request_id)
                    record = _deserialize_knoxel(command["record"])
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
                        result = app.supervisor.terminate(command["run_id"])
                    elif operation == "update":
                        result = app.supervisor.update(command["run_id"], command["text"])
                    elif operation == "answer":
                        result = app.supervisor.answer(command["run_id"], command["question_id"], command["text"])
                    else:
                        raise ValueError("Unknown managed-run operation.")
                    emit("run.control", result=result)
                elif action == "stop":
                    emit("ack", request_id=request_id)
                    break
                else:
                    raise ValueError("Unknown command.")
                emit("ack", request_id=request_id)
            except Exception as exc:
                emit("rejected", request_id=request_id, error=str(exc))
    finally:
        app.shutdown()
        for run_id, receipt in app.supervisor.receipts().items():
            if receipt["state"] in {"accepted", "running", "waiting_for_input"}:
                app.supervisor.terminate(run_id)
        app.join()
        app.supervisor.join_all()
        app.raise_if_error()


if __name__ == "__main__":
    try:
        run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")), Path(sys.argv[1]).parent)
    except BaseException:
        traceback.print_exc()
        raise
