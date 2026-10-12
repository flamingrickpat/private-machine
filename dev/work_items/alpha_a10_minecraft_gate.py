"""Run the A10 sustained wood leaf and controls against real services."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(args) -> None:
    """Own the real stack, retain failures, and inspect actual actor effects."""
    from main import build_app, _cognition_workflow_options
    from pm.bus.envelopes import Envelope, _new_id
    from pm.records.memory_records import ActionSchemaRecord
    from pm.records.enums import ActionType
    from runtime.body_lease import BODY_LEASES
    from runtime.minecraft_actor import READ_TOOLS
    from runtime.publication import exchange_record
    from pm.runs.workflow_runner import WorkflowRunner, WorkflowRequest
    from pm_workflows.runtime import WorkflowRuntime
    from dev.work_items.alpha_minecraft_gate import RealServerStack
    from tests.live.minecraft_stack import load_env_file
    from pm.host.folder_target import FolderEnvironment

    out = args.out.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Keep live evidence and character files under .runtime.")
    out.mkdir(parents=True, exist_ok=False)
    server = args.server.resolve()
    properties = load_env_file(server / "server.properties")
    values = load_env_file(ROOT.parent / "pm-minecraft/.env_real")
    values.update({"MINECRAFT_PORT": properties["server-port"], "MINECRAFT_RCON_PORT": properties["rcon.port"],
        "MINECRAFT_RCON_PASSWORD": properties["rcon.password"], "MINECRAFT_PLAYER": "tdd_player",
        "MCMCP_TEST_MODE": "true", "MCMCP_AGENT_HOME": str(out / "character"), "MCMCP_INSPECTOR_ENABLED": "false",
        "JAVA_EXE": "C:/Users/rick/AppData/Roaming/PrismLauncher/java/java-runtime-gamma/bin/java.exe",
        "PYTHON_EXE": str(ROOT.parent / "pm-minecraft/.venv/Scripts/python.exe"),
        "TEST_RUN_ROOT": str(out), "SERVER_MEMORY": "2G", "SERVER_START_TIMEOUT_SECONDS": "180"})
    for key in ("MINECRAFT_PORT", "MINECRAFT_RCON_PORT", "MCP_PORT", "MCMCP_BODY_PORT", "MCMCP_VIEWER_PORT"):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", int(values[key])))
    env_path = out / ".env.alpha"
    env_path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
    receipt = {"item": "A10", "started": datetime.now(timezone.utc).isoformat(), "server": str(server),
               "reset_world": False, "seed": properties["level-seed"],
               "source_seed_sha256": hashlib.sha256((ROOT / "test_data/personas/spamuel/canon.yaml").read_bytes()).hexdigest(),
               "status": "running", "checks": {}}
    receipt["source_files_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ("runtime", "workflows/minecraft", "workflows/dialogue_assistant")
        for path in (ROOT / folder).rglob("*") if path.is_file() and path.suffix in {".py", ".md", ".yaml"}}
    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    save()
    stack = RealServerStack(server, out, env_path)
    app = None
    try:
        stack.start()
        os.environ["MINECRAFT_MCP_URL"] = stack.mcp_url
        app = build_app(ROOT / "test_data/personas/spamuel/canon.yaml", out / "app", cognition="real",
                        config_path=str(args.config.resolve()), checkpoint_path=out / "ghost.json", record_lifecycle=False)
        app.supervisor.on_error = lambda error: print(f"Worker error: {type(error).__name__}: {error}", flush=True)
        task = app.supervisor.start(action_id="a10-five-trees", workflow_name="minecraft",
            instructions="Gather wood from five distinct trees and return to your starting position. "
                         "Use ordinary survival tools. Retain collected logs. Observe your position at each worksite. "
                         "Travel at least 16 blocks from the start. Inspect inventory and final progress.",
            arguments={"acceptance": {"kind": "wood_return", "trees": 5, "minimum_travel": 16, "return_radius": 3}})
        run_id = task["run_id"]
        receipt["run_id"] = run_id
        journal = Path(task["artifact_path"]) / "scratch/actor-journal.jsonl"
        deadline = time.monotonic() + args.timeout
        updated = competed = chatted = False
        chat_thread = None
        def inventory_chat():
            """Exercise real chat while the monitor continues applying controls."""
            try:
                prompt = "What is in your Minecraft inventory right now? Use one fresh observation, and keep the wood task running."
                selection = {"companion_name": app.ghost.get_companion_name(), "user_name": app.ghost.get_user_name(),
                    "available_workflows": _cognition_workflow_options(app.supervisor), "capabilities": ["Minecraft numeric observations"],
                    "public_context": {"minecraft": {"status": "connected"}},
                    "stimuli": [{"text": prompt}], "managed_runs": list(app.supervisor.receipts().values())}
                environment = FolderEnvironment(out / "selection", "bash")
                runtime = WorkflowRuntime(environment, threading.Event(), {})
                from pm.llm.stage_model import LazyStageModel
                from pm.llm.activation import ModelActivation
                from pm.llm.config import load_config, resolve_connection
                model = LazyStageModel(resolve_connection(load_config(args.config)), ModelActivation())
                runtime.tools = {"llm_complete": model.complete}
                selected = WorkflowRunner(ROOT / "workflows/main_cognition/stages/action_selection/action_selection.workflow.md",
                    out / "routing-runtime").invoke(WorkflowRequest("a10-inventory-route", selection), runtime)
                receipt["inventory_selection"] = selected.report
                chosen = next(row for row in selected.report["actions"] if row["workflow_name"] == "dialogue_assistant")
                correlation = app.user_input.submit(prompt, session="a10")
                user = exchange_record(app.ghost, correlation)
                batch = app.ghost.begin_batch("a10-inventory-action")
                action = batch.create(ActionSchemaRecord(content=chosen["content"], action_type=ActionType.Reply,
                                                        metadata={"selection_source": "live_action_selection"}))
                app.ghost.commit(batch)
                envelope = Envelope(id=_new_id(), source="a10_gate", kind="data", topic="action.request", correlation_id=correlation,
                    target="dialogue_assistant", payload={"knoxel_id": action.id, "arguments": chosen["arguments"],
                    "stimuli": [{"knoxel_id": user.id, "text": prompt}]})
                reports = []
                publish = app.assistant.publish_output
                def capture(report, batch):
                    reports.append(report)
                    publish(report, batch)
                app.assistant.publish_output = capture
                began = time.monotonic()
                try:
                    app.assistant.handle_batch([envelope])
                finally:
                    app.assistant.publish_output = publish
                receipt["inventory_reply"] = {"reports": reports, "elapsed_seconds": time.monotonic() - began,
                    "exchange": exchange_record(app.ghost, correlation).model_dump(mode="json"),
                    "actor_state_after_reply": app.supervisor.status(run_id)["state"],
                    "managed_run_ids_after_reply": list(app.supervisor.receipts())}
                calls = [call for report in reports for call in report["tool_receipts"] if call["executed"]]
                receipt["checks"]["direct_inventory"] = (len(calls) == 1 and calls[0]["name"] == "minecraft_observe"
                    and calls[0]["result"]["ok"] and len(app.supervisor.receipts()) == 2)
                receipt["checks"]["chat_during_actor"] = bool(reports) and reports[-1]["accepted"] and app.supervisor.status(run_id)["state"] == "running"
            except BaseException as error:
                receipt["inventory_error"] = f"{type(error).__name__}: {error}"
                receipt["checks"]["direct_inventory"] = receipt["checks"]["chat_during_actor"] = False
        while app.supervisor.status(run_id)["state"] in {"accepted", "running"}:
            if time.monotonic() >= deadline:
                raise TimeoutError("The sustained live task exceeded the operator deadline.")
            rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()] if journal.exists() else []
            results = [row for row in rows if row["event"] == "result"]
            if not competed and BODY_LEASES.owner(stack.mcp_url):
                second = app.supervisor.start(action_id="a10-competing", workflow_name="minecraft", instructions="Gather another tree.")
                app.supervisor.join(second["run_id"], 15)
                other = app.supervisor.status(second["run_id"])
                receipt["competing_actor"] = other
                receipt["checks"]["second_actor_rejected"] = other["state"] == "failed" and "owned by" in other["latest_event"]
                competed = True
                save()
            if competed and receipt["competing_actor"]["state"] in {"accepted", "running"}:
                other = app.supervisor.status(receipt["competing_actor"]["run_id"])
                if other["state"] not in {"accepted", "running"}:
                    receipt["competing_actor"] = other
                    receipt["checks"]["second_actor_rejected"] = other["state"] == "failed" and "owned by" in other["latest_event"]
                    save()
            if not updated and any(row["tool"] == "minecraft_harvest_tree" and row["ok"] and row["result"].get("logs_collected", 0) > 0 for row in results):
                receipt["update"] = app.supervisor.update(run_id,
                    "Continue gathering wood from five distinct trees in total, then return to the original starting position. "
                    "Keep the logs in inventory. Do not convert them to planks or sticks. Observe your position at worksites. "
                    "Travel at least 16 blocks from the original start. Save the final inventory and return evidence.",
                    request_id="a10-update-1")
                updated = True
                save()
            if not chatted and len(results) >= 2:
                chat_thread = threading.Thread(target=inventory_chat, name="a10-inventory-chat")
                chat_thread.start()
                chatted = True
                save()
            time.sleep(0.5)
        app.supervisor.join(run_id, 30)
        while chat_thread is not None and chat_thread.is_alive():
            if time.monotonic() >= deadline:
                raise TimeoutError("The live chat exceeded the operator deadline.")
            chat_thread.join(0.5)
        final = app.supervisor.status(run_id)
        if competed:
            app.supervisor.join(receipt["competing_actor"]["run_id"], 30)
            other = app.supervisor.status(receipt["competing_actor"]["run_id"])
            receipt["competing_actor"] = other
            receipt["checks"]["second_actor_rejected"] = other["state"] == "failed" and "owned by" in other["latest_event"]
        receipt["actor_receipt"] = final
        save()
        if final["state"] != "completed":
            raise AssertionError("The actor did not complete: " + final["latest_event"])
        report = yaml.safe_load((Path(final["artifact_path"]) / "outbox/report.yaml").read_text(encoding="utf-8"))
        receipt["actor_report"] = report
        receipt["checks"]["wood_return"] = report["accepted"] and report["verification"]["verified"]
        receipt["checks"]["instruction_applied"] = updated and final["applied_revision"] == 1 and report["instruction_revision"] == 1
        receipt["checks"]["sustained_session"] = bool(report["agent_session"]) and report["call_count"] > 5
        cancelled = app.supervisor.start(action_id="a10-cancel", workflow_name="minecraft",
            instructions="Scout for another distant tree. Walk at least 16 blocks from the start and inspect a safe route. "
                         "Continue until cancelled. Do not harvest.")
        cancel_id = cancelled["run_id"]
        cancel_journal = Path(cancelled["artifact_path"]) / "scratch/actor-journal.jsonl"
        cancel_deadline = time.monotonic() + 300
        cancel_rows = []
        while time.monotonic() < cancel_deadline:
            cancel_rows = [json.loads(line) for line in cancel_journal.read_text(encoding="utf-8").splitlines()] if cancel_journal.exists() else []
            if any(row["event"] == "result" and row["tool"] not in READ_TOOLS for row in cancel_rows):
                break
            time.sleep(0.25)
        signal = {}
        def observe_cancel():
            app.supervisor._cancels[cancel_id].wait()
            signal["time"] = time.time()
        signal_thread = threading.Thread(target=observe_cancel, name="a10-cancel-signal", daemon=True)
        signal_thread.start()
        terminated = app.supervisor.terminate(cancel_id)
        signal_thread.join(1)
        cancel_time = signal["time"]
        for _ in range(360):
            app.supervisor.join(cancel_id, 0.5)
            if BODY_LEASES.owner(stack.mcp_url) is None:
                break
        before = cancel_journal.read_bytes() if cancel_journal.exists() else b""
        time.sleep(2)
        after = cancel_journal.read_bytes() if cancel_journal.exists() else b""
        receipt["cancel_receipt"] = terminated
        receipt["cancel_observations"] = {"completed_calls_before_cancel": len([row for row in cancel_rows if row["event"] == "result"]),
            "body_mutation_completed": any(row["event"] == "result" and row["tool"] not in READ_TOOLS for row in cancel_rows),
            "cancel_time": cancel_time, "journal_unchanged_after_cleanup": before == after}
        stopped_rows = [json.loads(line) for line in after.decode("utf-8").splitlines()]
        receipt["checks"]["cancel_stops_future_calls"] = (receipt["cancel_observations"]["completed_calls_before_cancel"] >= 2
            and receipt["cancel_observations"]["body_mutation_completed"]
            and before == after and BODY_LEASES.owner(stack.mcp_url) is None
            and not any(row["event"] == "dispatch" and row["time"] > cancel_time for row in stopped_rows))
        receipt["status"] = "passed" if all(receipt["checks"].values()) and len(receipt["checks"]) == 7 else "failed"
        save()
        if receipt["status"] != "passed":
            raise AssertionError("A10 checks failed. Inspect the original receipts.")
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        if app is not None:
            app.assistant.stop_event.set()
            app.assistant.run_stop.set()
            for row in app.supervisor.receipts().values():
                if row["state"] in {"accepted", "running", "waiting_for_input"}:
                    app.supervisor.terminate(row["run_id"])
            app.supervisor.join_all(30)
        stack.stop()
        receipt["finished"] = datetime.now(timezone.utc).isoformat()
        save()
        print(out / "receipt.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--server", default=Path("C:/source/minecraft-server-real"), type=Path)
    parser.add_argument("--timeout", default=7200, type=float)
    run(parser.parse_args())
