"""Exercise process interruption and explicit recovery on the real deployment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def wait_for(predicate, timeout=120):
    """Wait for real process evidence. A deadline raises without inventing a pass."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise TimeoutError("The real recovery boundary did not arrive.")


def read(path):
    """Read an atomically published receipt, or return no receipt yet."""
    from pm.host.durable_files import read_json
    try:
        return read_json(path)
    except FileNotFoundError:
        return None


def spawn(arguments, out):
    """Start a hidden owned worker and retain its complete output."""
    log = (out / "worker.log").open("a", encoding="utf-8")
    log.write(f"\nWorker launch {datetime.now(timezone.utc).isoformat()} {arguments}\n")
    log.flush()
    process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), *arguments], cwd=ROOT,
        stdout=log, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    log.close()
    return process


def kill_worker(process):
    """Stop the owned process tree, including the Windows interpreter launcher."""
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        else:
            process.kill()
        process.wait(15)


def protocol_worker(args):
    """Pause at a real call boundary until the parent kills this process."""
    from pm_workflows.runtime import WorkflowRuntime
    from workflows.minecraft_body.lease import BODY_LEASES
    from pm.host.durable_files import atomic_json
    from workflows.minecraft_body.gateway import ActorGateway
    from workflows.minecraft_body.mcp_client import build_minecraft_tools
    from pm.host.folder_target import FolderEnvironment
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    environment = FolderEnvironment(out, "bash")
    tools = build_minecraft_tools()
    tools["managed_instructions"] = lambda **kwargs: {"instruction": "Observe and turn once.", "revision": 0}
    runtime = WorkflowRuntime(environment, threading.Event(), tools)
    gateway = ActorGateway("http://127.0.0.1:6767/mcp", "a12-protocol", environment, runtime, tools)
    original = tools["minecraft_fenced_call"]
    def transport(name, arguments, token):
        result = original(name, arguments, token)
        if name not in {"minecraft_observe", "minecraft_inspect_block"} and args.worker == "after_effect":
            atomic_json(out / "boundary.json", {"phase": "after_effect", "result": result, "token": token})
            threading.Event().wait()
        return result
    gateway.raw_tools["minecraft_fenced_call"] = transport
    with BODY_LEASES.acquire(gateway.endpoint, gateway.owner) as token:
        initial = gateway.call("minecraft_observe", {"include_image": False})
        atomic_json(out / "token.json", token)
        atomic_json(out / "initial.json", initial)
        if args.worker == "before_call":
            atomic_json(out / "boundary.json", {"phase": "before_call", "token": token})
            threading.Event().wait()
        # A sneak pulse keeps the player on its current cell. Its body result
        # includes real asynchronous camera capture while the call is active.
        if args.worker == "in_flight":
            result = gateway.call("minecraft_fine_control", {"sneak": True})
        else:
            result = gateway.call("minecraft_rotate", {"yaw_degrees": 15, "pitch_degrees": 0})
        atomic_json(out / "boundary.json", {"phase": "result_retained", "result": result, "token": token})
        threading.Event().wait()


def native_worker(args):
    """Run the production supervisor, seed, model session, and body workflow."""
    from main import build_app
    from pm.host.durable_files import atomic_json
    out = args.out.resolve()
    scenario = read(out / "scenario.json")
    app = build_app(ROOT / "test_data/personas/spamuel/canon.yaml", out / "app", cognition="real",
        config_path=str(args.config.resolve()), checkpoint_path=out / "ghost.json", record_lifecycle=False)
    app.supervisor.on_error = lambda error: print(f"Worker error: {type(error).__name__}: {error}", flush=True)
    previous = read(out / "task.json")
    if previous is None:
        task = app.supervisor.start(action_id="a12-" + scenario["name"], workflow_name=scenario["workflow"],
            instructions=scenario["instruction"], arguments=scenario["arguments"])
    else:
        task = app.supervisor.resume(previous["run_id"], request_id="restart-" + str(previous["execution_epoch"]))
    atomic_json(out / "task.json", task)
    # A command acknowledged before restart belongs to its earlier epoch.
    # Reapplying an old cancellation here would cancel the replacement run.
    command = read(out / "command.json")
    last = command["id"] if command and (out / ("command-" + command["id"] + ".json")).is_file() else None
    try:
        while True:
            command = read(out / "command.json")
            if command and command["id"] != last:
                operation = command["operation"]
                if operation == "cancel":
                    result = app.supervisor.terminate(task["run_id"], request_id=command["id"])
                elif operation == "update":
                    result = app.supervisor.update(task["run_id"], command["instruction"], request_id=command["id"])
                elif operation == "resume":
                    app.supervisor.join(task["run_id"], 30)
                    result = app.supervisor.resume(task["run_id"], request_id=command["id"])
                else:
                    raise ValueError(operation)
                atomic_json(out / ("command-" + command["id"] + ".json"),
                            {**result, "accepted_wall_time": time.time()})
                last = command["id"]
            status = app.supervisor.status(task["run_id"])
            atomic_json(out / "status.json", status)
            if status["state"] not in {"accepted", "running", "waiting_for_input", "cancelled"}:
                app.supervisor.join(task["run_id"], 30)
                return
            time.sleep(0.1)
    finally:
        app.supervisor.terminate(task["run_id"])
        app.supervisor.join(task["run_id"], 30)


def run(args):
    """Own the selected real server without resets, grants, or lost evidence."""
    import yaml
    from urllib.request import urlopen
    from pm_workflows.runtime import WorkflowRuntime
    from workflows.minecraft_body.lease import BODY_LEASES, StaleEpoch
    from pm.host.durable_files import BodyBusy, atomic_json
    from workflows.minecraft_body.gateway import ActorGateway
    from workflows.minecraft_body.journal import read_actor_journal
    from workflows.minecraft_body.mcp_client import build_minecraft_tools
    from dev.work_items.alpha_minecraft_gate import RealServerStack
    from tests.live.minecraft_stack import load_env_file
    from pm.host.folder_target import FolderEnvironment
    out, server = args.out.resolve(), args.server.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Keep evidence below the workspace .runtime directory.")
    out.mkdir(parents=True, exist_ok=False)
    properties = load_env_file(server / "server.properties")
    values = load_env_file(ROOT.parent / "pm-minecraft/.env_real")
    values.update({"MINECRAFT_PORT": properties["server-port"], "MINECRAFT_RCON_PORT": properties["rcon.port"],
        "MINECRAFT_RCON_PASSWORD": properties["rcon.password"], "MINECRAFT_PLAYER": "tdd_player",
        "MCMCP_TEST_MODE": "true", "MCMCP_AGENT_HOME": str(out / "character"), "MCMCP_INSPECTOR_ENABLED": "false",
        "JAVA_EXE": "C:/Users/rick/AppData/Roaming/PrismLauncher/java/java-runtime-gamma/bin/java.exe",
        "PYTHON_EXE": str(ROOT.parent / "pm-minecraft/.venv/Scripts/python.exe"),
        "TEST_RUN_ROOT": str(out), "SERVER_MEMORY": "1G", "SERVER_START_TIMEOUT_SECONDS": "180"})
    for key in ("MINECRAFT_PORT", "MINECRAFT_RCON_PORT", "MCP_PORT", "MCMCP_BODY_PORT", "MCMCP_VIEWER_PORT"):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", int(values[key])))
    env_path = out / ".env.alpha"
    env_path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
    configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    chat = configuration["chat"]["Default"]
    with urlopen(chat["base_url"].rstrip("/") + "/models", timeout=30) as response:
        catalog = json.load(response)
    if chat["model"] not in {row["id"] for row in catalog["data"]}:
        raise ValueError("The configured model is absent from the live endpoint.")
    receipt = {"item": "A12", "status": "running", "started": datetime.now(timezone.utc).isoformat(),
        "server": str(server), "seed": properties["level-seed"], "reset_world": False, "item_grants": False,
        "model": {"url": chat["base_url"], "model": chat["model"], "catalog": catalog},
        "canon_sha256": hashlib.sha256((ROOT / "test_data/personas/spamuel/canon.yaml").read_bytes()).hexdigest(),
        "checks": {}, "scenarios": []}
    def save():
        atomic_json(out / "receipt.json", receipt)
    save()
    stack, workers = RealServerStack(server, out, env_path), []
    try:
        stack.start()
        os.environ["MINECRAFT_MCP_URL"] = stack.mcp_url
        tools = build_minecraft_tools(stack.mcp_url)
        if args.protocol:
            for phase in ("before_call", "in_flight", "after_effect", "before_publication"):
                target = out / phase
                target.mkdir()
                process = spawn(["--worker", phase, "--out", str(target)], target)
                workers.append(process)
                wait_for(lambda: read(target / "token.json"))
                if phase == "in_flight":
                    wait_for(lambda: any(row.get("event") == "dispatch" and row["tool"] == "minecraft_fine_control"
                        for row in read_actor_journal(target / "scratch/actor-journal.jsonl")))
                else:
                    wait_for(lambda: read(target / "boundary.json"))
                token = read(target / "token.json")
                try:
                    with BODY_LEASES.acquire(stack.mcp_url, "contender"):
                        raise AssertionError("Two processes obtained the body.")
                except BodyBusy:
                    pass
                kill_worker(process)
                environment = FolderEnvironment(target, "bash")
                runtime = WorkflowRuntime(environment, threading.Event(),
                    {"managed_instructions": lambda **kwargs: {"instruction": "Reconcile without mutation.", "revision": 0}})
                gateway = ActorGateway(stack.mcp_url, "a12-resume", environment, runtime, tools)
                with BODY_LEASES.acquire(stack.mcp_url, gateway.owner):
                    observation = gateway.call("minecraft_observe", {"include_image": False})
                    if not observation["ok"]:
                        raise AssertionError("Recovery had no fresh world observation.")
                    reconciliation = gateway.reconcile([])
                    try:
                        tools["minecraft_fenced_call"]("minecraft_rotate", {"yaw_degrees": 90}, token)
                    except Exception as error:
                        rejection = f"{type(error).__name__}: {error}"
                        if "Stale Minecraft body epoch" not in rejection:
                            raise
                    else:
                        raise AssertionError("The server accepted a stale epoch.")
                    gateway.wait_idle()
                rows = read_actor_journal(target / "scratch/actor-journal.jsonl")
                old = [row for row in rows if row.get("epoch") == token["epoch"]]
                if phase == "before_call" and any(row.get("event") == "dispatch" and row["tool"] == "minecraft_rotate" for row in old):
                    raise AssertionError("A before-call interruption sent a mutation.")
                receipt["scenarios"].append({"phase": phase, "token": token, "reconciliation": reconciliation,
                    "rejection": rejection, "observation": observation["result"], "exit_code": process.returncode})
                receipt["checks"][phase] = True
                print("Passed protocol interruption:", phase, flush=True)
                save()
            protocol_controls(out, stack, tools, receipt, save)
        if args.native:
            native_scenarios(args, out, stack, tools, workers, receipt, save)
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for process in workers:
            if process.poll() is None:
                kill_worker(process)
        stack.stop()
        receipt["finished"] = datetime.now(timezone.utc).isoformat()
        save()


def protocol_controls(out, stack, tools, receipt, save):
    """Stop a real active navigation call, then fence an actual MCP restart."""
    from pm_workflows.runtime import WorkflowRuntime
    from workflows.minecraft_body.lease import BODY_LEASES
    from pm.host.durable_files import atomic_json
    from workflows.minecraft_body.gateway import ActorGateway
    from workflows.minecraft_body.journal import read_actor_journal
    from tests.live.minecraft_stack import _wait_port
    from pm.host.folder_target import FolderEnvironment
    target = out / "cancel-active"
    target.mkdir()
    stop = threading.Event()
    instructions = lambda **kwargs: {"instruction": "Observe, walk, and accept cancellation.", "revision": 0}
    runtime = WorkflowRuntime(FolderEnvironment(target, "bash"), stop, {"managed_instructions": instructions})
    gateway = ActorGateway(stack.mcp_url, "a12-cancel-active", runtime.environment, runtime, tools)
    with BODY_LEASES.acquire(stack.mcp_url, gateway.owner):
        initial = gateway.call("minecraft_observe", {"include_image": False})
        results, errors = [], []
        def walk():
            try:
                # This real terrain target retained a 600-second navigation
                # timeout in a12-native-01. It supplies a known active command.
                results.append(gateway.call("minecraft_walk_to_exact", {"x": -22, "y": 113, "z": -2}))
            except BaseException as error:
                errors.append(str(error))
        thread = threading.Thread(target=walk)
        thread.start()
        journal = target / "scratch/actor-journal.jsonl"
        wait_for(lambda: any(row.get("event") == "dispatch" and row.get("tool") == "minecraft_walk_to_exact"
                             for row in read_actor_journal(journal)))
        # Give the real body time to enter its movement loop. This is not a
        # fabricated running result. The thread and journal retain its outcome.
        time.sleep(1)
        if not thread.is_alive():
            raise AssertionError("The real navigation ended before cancellation could be tested.")
        started = time.monotonic()
        stop.set()
        thread.join(30)
        rows = read_actor_journal(journal)
        receipt["cancellation"] = {"elapsed_seconds": time.monotonic() - started, "results": results,
                                    "errors": errors, "journal": str(journal)}
        receipt["checks"]["active_call_cancelled"] = not thread.is_alive() and any(
            row.get("event") == "stop_result" and row["result"]["ok"] for row in rows)
        gateway.wait_idle()
    save()
    if not receipt["checks"]["active_call_cancelled"]:
        raise AssertionError("Cancellation did not stop the actual navigation command.")

    target = out / "mcp-restart"
    target.mkdir()
    runtime = WorkflowRuntime(FolderEnvironment(target, "bash"), threading.Event(), {"managed_instructions": instructions})
    gateway = ActorGateway(stack.mcp_url, "a12-before-mcp-restart", runtime.environment, runtime, tools)
    with BODY_LEASES.acquire(stack.mcp_url, gateway.owner) as token:
        before = gateway.call("minecraft_observe", {"include_image": False})
        atomic_json(target / "old-token.json", token)
        kill_worker(stack.mcp)
        # Preserve the first service log before the normal stack startup opens
        # its new log. The character directory and high-water epoch survive.
        (stack.run_dir / "mcp.log").replace(target / "mcp-before-restart.log")
        stack._start_mcp()
        _wait_port(stack.mcp, stack.host, stack.port, 60, "The restarted MCP")
        stack._control("/test/body/start")
        stack._wait_body(120)
        try:
            tools["minecraft_fenced_call"]("minecraft_rotate", {"yaw_degrees": 90}, token)
        except Exception as error:
            rejection = str(error)
            if "Stale Minecraft body epoch" not in rejection:
                raise
        else:
            raise AssertionError("The restarted MCP accepted its previous active token.")
    gateway = ActorGateway(stack.mcp_url, "a12-after-mcp-restart", runtime.environment, runtime, tools)
    with BODY_LEASES.acquire(stack.mcp_url, gateway.owner) as replacement:
        after = gateway.call("minecraft_observe", {"include_image": False})
        reconciliation = gateway.reconcile([])
        gateway.wait_idle()
    receipt["mcp_restart"] = {"old_token": token, "replacement": replacement, "rejection": rejection,
                              "before": before, "after": after, "reconciliation": reconciliation}
    receipt["checks"]["mcp_process_restart"] = after["ok"] and replacement["epoch"] > token["epoch"]
    save()
    print("Passed active-call cancellation and MCP-process restart.", flush=True)


def native_scenarios(args, out, stack, tools, workers, receipt, save):
    """Cancel, reconnect, and resume wood; kill and restart a real house leaf."""
    from pm.host.durable_files import atomic_json
    from workflows.minecraft_body.journal import read_actor_journal
    if args.house_only:
        return house_scenario(args, out, stack, tools, workers, receipt, save)
    target = out / "wood"
    target.mkdir()
    scenario = {"name": "wood", "workflow": "minecraft", "instruction":
        "Gather logs from two distinct trees, retain the logs, and return to your original starting position. "
        "Use ordinary survival tools. Observe your position at each tree. Keep the task active until you return.",
        "arguments": {"include_image": False,
                      "acceptance": {"kind": "wood_return", "trees": 2, "minimum_travel": 0, "return_radius": 3}}}
    atomic_json(target / "scenario.json", scenario)
    process = spawn(["--worker", "native", "--out", str(target), "--config", str(args.config.resolve())], target)
    workers.append(process)
    task = wait_for(lambda: read(target / "task.json"))
    journal = Path(task["artifact_path"]) / "scratch/actor-journal.jsonl"
    wait_for(lambda: any(row.get("event") == "result" and row["tool"] == "minecraft_observe"
                        for row in read_actor_journal(journal)), args.timeout)
    updated_instruction = scenario["instruction"] + " Keep every collected log. Do not convert the logs into planks or sticks."
    atomic_json(target / "command.json", {"id": "wood-update", "operation": "update", "instruction": updated_instruction})
    queued = wait_for(lambda: read(target / "command-wood-update.json"))
    wait_for(lambda: any(row.get("event") == "result" and row["tool"] == "minecraft_harvest_tree"
        and row["ok"] and row["result"].get("logs_collected", 0) > 0 for row in read_actor_journal(journal)), args.timeout)
    atomic_json(target / "command.json", {"id": "wood-cancel", "operation": "cancel"})
    cancelled = wait_for(lambda: read(target / "command-wood-cancel.json"))
    if cancelled["state"] != "cancelled":
        raise AssertionError("The wood task did not accept cancellation.")
    # Kill only after the worker closes the body lease. The cancellation keeps
    # every result from the call that was in flight at the accepted boundary.
    from workflows.minecraft_body.lease import BODY_LEASES
    wait_for(lambda: BODY_LEASES.owner(stack.mcp_url) is None)
    kill_worker(process)
    stack._control("/test/body/stop")
    stack._control("/test/body/start")
    stack._wait_body(120)
    resumed = spawn(["--worker", "native", "--out", str(target), "--config", str(args.config.resolve())], target)
    workers.append(resumed)
    final = wait_for(lambda: (status if (status := read(target / "status.json")) and
        status.get("execution_epoch", 1) > 1 and status["state"] in {"completed", "failed"} else None), args.timeout)
    resumed.wait(30)
    receipt["wood"] = {"update": queued, "cancelled": cancelled, "final": final}
    receipt["checks"]["instruction_revision_retained"] = final["revision"] >= queued["revision"]
    receipt["checks"]["no_superseded_dispatch"] = all(
        row["revision"] >= queued["revision"] for row in read_actor_journal(journal)
        if row.get("event") == "dispatch" and row["time"] > queued["accepted_wall_time"])
    receipt["checks"]["wood_cancel_reconnect_resume"] = final["state"] == "completed"
    save()
    if final["state"] != "completed":
        raise AssertionError("The resumed wood task did not complete: " + final["latest_event"])
    receipt["wood"]["journal"] = str(journal)
    print("Passed native wood cancellation, reconnect, and resume.", flush=True)
    house_scenario(args, out, stack, tools, workers, receipt, save)


def house_scenario(args, out, stack, tools, workers, receipt, save):
    """Retain exact A11 cells across a real inspection-leaf process restart."""
    from pm.host.durable_files import atomic_json
    from workflows.minecraft_body.journal import read_actor_journal
    target = out / "house"
    target.mkdir()
    source = ROOT / ".runtime/alpha/a11h16/reviewed-tree.json"
    original = read(source)
    retained = next(node for node in original["nodes"] if node["id"] == "verify_house")
    physical = [condition for condition in retained["criteria"] if condition["kind"] != "accepted"]
    instruction = "Inspect and preserve the existing birch house at x3..6,z-8..-5. " \
        "Verify every supplied exact floor, wall, roof, doorway, air-clearance, and support cell. " \
        "Retain all correct blocks and stay alive. This maintenance leaf asks for exact structure evidence."
    tree = {"goal_id": "house_recovery", "nodes": [
        {"id": "house_recovery", "kind": "goal", "children": ["house_inspection"], "instruction": instruction,
         "criteria": [{"id": "inspected", "description": "Independent structure verification passed.",
                       "kind": "accepted", "node_id": "house_inspection"}]},
        {"id": "house_inspection", "kind": "leaf", "parent": "house_recovery", "specialist": "builder",
         "instruction": instruction, "criteria": physical}]}
    scenario = {"name": "house", "workflow": "minecraft_project", "instruction": instruction,
                "arguments": {"proposed_tree": tree, "include_image": False}}
    # include_image is not a project request field. Its actors retain text
    # observations through the existing project contract.
    scenario["arguments"].pop("include_image")
    atomic_json(target / "scenario.json", scenario)
    process = spawn(["--worker", "native", "--out", str(target), "--config", str(args.config.resolve())], target)
    workers.append(process)
    task = wait_for(lambda: read(target / "task.json"))
    task_root = Path(task["artifact_path"])
    def active_house():
        state = read(task_root / "planning/state.json")
        if not state:
            return None
        for attempt in state["attempts"]:
            journal = task_root / attempt["path"] / "scratch/actor-journal.jsonl"
            if attempt["state"] == "running" and any(row.get("event") == "result" for row in read_actor_journal(journal)):
                return attempt
        status = read(target / "status.json")
        if status and status["state"] in {"failed", "completed"}:
            raise AssertionError("The house did not reach its restart boundary: " + status["latest_event"])
        return None
    interrupted = wait_for(active_house, args.timeout)
    kill_worker(process)
    resumed = spawn(["--worker", "native", "--out", str(target), "--config", str(args.config.resolve())], target)
    workers.append(resumed)
    final = wait_for(lambda: (status if (status := read(target / "status.json")) and
        status.get("execution_epoch", 1) > 1 and status["state"] in {"completed", "failed"} else None), args.timeout)
    resumed.wait(30)
    from workflows.minecraft_project.acceptance import verified_project
    evidence = verified_project(task_root)
    receipt["house"] = {"source_tree": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "interrupted_attempt": interrupted, "final": final, "verification": evidence,
                        "state": read(task_root / "planning/state.json")}
    receipt["checks"]["house_process_restart"] = final["state"] == "completed" and evidence["verified"]
    save()
    if not receipt["checks"]["house_process_restart"]:
        raise AssertionError("The restarted house leaf did not verify: " + final["latest_event"])
    print("Passed native house-leaf process restart and independent verification.", flush=True)
    if not all(receipt["checks"].values()):
        raise AssertionError("A real recovery contract failed. Inspect the retained checks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.alpha.example.yaml")
    parser.add_argument("--server", type=Path, default=Path("C:/source/minecraft-server-real"))
    parser.add_argument("--protocol", action="store_true")
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--house-only", action="store_true", help="Run only the native house-leaf restart case.")
    parser.add_argument("--worker", choices=["before_call", "in_flight", "after_effect", "before_publication", "native"])
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.worker == "native":
        native_worker(args)
    elif args.worker:
        protocol_worker(args)
    else:
        if not args.protocol and not args.native:
            parser.error("Select --protocol, --native, or both.")
        run(args)


if __name__ == "__main__":
    main()
