"""Inspect and execute durable subtasks against the real Minecraft deployment."""
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
import time
from urllib.request import urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DEFAULT_GOAL = (
    "Build a small usable wooden house near the current starting location in ordinary survival. "
    "Use a modest 4 by 4 outer footprint, a solid supported floor, two-block-high walls, a full roof, "
    "a two-cell doorway with a wooden door, and a clear 2 by 2 interior. "
    "Use the current observed coordinates to choose the site; do not invent an existing cleared area. "
    "Gather needed logs, craft counted planks and a door, build the shell, then demonstrate entry and exit "
    "in separate prerequisite-ordered meaningful subtasks. "
    "The gathering leaf must obtain at least twenty additional birch logs beyond the initial inventory. "
    "Use exact cell acceptance for the floor, wall ring, roof, door, and interior clearance. "
    "A separate entry leaf ends inside the house; an exit leaf ends outside by the doorway. "
    "Preserve survival safety and all world effects. Do not grant items or use operator commands."
)


def run(args) -> None:
    """Own the real stack and retain a reviewed plan or verified project receipt.

    An optional reset uses the authorized real-server procedure after port and
    removal-target checks. Character evidence always stays below .runtime.
    """
    from main import build_app
    from runtime.reply_recovery import task_evidence
    from scripts.alpha_minecraft_gate import RealServerStack
    from tests.live.minecraft_stack import load_env_file
    from workflows.minecraft_tasks.tree import TreeStore, acceptance_requirements, validate_tree
    from workflows.minecraft_tasks.verification import verified_project

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
    receipt = {"item": "A11", "status": "running", "started": datetime.now(timezone.utc).isoformat(),
        "server": str(server), "seed": properties["level-seed"], "reset_world": args.reset_world,
        "plan_only": args.plan_only, "item_grants": False, "checks": {},
        "source_seed_sha256": hashlib.sha256((ROOT / "data/spamuel_agentic_canon.yaml").read_bytes()).hexdigest(),
        "source_files_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("runtime", "workflows/minecraft", "workflows/minecraft_tasks")
            for path in (ROOT / folder).rglob("*") if path.is_file() and path.suffix in {".py", ".md"}}}
    receipt["source_files_sha256"].update({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("main.py", "scripts/alpha_a11_minecraft_gate.py", "workflows/main_cognition/stages/action_selection/prompts.yaml")})
    component = ROOT.parent / "pm-minecraft"
    receipt["minecraft_component"] = {
        "repository": str(component),
        "head": subprocess.check_output(["git", "-C", str(component), "rev-parse", "HEAD"], text=True).strip(),
        "working_changes": subprocess.check_output(["git", "-C", str(component), "status", "--short", "--untracked-files=no"], text=True).splitlines(),
        "source_files_sha256": {name: hashlib.sha256((component / name).read_bytes()).hexdigest()
                                 for name in ("body/src/bot.ts", "body/src/server.ts", "body/dist/bot.js", "body/dist/server.js",
                                              "mcmcp/body.py", "mcmcp/body_models.py", "mcmcp/runtime.py", "mcmcp/models.py")}}
    def save():
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    save()
    stack, app = RealServerStack(server, out, env_path), None
    try:
        configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        chat = configuration["chat"]["Default"]
        with urlopen(chat["base_url"].rstrip("/") + "/models", timeout=30) as response:
            catalog = json.load(response)
        if chat["model"] not in {row["id"] for row in catalog["data"]}:
            raise ValueError("The selected model is absent from the live chat catalog.")
        receipt["model_identity"] = {"base_url": chat["base_url"], "model": chat["model"], "catalog": catalog,
            "embedding": {key: configuration["embedding"][key] for key in ("base_url", "model", "dimensions", "api_key_env")}}
        save()
        if args.reset_world:
            if properties["level-name"] != "world":
                raise ValueError("The real wipe procedure only supports level-name=world.")
            for name in ("world", "logs", ".fabric"):
                target = (server / name).resolve()
                if not target.is_relative_to(server) or target == server:
                    raise ValueError("A world reset target escaped the named server directory.")
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(server / "wipe-world.ps1")], check=True)
        stack.start()
        os.environ["MINECRAFT_MCP_URL"] = stack.mcp_url
        app = build_app(ROOT / "data/spamuel_agentic_canon.yaml", out / "app", cognition="real",
            config_path=str(args.config.resolve()), checkpoint_path=out / "ghost.json", record_lifecycle=False)
        app.supervisor.on_error = lambda error: print(f"Worker error: {type(error).__name__}: {error}", flush=True)
        from runtime.minecraft_tools import build_minecraft_tools
        initial = build_minecraft_tools(stack.mcp_url)["minecraft_call"]("minecraft_observe", {"include_image": False})
        receipt["initial_observation"] = initial["result"]
        save()
        arguments = {"plan_only": args.plan_only}
        if args.source_tree:
            arguments["proposed_tree"] = json.loads(args.source_tree.read_text(encoding="utf-8"))
            receipt["source_tree_sha256"] = hashlib.sha256(args.source_tree.read_bytes()).hexdigest()
        goal = args.goal or DEFAULT_GOAL
        task = app.supervisor.start(action_id="a11-project", workflow_name="minecraft_project", instructions=goal, arguments=arguments)
        receipt["run_id"] = run_id = task["run_id"]
        receipt["task_root"] = str(task_root := Path(task["artifact_path"]))
        save()
        deadline = time.monotonic() + args.timeout
        last = None
        while app.supervisor.status(run_id)["state"] in {"accepted", "running"}:
            if time.monotonic() >= deadline:
                raise TimeoutError("The real A11 project exceeded its operator observation deadline.")
            state_path = task_root / "planning/state.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                current = (state["revision"], len(state["attempts"]), tuple(state["acceptances"]))
                if current != last:
                    print("Tree progress:", current, flush=True)
                    last = current
            time.sleep(0.5)
        app.supervisor.join(run_id, 30)
        final = app.supervisor.status(run_id)
        receipt["managed_receipt"] = final
        store = TreeStore(task_root)
        receipt["state"] = store.state
        if store.state["revision"]:
            tree = store.load()
            validate_tree(tree)
            receipt["tree"] = tree.model_dump(mode="json")
            receipt["checks"]["reviewed_plan"] = store.state.get("reviewed_revision") == tree.revision
            name = "reviewed-tree.json" if receipt["checks"]["reviewed_plan"] else "unreviewed-tree.json"
            (out / name).write_text(json.dumps(receipt["tree"], indent=2), encoding="utf-8")
        save()
        if final["state"] != "completed":
            raise AssertionError("The project did not complete: " + final["latest_event"])
        if not args.plan_only:
            verification = verified_project(task_root)
            receipt["verification"] = verification
            receipt["public_evidence"] = task_evidence(final)
            sessions = [attempt.get("session") for attempt in store.state["attempts"] if attempt["state"] == "complete"]
            receipt["checks"].update(independent_world_verification=verification["verified"],
                fresh_specialists=len(sessions) >= 3 and len(set(sessions)) == len(sessions),
                public_evidence_recomputed=receipt["public_evidence"]["task_verified"],
                dependency_order=True)
            for attempt in store.state["attempts"]:
                from workflows.minecraft_tasks.schemas import TaskTree
                attempt_tree = TaskTree.model_validate(json.loads(store.path_for(f"planning/specs/{attempt['tree_revision']:04d}.json").read_text(encoding="utf-8")))
                for dependency in acceptance_requirements(attempt_tree)[attempt["leaf_id"]]:
                    certificate = attempt.get("prerequisites", {}).get(dependency)
                    if certificate is None:
                        receipt["checks"]["dependency_order"] = False
                    else:
                        verified = json.loads((task_root / certificate["verification"]).read_text(encoding="utf-8"))
                        if max(row["time"] for row in verified["reads"]) > attempt["started"]:
                            receipt["checks"]["dependency_order"] = False
        receipt["status"] = "passed" if all(receipt["checks"].values()) else "failed"
        save()
        if receipt["status"] != "passed":
            raise AssertionError("A11 evidence checks failed.")
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        if app is not None:
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--server", type=Path, default=Path("C:/source/minecraft-server-real"))
    parser.add_argument("--source-tree", type=Path)
    parser.add_argument("--goal")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--reset-world", action="store_true")
    parser.add_argument("--timeout", type=float, default=10800)
    run(parser.parse_args())
