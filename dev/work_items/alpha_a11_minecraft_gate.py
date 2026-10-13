"""Inspect and execute durable subtasks against the real Minecraft deployment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

import yaml

ROOT = Path(__file__).resolve().parents[2]
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


def project_contract_manifest(source: Path, target: Path, *, plan: bool, completion: bool, evidence: bool = False,
                              generation: bool = False) -> Path:
    """Grant optional scenario checks to their actual trusted Python roles.

    The original manifest stays unchanged. The fresh runtime copy retains all
    existing role grants and records the caller's additional capabilities.
    """
    _, header, body = source.read_text(encoding="utf-8").split("---", 2)
    config = yaml.safe_load(header)
    for enabled, role, tool in ((plan, "planner", "minecraft_plan_constraints"),
                                (plan, "reviewer", "minecraft_plan_constraints"),
                                (completion, "final_verifier", "minecraft_project_check")):
        if enabled:
            config["roles"][role]["tools"].append(tool)
    if evidence:
        for role in config["roles"].values():
            if "minecraft_reason" in role["tools"]:
                role["tools"].append("minecraft_evidence_sources")
    if generation:
        for role in config["roles"].values():
            if "minecraft_reason" in role["tools"]:
                role["tools"].append("minecraft_decision_generation")
    with target.open("x", encoding="utf-8") as file:
        file.write("---\n" + yaml.safe_dump(config, sort_keys=False) + "---" + body)
    return target


def run(args) -> None:
    """Own the real stack and retain a reviewed plan or verified project receipt.

    An optional reset uses the authorized real-server procedure after port and
    removal-target checks. Character evidence always stays below .runtime.
    """
    from workflows.companion import build_app
    from workflows.public_speech.reply_recovery import task_evidence
    from dev.work_items.alpha_minecraft_gate import RealServerStack
    from tests.live.minecraft_stack import load_env_file
    from workflows.minecraft_project.tree_store import TreeStore, acceptance_requirements, validate_tree
    from workflows.minecraft_project.acceptance import verified_project

    out, server = args.out.resolve(), args.server.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Keep evidence below the workspace .runtime directory.")
    previous, resume_root = None, getattr(args, "resume_from", None)
    runtime_root, checkpoint, character = out / "app", out / "ghost.json", out / "character"
    goal = args.goal or DEFAULT_GOAL
    generation_path = getattr(args, "decision_generation", None)
    decision_generation = json.loads(generation_path.read_text(encoding="utf-8")) if generation_path else {}
    if not isinstance(decision_generation, dict) or set(decision_generation) & {"model", "messages", "stream", "response_format"}:
        raise ValueError("Decision generation settings cannot change routing, messages, or the result schema.")
    if resume_root is not None:
        resume_root = resume_root.resolve()
        if not resume_root.is_relative_to(ROOT / ".runtime") or resume_root == out:
            raise ValueError("Resume retained evidence below .runtime into a fresh output directory.")
        if args.reset_world or args.source_tree:
            raise ValueError("An explicit resume preserves the world and original task tree.")
        previous = json.loads((resume_root / "receipt.json").read_text(encoding="utf-8"))
        retained = previous["managed_receipt"]
        if (Path(previous["server"]).resolve() != server or previous["plan_only"] != args.plan_only
                or retained["instructions"] != goal):
            raise ValueError("Resume must retain the server, execution mode, and original instructions.")
        runtime_root = Path(previous.get("runtime_root", resume_root / "app")).resolve()
        checkpoint = Path(previous.get("checkpoint_path", resume_root / "ghost.json")).resolve()
        character = Path(previous.get("character_root", resume_root / "character")).resolve()
        for path in (runtime_root, checkpoint, character):
            if not path.is_relative_to(ROOT / ".runtime") or not path.exists():
                raise ValueError("Retained resume paths must exist below the workspace .runtime directory.")
        if Path(previous["task_root"]).resolve() != runtime_root / "managed_tasks" / previous["run_id"]:
            raise ValueError("The original task must belong to the retained runtime.")
        # Cleanup can cancel a waiting task after its failed receipt was saved.
        # Preserve that snapshot and inspect the canonical control boundary.
        control = json.loads((Path(previous["task_root"]) / "control.json").read_text(encoding="utf-8"))
        if control["state"] not in {"cancelled", "failed", "interrupted"}:
            raise ValueError("Stop the original worker before an explicit resume.")
    out.mkdir(parents=True, exist_ok=False)
    if previous is not None:
        (out / "previous-receipt.json").write_bytes((resume_root / "receipt.json").read_bytes())
    properties = load_env_file(server / "server.properties")
    values = load_env_file(ROOT.parent / "pm-minecraft/.env_real")
    values.update({"MINECRAFT_PORT": properties["server-port"], "MINECRAFT_RCON_PORT": properties["rcon.port"],
        "MINECRAFT_RCON_PASSWORD": properties["rcon.password"], "MINECRAFT_PLAYER": "tdd_player",
        "MCMCP_TEST_MODE": "true", "MCMCP_AGENT_HOME": str(character), "MCMCP_INSPECTOR_ENABLED": "false",
        "JAVA_EXE": "C:/Users/rick/AppData/Roaming/PrismLauncher/java/java-runtime-gamma/bin/java.exe",
        "PYTHON_EXE": str(ROOT.parent / "pm-minecraft/.venv/Scripts/python.exe"),
        "TEST_RUN_ROOT": str(out), "SERVER_MEMORY": "1G", "SERVER_START_TIMEOUT_SECONDS": "180"})
    for key in ("MINECRAFT_PORT", "MINECRAFT_RCON_PORT", "MCP_PORT", "MCMCP_BODY_PORT", "MCMCP_VIEWER_PORT"):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", int(values[key])))
    env_path = out / ".env.alpha"
    env_path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
    item = getattr(args, "item", "A11")
    receipt = {"item": item, "status": "running", "started": datetime.now(timezone.utc).isoformat(),
        "server": str(server), "seed": properties["level-seed"], "reset_world": args.reset_world,
        "plan_only": args.plan_only, "item_grants": False, "checks": {},
        "decision_generation": decision_generation,
        "runtime_root": str(runtime_root), "checkpoint_path": str(checkpoint), "character_root": str(character),
        "source_seed_sha256": hashlib.sha256((ROOT / "test_data/personas/spamuel/canon.yaml").read_bytes()).hexdigest(),
        "source_files_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in ("pm", "workflows/minecraft", "workflows/minecraft_body", "workflows/minecraft_project")
            for path in (ROOT / folder).rglob("*") if path.is_file() and path.suffix in {".py", ".md"}}}
    receipt["source_files_sha256"].update({name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("main.py", "dev/work_items/alpha_a11_minecraft_gate.py", "workflows/main_cognition/stages/action_selection/prompts.yaml")})
    import inspect
    from importlib.metadata import distribution
    import pm_coder
    installed_agent = Path(inspect.getfile(pm_coder))
    receipt["agent_component"] = {"installed_source": str(installed_agent),
        "source_sha256": hashlib.sha256(installed_agent.read_bytes()).hexdigest(),
        "distribution_origin": json.loads(distribution("pm-coder").read_text("direct_url.json"))}
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
        if previous is not None and (previous["seed"] != receipt["seed"]
                or previous["source_seed_sha256"] != receipt["source_seed_sha256"]):
            raise ValueError("An explicit resume must retain the server and character seeds.")
        configuration = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        chat = configuration["chat"]["Default"]
        with urlopen(chat["base_url"].rstrip("/") + "/models", timeout=30) as response:
            catalog = json.load(response)
        if chat["model"] not in {row["id"] for row in catalog["data"]}:
            raise ValueError("The selected model is absent from the live chat catalog.")
        receipt["model_identity"] = {"base_url": chat["base_url"], "model": chat["model"], "catalog": catalog,
            "embedding": {key: configuration["embedding"][key] for key in ("base_url", "model", "dimensions", "api_key_env")}}
        if previous is not None:
            for key in ("embedding",):
                if previous["model_identity"][key] != receipt["model_identity"][key]:
                    raise ValueError("An explicit resume must retain its model connections.")
            changed_model = any(previous["model_identity"][key] != receipt["model_identity"][key]
                                for key in ("base_url", "model"))
            if changed_model and not getattr(args, "allow_model_change", False):
                raise ValueError("A different continuation model requires explicit --allow-model-change.")
            if changed_model:
                receipt["model_transition"] = {"explicit": True,
                    "previous": {key: previous["model_identity"][key] for key in ("base_url", "model")},
                    "current": {key: receipt["model_identity"][key] for key in ("base_url", "model")}}
            receipt["resume_from"] = str(resume_root)
            receipt["previous_receipt_sha256"] = hashlib.sha256((out / "previous-receipt.json").read_bytes()).hexdigest()
        save()
        if args.reset_world:
            if properties["level-name"] != "world":
                raise ValueError("The real wipe procedure only supports level-name=world.")
            for name in ("world", "logs", ".fabric"):
                target = (server / name).resolve()
                if not target.is_relative_to(server) or target == server:
                    raise ValueError("A world reset target escaped the named server directory.")
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(server / "wipe-world.ps1")], check=True)
        stack.start(reset_character=previous is None)
        os.environ["MINECRAFT_MCP_URL"] = stack.mcp_url
        app = build_app(ROOT / "test_data/personas/spamuel/canon.yaml", runtime_root, cognition="real",
            config_path=str(args.config.resolve()), checkpoint_path=checkpoint, record_lifecycle=False)
        plan_constraints = getattr(args, "plan_constraints", None)
        project_check = getattr(args, "project_check", None)
        if plan_constraints is not None or project_check is not None or decision_generation:
            from dataclasses import replace
            import inspect
            spec = app.supervisor.workflows["minecraft_project"]
            factory = spec.tools_factory
            def constrained_tools(run_id, environment):
                tools = dict(factory(run_id, environment))
                tools["minecraft_evidence_sources"] = lambda: {
                    "companion_seed": str(ROOT / "test_data/personas/spamuel/canon.yaml"),
                    "companion_checkpoint": str(checkpoint)}
                if decision_generation:
                    tools["minecraft_decision_generation"] = lambda: dict(decision_generation)
                if plan_constraints is not None:
                    tools["minecraft_plan_constraints"] = plan_constraints
                if project_check is not None:
                    tools["minecraft_project_check"] = lambda root, tree: project_check(root, tree, receipt["initial_observation"], character)
                return tools
            manifest = project_contract_manifest(spec.manifest, out / "minecraft_tasks.workflow.md",
                                                 plan=plan_constraints is not None, completion=project_check is not None,
                                                 evidence=True, generation=bool(decision_generation))
            app.supervisor.workflows["minecraft_project"] = replace(spec, manifest=manifest, tools_factory=constrained_tools)
            receipt["project_contracts"] = {"manifest": {
                "source": str(spec.manifest), "source_sha256": hashlib.sha256(spec.manifest.read_bytes()).hexdigest(),
                "runtime": str(manifest), "runtime_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}}
            for name, callback in (("plan", plan_constraints), ("completion", project_check)):
                if callback is not None:
                    source = Path(inspect.getsourcefile(callback))
                    receipt["project_contracts"][name] = {"module": callback.__module__, "function": callback.__name__,
                        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        app.supervisor.on_error = lambda error: print(f"Worker error: {type(error).__name__}: {error}", flush=True)
        from workflows.minecraft_body.mcp_client import build_minecraft_tools
        initial = build_minecraft_tools(stack.mcp_url)["minecraft_call"]("minecraft_observe", {"include_image": False})
        receipt["initial_observation"] = previous["initial_observation"] if previous else initial["result"]
        if previous is not None:
            receipt["resume_observation"] = initial["result"]
        save()
        arguments = {"plan_only": args.plan_only}
        if args.source_tree:
            arguments["proposed_tree"] = json.loads(args.source_tree.read_text(encoding="utf-8"))
            receipt["source_tree_sha256"] = hashlib.sha256(args.source_tree.read_bytes()).hexdigest()
        if previous is not None:
            retained = app.supervisor.status(previous["run_id"])
            if retained["instructions"] != goal or retained["arguments"] != previous["managed_receipt"]["arguments"]:
                raise ValueError("The checkpoint task differs from the saved original intent.")
            # Import retained proof before the worker starts. The native planner
            # must restore exact specifications before these records can count.
            archive = TreeStore(Path(previous["task_root"]))
            archive.archive_acceptances(previous["state"], str(resume_root / "receipt.json"))
            ancestor_path = resume_root / "previous-receipt.json"
            if ancestor_path.is_file() and previous.get("previous_receipt_sha256"):
                if hashlib.sha256(ancestor_path.read_bytes()).hexdigest() != previous["previous_receipt_sha256"]:
                    raise ValueError("The retained predecessor receipt changed.")
                ancestor = json.loads(ancestor_path.read_text(encoding="utf-8"))
                if ancestor["run_id"] != previous["run_id"] or ancestor["task_root"] != previous["task_root"]:
                    raise ValueError("The predecessor certificate snapshot belongs to another task.")
                archive.archive_acceptances(ancestor["state"], str(ancestor_path))
            task = app.supervisor.resume(previous["run_id"], request_id="alpha-resume-" + out.name)
        else:
            task = app.supervisor.start(action_id=item.lower() + "-project", workflow_name="minecraft_project", instructions=goal, arguments=arguments)
        receipt["run_id"] = run_id = task["run_id"]
        receipt["task_root"] = str(task_root := Path(task["artifact_path"]))
        receipt["execution_epoch"] = task["execution_epoch"]
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
                from workflows.minecraft_project.schemas import TaskTree
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
        # Explicit resumes share a character directory. Save its body log
        # before a later body startup replaces that file.
        body_log = character / "body.log"
        if body_log.exists():
            shutil.copy2(body_log, out / "services/body.log")
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
    parser.add_argument("--resume-from", type=Path, help="Resume a stopped project and retain its original evidence and inventory baseline.")
    parser.add_argument("--decision-generation", type=Path, help="JSON generation settings for direct decisions; result schema and routing stay fixed.")
    parser.add_argument("--timeout", type=float, default=10800)
    run(parser.parse_args())
