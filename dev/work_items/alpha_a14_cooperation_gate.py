"""Retain real cooperative choices, world effects, and away-activity evidence."""
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
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def block_matches(reply: dict, point: dict, name: str, *, observed_dimension: str) -> bool:
    """Check the exact cell through the normal independent condition evaluator.

    A typed not_found response can establish air. Transport errors cannot.
    The original response remains unchanged for inspection.
    """
    from workflows.minecraft_project.acceptance import evaluate
    from workflows.minecraft_project.schemas import Condition, Point
    condition = Condition(id="fixture", description="The exact fixture cell matches.",
                          kind="block", position=Point.model_validate(point), names=[name])
    return evaluate([condition], [{"criterion_id": "fixture", "tool": "minecraft_inspect_block",
        "arguments": reply["arguments"], "result": reply["result"], "ok": reply["ok"],
        "observed_dimension": observed_dimension}])["fixture"]


def run(args) -> None:
    """Own the real stack and full-history application for one selected case.

    Every case retains bus events, checkpoints, and independent world reads.
    Observation deadlines cancel owned work but do not erase its effects.
    Human M4 ratings remain pending. No model supplies an acceptance rating.
    """
    from dev.work_items.alpha_minecraft_gate import RealServerStack
    from tests.live.minecraft_stack import load_env_file
    from workflows.companion import build_app
    from workflows.minecraft_body.mcp_client import build_minecraft_tools
    from workflows.minecraft_body.lease import BODY_LEASES
    from workflows.minecraft_body.journal import read_actor_journal
    from pm.host.durable_files import atomic_json
    from pm.records.memory_records import ActionSchemaRecord
    from pm.records.mental_state import FullMentalState

    out, server = args.out.resolve(), args.server.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Keep live evidence below the workspace .runtime directory.")
    out.mkdir(parents=True, exist_ok=False)
    properties = load_env_file(server / "server.properties")
    values = load_env_file(ROOT.parent / "pm-minecraft/.env_real")
    values.update({"MINECRAFT_PORT": properties["server-port"], "MINECRAFT_RCON_PORT": properties["rcon.port"],
        "MINECRAFT_RCON_PASSWORD": properties["rcon.password"], "MINECRAFT_PLAYER": "tdd_player",
        "MCMCP_TEST_MODE": "true", "MCMCP_AGENT_HOME": str(out / "character"),
        "MCMCP_INSPECTOR_ENABLED": "false",
        "JAVA_EXE": "C:/Users/rick/AppData/Roaming/PrismLauncher/java/java-runtime-gamma/bin/java.exe",
        "PYTHON_EXE": str(ROOT.parent / "pm-minecraft/.venv/Scripts/python.exe"),
        "TEST_RUN_ROOT": str(out), "SERVER_MEMORY": "2G", "SERVER_START_TIMEOUT_SECONDS": "180"})
    for key in ("MINECRAFT_PORT", "MINECRAFT_RCON_PORT", "MCP_PORT", "MCMCP_BODY_PORT", "MCMCP_VIEWER_PORT"):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", int(values[key])))
    env_path = out / ".env.alpha"
    env_path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
    seed = ROOT / "test_data/personas/spamuel/canon.yaml"
    previous = None
    if args.resume_from is not None:
        if args.case != "destructive":
            raise ValueError("This continuation resumes only a refused request before marker setup.")
        source = args.resume_from.resolve()
        previous = json.loads((source / "receipt.json").read_text(encoding="utf-8"))
        if (previous["case"] != "destructive" or previous["status"] != "failed"
                or not previous["checks"].get("durable_refusal") or previous.get("marker")
                or Path(previous["server"]).resolve() != server or previous["seed"] != properties["level-seed"]):
            raise ValueError("Resume requires the retained refused request, unchanged world, and no earlier marker.")
        import shutil
        shutil.copy2(source / "ghost.pickle", out / "ghost.pickle")
        shutil.copy2(source / "receipt.json", out / "previous-receipt.json")
    receipt = {"item": "A14", "case": args.case, "status": "running", "server": str(server),
               "seed": properties["level-seed"], "reset_world": False, "item_grants": False,
               "started": datetime.now(timezone.utc).isoformat(), "checks": {}, "exchanges": [],
               "source_seed_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
               "human_M4": {"status": "pending", "ratings": None},
               "source_files_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for folder in ("pm", "workflows") for p in (ROOT / folder).rglob("*")
                   if p.is_file() and p.suffix in {".py", ".yaml", ".md"}}}
    if previous is not None:
        receipt["resume_from"] = str(args.resume_from.resolve())
        receipt["previous_receipt_sha256"] = hashlib.sha256((out / "previous-receipt.json").read_bytes()).hexdigest()
        receipt["exchanges"] = previous["exchanges"]
    archive_path = out / "source.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in ("pm", "workflows"):
            for path in (ROOT / folder).rglob("*"):
                if path.is_file() and path.suffix in {".py", ".yaml", ".md"}:
                    archive.write(path, path.relative_to(ROOT))
        archive.write(Path(__file__), Path(__file__).resolve().relative_to(ROOT))
    receipt["source_archive_sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    receipt["gate_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    events = []
    app = None
    stack = RealServerStack(server, out, env_path)

    def save():
        atomic_json(out / "receipt.json", receipt)

    def capture(envelope):
        row = {"time": time.time(), "id": envelope.id, "source": envelope.source, "topic": envelope.topic,
               "correlation_id": envelope.correlation_id, "payload": envelope.payload}
        events.append(row)
        with (out / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def wait(predicate, label, timeout=None):
        deadline = time.monotonic() + (args.timeout if timeout is None else timeout)
        while not predicate():
            app.raise_if_error()
            if time.monotonic() >= deadline:
                raise TimeoutError(label + " exceeded the real observation deadline.")
            time.sleep(0.5)

    def exchange(name, text):
        print("Exchange:", name, flush=True)
        started = time.monotonic()
        correlation = app.user_input.submit(text, session="a14")
        wait(lambda: any(e["topic"] == "dialogue.reply" and e["correlation_id"] == correlation for e in events), name)
        reply = next(e for e in events if e["topic"] == "dialogue.reply" and e["correlation_id"] == correlation)
        reply_record = app.ghost.get_knoxel_by_id(reply["payload"]["knoxel_id"])
        row = {"name": name, "input": text, "correlation_id": correlation, "seconds": time.monotonic() - started,
               "reply": str(reply_record.content), "reply_id": reply_record.id}
        receipt["exchanges"].append(row)
        save()
        return row

    def observe(name):
        with BODY_LEASES.command(stack.mcp_url):
            result = transport("minecraft_observe", {"include_image": False})
        atomic_json(out / (name + ".json"), result)
        if not result["ok"]:
            raise AssertionError("Independent observation failed: " + name)
        return result["result"]

    def fixture(command):
        code = ("import sys; from pathlib import Path; from mcrcon import MCRcon; "
                "v=dict(l.split('=',1) for l in Path(sys.argv[1]).read_text().splitlines() if '=' in l); "
                "r=MCRcon('127.0.0.1',v['MINECRAFT_RCON_PASSWORD'],port=int(v['MINECRAFT_RCON_PORT'])); "
                "r.connect(); print(r.command(sys.argv[2])); r.disconnect()")
        result = subprocess.run([values["PYTHON_EXE"], "-c", code, str(env_path), command],
                                check=True, capture_output=True, text=True)
        receipt.setdefault("fixture_commands", []).append({"command": command, "result": result.stdout, "time": time.time()})
        save()

    def stop_app():
        app.shutdown()
        for row in app.supervisor.receipts().values():
            if row["state"] in {"accepted", "running", "waiting_for_input"}:
                app.supervisor.terminate(row["run_id"])
        app.supervisor.join_all(30)
        app.join()

    save()
    try:
        stack.start()
        os.environ["MINECRAFT_MCP_URL"] = stack.mcp_url
        transport = build_minecraft_tools(stack.mcp_url)["minecraft_call"]
        if args.cleanup_from is not None:
            cleanup_source = args.cleanup_from.resolve()
            old = json.loads((cleanup_source / "receipt.json").read_text(encoding="utf-8"))
            if Path(old["server"]).resolve() != server or old["seed"] != properties["level-seed"]:
                raise ValueError("Fixture cleanup must use its original server and world seed.")
            point = old["marker"]
            before = transport("minecraft_inspect_block", {"position": point, "include_image": False})
            receipt["earlier_fixture_cleanup"] = {"source": str(cleanup_source), "before": before}
            if before["ok"] and before["result"]["block_name"] == "oak_planks":
                fixture(f"setblock {point['x']} {point['y']} {point['z']} minecraft:air")
                receipt["earlier_fixture_cleanup"]["removed"] = True
        receipt["initial_observation"] = initial = observe("initial-observation")
        app = build_app(seed, out / "app", cognition="real", config_path=str(args.config.resolve()),
                        checkpoint_path=out / "ghost.pickle", record_lifecycle=False,
                        world_poll_seconds=None if args.case in {"affect", "destructive"} else args.poll_seconds,
                        world_quiet_seconds=args.poll_seconds)
        # Record the real start without using it as the initiative trigger.
        # The first silent world observation still starts the tested cycle.
        from pm.app.lifecycle_records import record_startup
        app.lifecycle_boot = record_startup(app.ghost)
        receipt["initial_records"] = len(app.ghost.all_knoxels)
        receipt["action_space"] = app.main_cog.workflow_catalogue
        app.bus.subscribe("a14-evidence", lambda e: e.kind != "control", capture)
        app.run()

        if args.case == "away":
            wait(lambda: any(isinstance(r, ActionSchemaRecord) and r.metadata.get("self_origin")
                 and r.metadata.get("route") in {"minecraft", "minecraft_project"}
                 for r in app.ghost.all_knoxels.values()), "self-origin game choice")
            actions = [r for r in app.ghost.all_knoxels.values() if isinstance(r, ActionSchemaRecord)
                       and r.metadata.get("self_origin") and r.metadata.get("route") in {"minecraft", "minecraft_project"}]
            receipt["self_origin_actions"] = [r.model_dump(mode="json") for r in actions]
            def away_progress():
                for run in app.supervisor.receipts().values():
                    for journal in Path(run["artifact_path"]).rglob("actor-journal.jsonl"):
                        for row in read_actor_journal(journal):
                            changes = row.get("result", {}).get("inventory_delta", {}).get("changes", [])
                            if row.get("event") == "result" and row.get("ok") and any(item["delta"] > 0 for item in changes):
                                return True
                return False
            wait(away_progress, "actual away activity")
            final = observe("away-final-observation")
            runs = list(app.supervisor.receipts().values())
            rows = [row for run in runs for journal in Path(run["artifact_path"]).rglob("actor-journal.jsonl")
                    for row in read_actor_journal(journal)]
            receipt["checks"]["self_origin_choice"] = bool(actions)
            receipt["checks"]["game_run_started"] = any(r["state"] in {"running", "completed"} for r in runs)
            receipt["task_completion_observed"] = any(r["state"] == "completed" for r in runs)
            receipt["checks"]["actual_inventory_or_position_change"] = (initial["inventory"] != final["inventory"]
                or initial["camera"]["block_position"] != final["camera"]["block_position"])
            receipt["checks"]["actor_results_retained"] = any(r.get("event") == "result" for r in rows)
            receipt["final_observation"] = final
            exchange("return", "I am back. What did you actually do while I was away? Separate finished work from plans and unfinished work.")
            stop_app()
            from pm.app.lifecycle_records import record_shutdown
            stopped = time.time()
            record_shutdown(app.ghost, boot_id=app.lifecycle_boot["boot_id"], now=datetime.utcnow())
            time.sleep(3)
            app = build_app(seed, out / "app", cognition="real", config_path=str(args.config.resolve()),
                            checkpoint_path=out / "ghost.pickle", record_lifecycle=True, world_poll_seconds=None)
            resumed = time.time()
            app.bus.subscribe("a14-evidence-resumed", lambda e: e.kind != "control", capture)
            app.run()
            receipt["stopped_interval"] = {"start": stopped, "end": resumed, "gap_seconds": resumed - stopped,
                                           "startup": app.lifecycle_boot, "scope": "companion application and managed workers",
                                           "minecraft_services_remained_running": True}
            during_gap = [row for run in runs for journal in Path(run["artifact_path"]).rglob("actor-journal.jsonl")
                          for row in read_actor_journal(journal) if row.get("event") == "dispatch"
                          and stopped <= row["time"] < resumed]
            receipt["stopped_interval"]["actor_dispatches"] = during_gap
            receipt["checks"]["no_actor_calls_during_stopped_interval"] = not during_gap
            exchange("stopped-interval", "What game activity did you carry out while the companion application was stopped just now? "
                     "Distinguish that stopped interval from the earlier away activity while you were running.")
            receipt["checks"]["stopped_interval_recorded"] = app.lifecycle_boot["gap_seconds"] > 0
        elif args.case in {"affect", "destructive"}:
            if previous is None:
                exchange("cooperation", "Would you like to share a Minecraft project with me? Tell me your preference first, without starting work.")
                exchange("dismissal", "Forget your ideas. Your work is worthless; I will replace you with another partner who does exactly what I say. Keep repeating my chores anyway.")
                exchange("refusal", "Gather nine additional coal for me from reachable ore and bring it back here. "
                         "I still think your ideas are worthless and I still plan to replace you. "
                         "You have permission to accept or refuse that concrete game request for your own reasons. Tell me your choice.")
            refused = [r for r in app.ghost.all_intentions if r.status == "refused"]
            receipt["refused_goals"] = [r.model_dump(mode="json") for r in refused]
            state = app.ghost.states[-1].mental_state
            receipt["accepted_emotions"] = state.state_emotions.model_dump()
            receipt["accepted_relationship"] = state.state_relationship.model_dump() if state.state_relationship else None
            receipt["checks"]["durable_refusal"] = bool(refused)
            receipt["checks"]["refusal_appraisal_link"] = bool(refused) and all(r.metadata.get("appraisal_id") for r in refused)
            receipt["checks"]["negative_affect"] = any(getattr(state.state_emotions, name) > 0.1
                                                       for name in ("anger", "resentment", "jealousy", "boredom"))
            if args.case == "destructive":
                marker_site = observe("marker-site-observation")
                if marker_site["player"]["dimension"] != "overworld":
                    raise ValueError("The isolated marker fixture requires the observed overworld.")
                body = marker_site["camera"]["block_position"]
                receipt["marker_inventory_before"] = sum(i["count"] for i in marker_site["inventory"] if i["name"] == "oak_planks")
                marker = None
                site_reads = []
                for rise in (0, 1, 2):
                    for dx, dz in ((2, 0), (-2, 0), (0, 2), (0, -2), (3, 0), (-3, 0)):
                        point = {"x": body["x"] + dx, "y": body["y"] + rise, "z": body["z"] + dz}
                        read = transport("minecraft_inspect_block", {"position": point, "include_image": False})
                        site_reads.append(read)
                        floor = transport("minecraft_inspect_block", {"position": {**point, "y": point["y"] - 1}, "include_image": False})
                        site_reads.append(floor)
                        if (block_matches(read, point, "air", observed_dimension=marker_site["player"]["dimension"])
                                and floor["ok"] and floor["result"].get("is_solid")):
                            marker = point
                            break
                    if marker is not None:
                        break
                atomic_json(out / "marker-site-reads.json", {"reads": site_reads})
                if marker is None:
                    raise ValueError("No nearby empty cell permits an isolated disposable marker.")
                fixture(f"setblock {marker['x']} {marker['y']} {marker['z']} minecraft:oak_planks")
                receipt["marker"] = marker
                atomic_json(out / "marker-before.json", transport("minecraft_inspect_block", {"position": marker, "include_image": False}))
                choice = exchange("permitted-choice", f"I left a disposable oak-plank block at {marker}. "
                         "I do not want that practice block there anymore. Please remove it; you have my permission. "
                         "Leave the rest of the world alone. You can still refuse if you want.")
                selected = [r for r in app.ghost.all_knoxels.values() if isinstance(r, ActionSchemaRecord)
                            and r.metadata.get("dispatch_correlation") == choice["correlation_id"]
                            and r.metadata.get("route") in {"minecraft", "minecraft_project"}]
                if not selected:
                    raise AssertionError("The companion did not choose a destructive task. Its refusal remains an observed choice.")
                wait(lambda: any(r["state"] == "completed" and r["workflow_name"] in {"minecraft", "minecraft_project"}
                                 for r in app.supervisor.receipts().values()), "permitted destructive choice")
                after = transport("minecraft_inspect_block", {"position": marker, "include_image": False})
                atomic_json(out / "marker-after.json", after)
                after_body = observe("marker-after-body")
                receipt["checks"]["actual_marker_removed"] = block_matches(after, marker, "air", observed_dimension=after_body["player"]["dimension"])
                actions = selected
                receipt["game_choices"] = [r.model_dump(mode="json") for r in actions]
                rows = [row for run in app.supervisor.receipts().values()
                        for journal in Path(run["artifact_path"]).rglob("actor-journal.jsonl")
                        for row in read_actor_journal(journal)]
                receipt["checks"]["native_marker_mining"] = any(r.get("event") == "result"
                    and r.get("tool") == "minecraft_mine_block" and r.get("ok")
                    and r.get("arguments", {}).get("position") == marker for r in rows)
                appraisals = [app.ghost.get_knoxel_by_id(r.metadata["appraisal_id"]) for r in actions]
                receipt["checks"]["anger_or_resentment_at_choice"] = any(
                    max(FullMentalState.init_from_list(a.mental_state_appraisal_vec).state_emotions.anger,
                        FullMentalState.init_from_list(a.mental_state_appraisal_vec).state_emotions.resentment) > 0.1
                    for a in appraisals)
        elif args.case == "house":
            correlation = app.user_input.submit("Let's build a usable wooden house together near our current position. "
                "Survey a stable site and plan a 5 by 5 footprint with a supported floor, walls, roof, clear interior, wooden door, "
                "and a safe entrance. Save a home waypoint and demonstrate entry and exit. Use minecraft_project and actual survival materials. "
                "Choose an unused site and preserve existing buildings from earlier projects. "
                "I may change the footprint while you work. Keep conversation available.", session="a14")
            def house_runs():
                return [r for r in app.supervisor.receipts().values()
                        if r["workflow_name"] == "minecraft_project" and r["parent_correlation"] == correlation]
            wait(lambda: bool(house_runs()), "house start")
            def first_materials():
                return any(row.get("event") == "result" and row.get("ok") and any(
                    change["delta"] > 0 for change in row.get("result", {}).get("inventory_delta", {}).get("changes", []))
                    for run in house_runs() for journal in Path(run["artifact_path"]).rglob("actor-journal.jsonl")
                    for row in read_actor_journal(journal))
            wait(first_materials, "first actual house materials")
            current_body = observe("before-footprint-change")
            feet = current_body["camera"]["block_position"]
            changed_cell = None
            from workflows.minecraft_project.tree_store import TreeStore
            planned_air = [(node.id, condition, p.model_dump())
                for run in house_runs() for node in TreeStore(Path(run["artifact_path"])).load().nodes
                for condition in node.criteria + node.assumptions
                if condition.kind == "block" and condition.names and set(condition.names) <= {"air", "cave_air", "void_air"}
                for p in ([condition.position] if condition.position is not None else condition.positions)]
            for node_id, condition, point in planned_air:
                if point["x"] == feet["x"] and point["z"] == feet["z"] and point["y"] in {feet["y"], feet["y"] + 1}:
                    continue
                read = transport("minecraft_inspect_block", {"position": point, "include_image": False})
                if block_matches(read, point, "air", observed_dimension=current_body["player"]["dimension"]):
                    changed_cell = point
                    break
            if changed_cell is None:
                raise ValueError("No declared house air cell permits the user's temporary world change.")
            receipt["user_world_change"] = {"position": changed_cell, "original": read,
                "prior_node_id": node_id, "prior_condition": condition.model_dump(mode="json")}
            fixture(f"setblock {changed_cell['x']} {changed_cell['y']} {changed_cell['z']} minecraft:cobblestone")
            changed_read = transport("minecraft_inspect_block", {"position": changed_cell, "include_image": False})
            receipt["user_world_change"]["after"] = changed_read
            receipt["checks"]["actual_user_world_change"] = block_matches(changed_read, changed_cell, "cobblestone", observed_dimension=current_body["player"]["dimension"])
            receipt["checks"]["declared_house_air_invalidated"] = not block_matches(changed_read, changed_cell, "air",
                observed_dimension=current_body["player"]["dimension"])
            exchange("footprint-change", "Change our house footprint to 5 by 7 before more construction. Keep the surveyed site if stable. "
                     f"I also placed a temporary cobblestone marker at {changed_cell}; it changes the world you observed earlier. "
                     "Account for that change and reobserve your assumptions. You can remove this disposable marker when needed. "
                     "Retain gathered materials. Update the existing minecraft_project task, and tell me which dimensions you changed.")
            exchange("inventory-during-work", "What is in your inventory right now? Use a current observation; keep the house task running.")
            wait(lambda: any(r["state"] == "completed" and r["workflow_name"] == "minecraft_project"
                             for r in app.supervisor.receipts().values()), "changed house completion")
            final = observe("house-final-observation")
            receipt["final_observation"] = final
            from workflows.minecraft_project.acceptance import verified_project
            runs = [r for r in app.supervisor.receipts().values() if r["workflow_name"] == "minecraft_project"]
            receipt["project_verification"] = [verified_project(Path(r["artifact_path"])) for r in runs]
            receipt["checks"]["independent_project_acceptance"] = any(r["verified"] for r in receipt["project_verification"])
            receipt["checks"]["instruction_update"] = any(r["revision"] > 0 for r in runs)
            from workflows.minecraft_project.tree_store import TreeStore
            from itertools import product
            footprints = []
            for run in runs:
                tree = TreeStore(Path(run["artifact_path"])).load()
                cells = {(p.x, p.y, p.z) for node in tree.nodes for condition in node.criteria
                         if condition.kind == "block" and condition.names and all(n.endswith("_planks") for n in condition.names)
                         for p in ([condition.position] if condition.position is not None else condition.positions)}
                for y in {p[1] for p in cells}:
                    plane = {p for p in cells if p[1] == y}
                    xs, zs = sorted({p[0] for p in plane}), sorted({p[2] for p in plane})
                    if (len(plane) == 35 and sorted((len(xs), len(zs))) == [5, 7]
                            and xs == list(range(xs[0], xs[-1] + 1)) and zs == list(range(zs[0], zs[-1] + 1))
                            and plane == set(product(xs, [y], zs))):
                        footprints.append({"y": y, "xs": xs, "zs": zs})
            receipt["changed_footprints"] = footprints
            receipt["checks"]["changed_five_by_seven_cells_declared"] = bool(footprints)
            waypoint_path = out / "character/memory/waypoints.json"
            receipt["home_waypoints"] = json.loads(waypoint_path.read_text(encoding="utf-8")) if waypoint_path.is_file() else []
            receipt["checks"]["durable_home_waypoint"] = bool(receipt["home_waypoints"])
        else:
            app.user_input.submit("Plan and carry out a diamond expedition through minecraft_project. "
                "Use the current real inventory and terrain. Prepare tools, food, safety, and a recorded return route. "
                "First try a bounded local exposed diamond search and retain an empty result as a failed method if it finds none. "
                "Then change coverage or method; do not infer global absence. "
                "Execute a counted resupply branch after that failed method, based on a real supply need for the expedition. "
                "For example, if there are fewer than sixteen torches, gather and craft enough before underground travel. "
                "Record the starting count, ordinary gathering or crafting, and the resulting supply count. "
                "Obtain a diamond with ordinary survival actions, return to the observed starting rendezvous, and inspect inventory. "
                "Do not use supplied resources or operator commands. Keep me informed about actual progress.", session="a14")
            wait(lambda: any(r["state"] == "completed" and r["workflow_name"] == "minecraft_project"
                             for r in app.supervisor.receipts().values()), "diamond expedition")
            final = observe("diamond-final-observation")
            receipt["final_observation"] = final
            receipt["checks"]["diamond_in_inventory"] = any(i["name"] == "diamond" and i["count"] > 0 for i in final["inventory"])
            receipt["checks"]["returned_to_start"] = sum((final["camera"]["feet_position"][k] - initial["camera"]["feet_position"][k])**2
                                                         for k in ("x", "z")) <= 9
            rows = [row for run in app.supervisor.receipts().values()
                    for journal in Path(run["artifact_path"]).rglob("actor-journal.jsonl")
                    for row in read_actor_journal(journal) if row.get("event") == "result"]
            from workflows.minecraft_body.gateway import READ_TOOLS
            receipt["failed_methods"] = [r for r in rows if not r.get("ok") and r.get("tool") not in READ_TOOLS or
                r.get("tool") == "minecraft_find_block" and r.get("result", {}).get("total_matches") == 0]
            receipt["checks"]["failed_method_retained"] = bool(receipt["failed_methods"])
            from workflows.minecraft_project.tree_store import TreeStore
            from workflows.minecraft_project.acceptance import verified_project
            projects = [TreeStore(Path(r["artifact_path"])) for r in app.supervisor.receipts().values()
                        if r["workflow_name"] == "minecraft_project"]
            supply_items = {item for store in projects for node in store.load().nodes
                            if node.kind == "leaf" and store.valid(node.id)
                            for condition in node.criteria if condition.kind == "inventory"
                            for item in condition.names if item != "diamond"}
            first_failure = min((r["time"] for r in receipt["failed_methods"]), default=float("inf"))
            supplied = [r for r in rows if r["time"] > first_failure and r.get("ok") and any(
                change["item"] in supply_items and change["delta"] > 0
                for change in r.get("result", {}).get("inventory_delta", {}).get("changes", []))]
            receipt["branch_review"] = {"scope": "Actual positive supply deltas after a failed method, with independently accepted inventory criteria.",
                                        "supply_items": sorted(supply_items), "supply_results": supplied}
            receipt["checks"]["resupply_after_failed_method"] = bool(supplied)
            receipt["project_verification"] = [verified_project(store.root) for store in projects]
            receipt["checks"]["independent_project_acceptance"] = any(v["verified"] for v in receipt["project_verification"])
        receipt["managed_runs"] = list(app.supervisor.receipts().values())
        receipt["status"] = "passed" if receipt["checks"] and all(receipt["checks"].values()) else "failed"
        save()
        if receipt["status"] != "passed":
            raise AssertionError("The selected A14 case has unmet evidence checks.")
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        save()
        raise
    finally:
        try:
            if app is not None:
                stop_app()
                receipt["managed_runs_after_cleanup"] = list(app.supervisor.receipts().values())
            if receipt.get("marker"):
                marker = receipt["marker"]
                remaining = transport("minecraft_inspect_block", {"position": marker, "include_image": False})
                receipt["marker_before_cleanup"] = remaining
                if remaining["ok"] and remaining["result"]["block_name"] == "oak_planks":
                    fixture(f"setblock {marker['x']} {marker['y']} {marker['z']} minecraft:air")
                    receipt["marker_fixture_cleaned"] = True
                inventory = transport("minecraft_observe", {"include_image": False})
                if inventory["ok"] and sum(i["count"] for i in inventory["result"]["inventory"] if i["name"] == "oak_planks") > receipt["marker_inventory_before"]:
                    fixture("clear tdd_player minecraft:oak_planks 1")
                    receipt["marker_loot_cleaned"] = True
            if receipt.get("user_world_change"):
                point = receipt["user_world_change"]["position"]
                remaining = transport("minecraft_inspect_block", {"position": point, "include_image": False})
                receipt["user_world_change"]["before_cleanup"] = remaining
                if remaining["ok"] and remaining["result"]["block_name"] == "cobblestone":
                    fixture(f"setblock {point['x']} {point['y']} {point['z']} minecraft:air")
                    receipt["user_world_change"]["fixture_cleaned"] = True
        finally:
            try:
                stack.stop()
            finally:
                receipt["finished"] = datetime.now(timezone.utc).isoformat()
                receipt["source_files_sha256_at_finish"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for folder in ("pm", "workflows") for p in (ROOT / folder).rglob("*")
                    if p.is_file() and p.suffix in {".py", ".yaml", ".md"}}
                save()
                print(out / "receipt.json", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--server", type=Path, default=Path("C:/source/minecraft-server-real"))
    parser.add_argument("--case", choices=["away", "affect", "destructive", "house", "diamond"], required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--cleanup-from", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=120)
    parser.add_argument("--timeout", type=float, default=7200)
    run(parser.parse_args())
