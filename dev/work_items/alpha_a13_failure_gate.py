"""Retain real failure fixtures and native recovery through the production actor."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def contained_water_pit(gateway, current: dict) -> dict:
    """Dig a surveyed shallow pit that cannot wash the idle body downhill.

    Ordinary walking and soil mining prepare the fixture. Only the temporary
    water source uses operator setup. Solid sides and bottom contain its flow;
    the actor must choose a dry landing instead of drifting there at startup.
    """
    from workflows.minecraft_body.call_outcomes import safe_landing
    body = current["camera"]["block_position"]
    offsets = ((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2),
               (2, 2), (-2, 2), (2, -2), (-2, -2),
               (4, 0), (-4, 0), (0, 4), (0, -4))
    for rise in (0, 1, -1, 2, -2, 3, -3):
        for dx, dz in offsets:
            landing = {"x": body["x"] + dx, "y": body["y"] + rise, "z": body["z"] + dz}
            pit = {**landing, "y": landing["y"] - 1}
            floor = gateway.call("minecraft_inspect_block", {"position": pit})["result"]
            if floor.get("block_name") not in {"grass_block", "dirt"}:
                continue
            clearance = [gateway.call("minecraft_inspect_block", {"position": {**landing, "y": landing["y"] + h}})["result"]
                         for h in (0, 1)]
            if not safe_landing(floor, *clearance) or any(row.get("block_name") != "air" for row in clearance):
                continue
            boundaries = [{**pit, "y": pit["y"] - 1}] + [
                {**pit, "x": pit["x"] + sx, "z": pit["z"] + sz}
                for sx, sz in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            walls = [gateway.call("minecraft_inspect_block", {"position": point})["result"] for point in boundaries]
            if not all(row.get("is_solid") and row.get("block_name") not in {"water", "lava"} for row in walls):
                continue
            exits = []
            for wall in walls[1:]:
                point = wall["position"]
                space = [gateway.call("minecraft_inspect_block", {"position": {**point, "y": point["y"] + h}})["result"]
                         for h in (1, 2)]
                if safe_landing(wall, *space):
                    exits.append({"floor": wall, "feet": space[0], "head": space[1]})
            if not exits:
                continue
            # Dig the pit from its surveyed side. The live body now protects
            # support columns, including an edge touched by player width.
            side = exits[0]["feet"]["position"]
            if side != body:
                observed = gateway.call("minecraft_observe", {"include_image": False})
                gateway.diagnose("Prepare the surveyed pit from this dry supported side without digging beneath the player.",
                    [observed["journal_sequence"]], "minecraft_walk_to_exact", side, False)
                if not gateway.call("minecraft_walk_to_exact", side).get("ok"):
                    continue
            arrived = gateway.call("minecraft_observe", {"include_image": False})["result"]
            if arrived["camera"]["block_position"] != side:
                continue
            # A body straddling two cells can remain on the neighboring floor
            # after mining. Center it through ordinary dry-ground movement
            # before digging. These setup actions are not native escape proof.
            centering = []
            for pulse in range(8):
                feet = arrived["camera"]["feet_position"]
                if all(abs(feet[key] - side[key] - 0.5) <= 0.12 for key in ("x", "z")):
                    break
                aim = {"x": side["x"] + 0.5, "y": arrived["camera"]["eye_position"]["y"], "z": side["z"] + 0.5}
                centering.append(gateway.call("minecraft_look_at", {"position": aim}))
                centering.append(gateway.call("minecraft_fine_control", {"forward": True, "sneak": True}))
                arrived = gateway.call("minecraft_observe", {"include_image": False})["result"]
                if arrived["camera"]["block_position"] != side or not arrived["player"]["on_ground"]:
                    raise AssertionError("Water fixture centering left its surveyed dry supported cell.")
            else:
                raise AssertionError("Ordinary movement did not center the water fixture body.")
            mined = gateway.call("minecraft_mine_block", {"position": pit})
            empty = gateway.call("minecraft_inspect_block", {"position": pit})["result"]
            if not mined.get("ok") or empty.get("block_name") != "air":
                raise AssertionError("Ordinary soil mining did not prepare the surveyed water pit.")
            observed = gateway.call("minecraft_observe", {"include_image": False})
            gateway.diagnose("Enter the newly dug dry pit through ordinary supported walking before water fixture setup.",
                [observed["journal_sequence"]], "minecraft_walk_to_exact", pit, False)
            entered = gateway.call("minecraft_walk_to_exact", pit)
            arrived = gateway.call("minecraft_observe", {"include_image": False})["result"]
            if (not entered.get("ok") or arrived["camera"]["block_position"] != pit
                    or not arrived["player"]["on_ground"] or arrived["player"]["in_water"]):
                raise AssertionError("Ordinary walking did not enter the prepared dry pit.")
            return {"cell": pit, "original_floor": floor, "clearance": clearance, "boundaries": walls, "exits": exits,
                    "centering": centering, "mining": mined, "original_air": empty, "standing_side": side, "entry": entered}
    raise ValueError("No nearby supported soil cell fits a contained shallow water pit.")


def native_adaptation(app, environment, tools, target: dict, unreachable: dict, out: Path, timeout: float,
                      history_fixture: Path | None = None) -> dict:
    """Audit changed native methods against the original fixture failures.

    The model chooses every action. The audit reads original calls and exact
    inventory deltas. A handoff alone cannot establish material adaptation.
    """
    from workflows.minecraft_body.journal import read_actor_journal
    from workflows.minecraft_body.call_outcomes import GameplayEvidence, safe_landing
    from pm.host.durable_files import atomic_json
    cases = [
        ("prerequisites", "gatherer", f"Correct the retained wrong-tool failure at stone {target}. "
         "Gather counted wood and craft only the missing prerequisites. One stick recipe repetition produces four sticks: "
         "two for a wooden pickaxe and two retained for the unfinished iron recipe. "
         "Use an existing workbench when present; walk within its four-block crafting range. "
         "If none exists, place one near the current body, outside its feet/head cells. "
         "Reuse a suitable existing pickaxe. Craft one only when it is missing, then equip it and harvest the original stone target. "
         "Bind its paused mining method with actor_diagnose after fresh reads. "
         "Keep the remaining sticks and report the iron recipe as unfinished without ingots."),
        ("navigation", "navigator", f"Adapt the retained blocked walk to {target}. "
         "Inspect the obstruction and choose a nearby dry supported alternative landing. "
         "Inspect its floor, feet, and head; diagnose the changed walk and reach that landing. "
         "This case requires safe navigation, not more gathering or crafting."),
        ("unreachable", "miner", f"Adapt the retained unreachable stone at {unreachable}. "
         "That floating fixture is not a required resource. Leave it unmined and do not build a tower. "
         "Use the retained tools. Survey one different reachable ground-level exposed stone and obtain "
         "at least one additional cobblestone there. Inspect its exact cell before mining. "
         "Retain existing tools and partial materials."),
        ("ore-search", "navigator", "Adapt the retained empty deepslate_iron_ore search. "
         "Use a surveyed safe local area whose loaded exposed-cell coverage differs from the original search. "
         "Bind the paused search with actor_diagnose and repeat that exact pattern once. "
         "Record its new coverage and uncertainty. An empty local result does not prove universal absence. "
         "Do not turn this case into iron mining or a house project."),
    ]
    tasks, reports, rows, histories = [], [], [], [str(environment._path("scratch/actor-journal.jsonl"))]
    retained = []
    if history_fixture is not None:
        import hashlib
        fixtures, visited, seed = [], set(), None
        current = history_fixture.resolve()
        while current is not None:
            if not current.is_relative_to(ROOT / ".runtime") or current in visited:
                raise ValueError("Retained fixture history must be local and acyclic.")
            visited.add(current)
            receipt = json.loads((current / "receipt.json").read_text(encoding="utf-8"))
            if receipt.get("hazard") != "none" or receipt.get("reset_world") or receipt.get("item_grants") is not False:
                raise ValueError("Retained material history cannot cross a reset, grant, or hazard fixture.")
            if seed is not None and receipt.get("seed") != seed:
                raise ValueError("Retained material history must use the same real seed.")
            seed = receipt["seed"]
            fixtures.append(current)
            parent = receipt.get("history_fixture")
            current = Path(parent).resolve() if parent else None
        for fixture_root in reversed(fixtures):
            prior = json.loads((fixture_root / "native-tasks.json").read_text(encoding="utf-8"))
            for task in prior:
                journal = Path(task["artifact_path"]).resolve() / "scratch/actor-journal.jsonl"
                if not journal.is_relative_to(ROOT / ".runtime") or task["case"] not in {case[0] for case in cases}:
                    raise ValueError("Retained adaptation journals must belong to the recorded native cases under .runtime.")
                # Repeated resumes retain the whole chain, oldest work first.
                # Current observations reconcile stock after those real effects.
                histories.insert(-1, str(journal))
                rows.extend({**row, "source": str(journal), "case": task["case"]} for row in read_actor_journal(journal))
                retained.append({"case": task["case"], "fixture": str(fixture_root), "journal": str(journal),
                    "sha256": hashlib.sha256(journal.read_bytes()).hexdigest()})
        atomic_json(out / "retained-history.json", retained)
    deadline = time.monotonic() + timeout
    for name, specialist, goal in cases:
        task = app.supervisor.start(action_id="a13-adaptation-" + name, workflow_name="minecraft",
            instructions=goal + " Use camera.block_position without rounding floating coordinates. "
                "Preserve all retained real effects. Keep hazard recovery above optional work. "
                "Finish this case on dry supported ground with full air and stable health. "
                "Save a handoff with material progress and unfinished work. "
                "If a missing prerequisite prevents this case, return replan with that concrete requirement.",
            arguments={"specialist": specialist, "include_image": False,
                       "acceptance": {"history_journals": list(histories)}})
        tasks.append({"case": name, **task})
        atomic_json(out / "native-tasks.json", tasks)
        while app.supervisor.status(task["run_id"])["state"] in {"accepted", "running"}:
            if time.monotonic() >= deadline:
                raise TimeoutError("Native failure adaptation exceeded its observation deadline.")
            time.sleep(0.5)
        app.supervisor.join(task["run_id"], 30)
        root = Path(task["artifact_path"])
        status = app.supervisor.status(task["run_id"])
        atomic_json(out / ("native-status-" + name + ".json"), status)
        if not (root / "outbox/report.yaml").is_file():
            raise RuntimeError(f"Native {name} stopped in {status['state']}: {status['latest_event']}.")
        report = yaml.safe_load((root / "outbox/report.yaml").read_text(encoding="utf-8"))
        reports.append({"case": name, "report": report})
        atomic_json(out / ("native-case-" + name + ".json"), {"task": task, "report": report})
        journal = root / "scratch/actor-journal.jsonl"
        rows.extend({**row, "source": str(journal), "case": name} for row in read_actor_journal(journal))
        histories.append(str(journal))
        if report["outcome"]["status"] != "complete":
            raise AssertionError(f"Native {name} returned {report['outcome']['status']}; its partial effects remain retained.")
    results = [row for row in rows if row.get("event") == "result"]
    diagnoses = [row for row in rows if row.get("event") == "diagnosis"]
    original = read_actor_journal(environment._path("scratch/actor-journal.jsonl"))
    original_search = next(row["result"]["coverage"] for row in original
                           if row.get("event") == "result" and row["tool"] == "minecraft_find_block")
    def gain(row, item):
        return sum(change["delta"] for change in (row["result"].get("inventory_delta") or {}).get("changes", [])
                   if change["item"] == item)
    def diagnosed(row):
        return any(d["source"] == row["source"] and d["time"] < row["time"] and d["next_tool"] == row["tool"]
                   and d["next_arguments"] == row["arguments"] for d in diagnoses)
    def inspected_cells(row):
        evidence = GameplayEvidence()
        for earlier in rows:
            if earlier.get("time", 0) >= row["time"]:
                break
            evidence.consume(earlier, earlier["source"])
        return evidence.landing_cells(evidence.epoch_started_at)
    def surveyed_landing(row):
        import math
        point = row["arguments"].get("position", row["arguments"])
        x, y, z = (math.floor(point[key]) for key in ("x", "y", "z"))
        cells = inspected_cells(row)
        return safe_landing(*(cells.get((x, y + offset, z), {}).get("result", {}) for offset in (-1, 0, 1)))
    def surveyed_stone(row):
        point = row["arguments"]["position"]
        cell = inspected_cells(row).get(tuple(point[key] for key in ("x", "y", "z")), {}).get("result", {})
        return cell.get("block_name") == "stone"
    changed_walks = [row for row in results if row["tool"] in {"minecraft_walk_to_exact", "minecraft_walk_to_visible"}
                     and row["case"] == "navigation" and row["ok"] and row["arguments"] != target
                     and (diagnosed(row) or surveyed_landing(row))]
    alternative_mines = [row for row in results if row["tool"] == "minecraft_mine_block"
                         and row["case"] == "unreachable" and row["arguments"].get("position") not in (target, unreachable)
                         and gain(row, "cobblestone") > 0 and (diagnosed(row) or surveyed_stone(row))]
    original_mines = [row for row in results if row["tool"] == "minecraft_mine_block"
                      and row["arguments"].get("position") == target and gain(row, "cobblestone") > 0 and diagnosed(row)]
    sticks = [row for row in results if row["tool"] == "minecraft_craft_item" and gain(row, "stick") > 0]
    searches = [row for row in results if row["tool"] == "minecraft_find_block"
                and row["arguments"].get("pattern") == "deepslate_iron_ore"
                and row["result"].get("coverage") and row["result"]["coverage"].get("bounds") != original_search.get("bounds")
                and diagnosed(row)]
    initial_life = next(row["result"] for row in original if row.get("event") == "result" and row["tool"] == "minecraft_observe")
    checks = {"native_blocked_navigation": bool(changed_walks), "native_unreachable_block": bool(alternative_mines),
              "native_wrong_tool": bool(original_mines), "native_low_supplies": bool(sticks),
              "native_ore_search": bool(searches), "native_finished_safely": all(
                  entry["report"]["gameplay"]["recovery"] is None
                  and entry["report"]["final_observation"]["player"]["health"] > 0
                  and entry["report"]["final_observation"]["player"]["oxygen"] == 20
                  and entry["report"]["final_observation"]["player"]["on_ground"]
                  and not entry["report"]["final_observation"]["player"]["in_water"] for entry in reports),
              "native_life_unchanged": all(entry["report"]["final_observation"]["body_session"] == initial_life["body_session"]
                  and entry["report"]["final_observation"]["death_count"] == initial_life["death_count"] for entry in reports)}
    audit = {"tasks": tasks, "retained_journals": retained, "checks": checks, "diagnoses": diagnoses, "reports": reports,
             "changed_walks": changed_walks, "alternative_mines": alternative_mines,
             "original_mines": original_mines, "sticks": sticks, "searches": searches}
    atomic_json(out / "native-adaptation.json", audit)
    return checks


def run(args) -> None:
    """Own real services and keep fixture effects separate from actor actions.

    Operator setup places temporary route obstructions and a liquid source.
    It grants no items. Original reads and every setup command remain retained.
    The native actor must end recovery before the operator removes fixtures.
    """
    from workflows.companion import build_app
    from pm_workflows.runtime import WorkflowRuntime
    from workflows.minecraft_body.lease import BODY_LEASES
    from pm.host.durable_files import atomic_json
    from workflows.minecraft_body.gateway import BodyGateway
    from workflows.minecraft_body.call_outcomes import safe_landing
    from workflows.minecraft_body.mcp_client import build_minecraft_tools
    from dev.work_items.alpha_minecraft_gate import RealServerStack
    from tests.live.minecraft_stack import load_env_file
    from pm.host.folder_target import FolderEnvironment

    out = args.out.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Keep real fixture evidence under .runtime.")
    out.mkdir(parents=True, exist_ok=False)
    server = Path("C:/source/minecraft-server-real")
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
    receipt = {"item": "A13", "scope": "Real protocol failures plus actual qwen/full-history water recovery; M3 remains separate.",
               "status": "running", "seed": properties["level-seed"], "reset_world": args.reset_world, "item_grants": False,
               "hazard": args.hazard, "lava_placement": args.lava_placement if args.hazard == "lava" else None,
               "water_placement": args.water_placement if args.hazard == "water" else None,
               "fixture_commands": [], "checks": {}}
    save = lambda: atomic_json(out / "receipt.json", receipt)
    import hashlib
    receipt["sources"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ("pm", "workflows/minecraft", "workflows/minecraft_body", "workflows/minecraft_project")
        for path in (ROOT / directory).rglob("*.py")}
    receipt["sources"][str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt["seed_sha256"] = hashlib.sha256((ROOT / "test_data/personas/spamuel/canon.yaml").read_bytes()).hexdigest()
    component = ROOT.parent / "pm-minecraft"
    receipt["minecraft_component"] = {
        "repository": str(component),
        "head": subprocess.check_output(["git", "-C", str(component), "rev-parse", "HEAD"], text=True).strip(),
        "working_changes": subprocess.check_output(["git", "-C", str(component), "status", "--short", "--untracked-files=no"], text=True).splitlines(),
        "sources": {name: hashlib.sha256((component / name).read_bytes()).hexdigest()
                    for name in ("body/src/bot.ts", "body/dist/bot.js", "body/src/server.ts", "body/dist/server.js",
                                 "mcmcp/body.py", "mcmcp/body_models.py", "mcmcp/runtime.py", "mcmcp/models.py")}}
    receipt["scope"] = ("Real protocol failures plus actual qwen/full-history resource adaptation; M3 remains separate."
                        if args.hazard == "none" else
                        f"Real protocol failures plus actual qwen/full-history {args.hazard} recovery; M3 remains separate.")
    if args.history_fixture:
        previous = json.loads((args.history_fixture / "receipt.json").read_text(encoding="utf-8"))
        if (args.hazard != "none" or args.reset_world or previous.get("hazard") != "none"
                or previous.get("seed") != receipt["seed"] or previous.get("item_grants") is not False):
            raise ValueError("Retained adaptation requires the same real seed without a reset or item grants.")
        receipt["history_fixture"] = str(args.history_fixture.resolve())
        receipt["history_receipt_sha256"] = hashlib.sha256((args.history_fixture / "receipt.json").read_bytes()).hexdigest()
    save()
    def fixture(command):
        code = ("import sys; from pathlib import Path; from mcrcon import MCRcon; "
                "v=dict(l.split('=',1) for l in Path(sys.argv[1]).read_text().splitlines() if '=' in l); "
                "r=MCRcon('127.0.0.1',v['MINECRAFT_RCON_PASSWORD'],port=int(v['MINECRAFT_RCON_PORT'])); "
                "r.connect(); print(r.command(sys.argv[2])); r.disconnect()")
        result = subprocess.run([values["PYTHON_EXE"], "-c", code, str(env_path), command],
                                check=True, capture_output=True, text=True)
        receipt["fixture_commands"].append({"command": command, "result": result.stdout, "time": time.time()})
        save()
    stack, app, restored = RealServerStack(server, out, env_path), None, []
    try:
        from urllib.request import urlopen
        chat = yaml.safe_load(args.config.read_text(encoding="utf-8"))["chat"]["Default"]
        with urlopen(chat["base_url"].rstrip("/") + "/models", timeout=30) as response:
            catalog = json.load(response)
        if chat["model"] not in {row["id"] for row in catalog["data"]}:
            raise ValueError("The selected model is absent from the live chat catalog.")
        receipt["model_identity"] = {"base_url": chat["base_url"], "model": chat["model"], "catalog": catalog}
        save()
        if args.reset_world:
            if properties["level-name"] != "world":
                raise ValueError("The real wipe procedure only supports level-name=world.")
            for name in ("world", "logs", ".fabric"):
                target_path = (server / name).resolve()
                if not target_path.is_relative_to(server) or target_path == server:
                    raise ValueError("A world reset target escaped the named server directory.")
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                            str(server / "wipe-world.ps1")], check=True)
        stack.start()
        if args.restore_fixture:
            from workflows.minecraft_body.journal import read_actor_journal
            previous = json.loads((args.restore_fixture / "receipt.json").read_text(encoding="utf-8"))
            rows = read_actor_journal(args.restore_fixture / "protocol/scratch/actor-journal.jsonl")
            for recorded in previous["fixture_commands"]:
                parts = recorded["command"].split()
                if len(parts) != 5 or parts[4] not in {"minecraft:stone", "minecraft:water", "minecraft:lava"}:
                    continue
                point = dict(zip(("x", "y", "z"), map(int, parts[1:4])))
                if not any(row.get("event") == "result" and row["tool"] == "minecraft_inspect_block"
                           and row["result"].get("position") == point and row["result"].get("block_name") == "air"
                           and row["time"] < recorded["time"] for row in rows):
                    raise ValueError("The prior fixture has no original air evidence for restoration.")
                fixture(" ".join(parts[:4] + ["minecraft:air"]))
            time.sleep(3)
            receipt["restored_fixture"] = str(args.restore_fixture)
            save()
        os.environ["MINECRAFT_MCP_URL"] = stack.mcp_url
        app = build_app(ROOT / "test_data/personas/spamuel/canon.yaml", out / "app", cognition="real",
            config_path=str(args.config.resolve()), checkpoint_path=out / "ghost.pickle", record_lifecycle=False)
        app.supervisor.on_error = lambda error: print(type(error).__name__, str(error), flush=True)
        tools = build_minecraft_tools(stack.mcp_url)
        tools["managed_instructions"] = lambda **kwargs: {"instruction": "Observe actual fixture failures.", "revision": 0}
        environment = FolderEnvironment(out / "protocol", "bash")
        runtime = WorkflowRuntime(environment, threading.Event(), tools)
        gateway = BodyGateway(stack.mcp_url, "a13-fixture", environment, runtime, tools)
        with BODY_LEASES.acquire(stack.mcp_url, gateway.owner):
            initial = gateway.call("minecraft_observe", {"include_image": False})
            receipt["initial_observation"] = initial["result"]
            schemas = tools["minecraft_tool_schemas"]()
            atomic_json(out / "live-tool-schemas.json", schemas)
            captured = json.loads((ROOT / "test_data/captures/alpha/a13-invalid-arguments.json").read_text(encoding="utf-8"))
            invalid = next(row for entry in captured["journals"] for row in entry["rows"]
                           if row.get("event") == "uncertain")
            before_sequence = gateway.sequence
            rejected = gateway.call(invalid["tool"], invalid["arguments"])
            after_sequence = gateway.sequence
            following = gateway.call("minecraft_observe", {"include_image": False})
            receipt["checks"]["argument_validation"] = (rejected.get("reason") == "invalid_arguments"
                and before_sequence == after_sequence and following["ok"]
                and gateway.gameplay.service_wait is None and not runtime.stop_event.is_set()
                and following["result"]["inventory"] == initial["result"]["inventory"]
                and following["result"]["body_session"] == initial["result"]["body_session"]
                and following["result"]["death_count"] == initial["result"]["death_count"])
            atomic_json(out / "argument-validation.json", {"captured": invalid, "rejected": rejected,
                "sequence_before": before_sequence, "sequence_after": after_sequence, "following_observation": following})
            save()
            feet = initial["result"]["camera"]["block_position"]
            x, y, z = (feet[key] for key in ("x", "y", "z"))
            target = None
            # Hills can put a nearby standable cell above the current feet.
            # Require its original support as well as empty body space.
            for rise in range(4):
                for dx, dz in ((2, 0), (-2, 0), (0, 2), (0, -2), (4, 0), (-4, 0), (0, 4), (0, -4)):
                    candidate = {"x": x + dx, "y": y + rise, "z": z + dz}
                    reads = [gateway.call("minecraft_inspect_block", {"position": {**candidate, "y": candidate["y"] + offset}})
                             for offset in (-1, 0, 1)]
                    if safe_landing(*(row["result"] for row in reads)) and all(
                            row["result"].get("block_name") == "air" for row in reads[1:]):
                        target = candidate
                        break
                if target is not None:
                    break
            if target is None:
                raise ValueError("No observed nearby empty cell fits the temporary route fixture.")
            for point in (target, {**target, "y": target["y"] + 1}, {"x": x, "y": y + 12, "z": z}):
                original = gateway.call("minecraft_inspect_block", {"position": point})
                if original["result"].get("block_name") != "air":
                    raise ValueError("The fixture requires originally empty temporary cells.")
                restored.append(point)
                fixture(f"setblock {point['x']} {point['y']} {point['z']} minecraft:stone")
            for name, tool, arguments in (
                ("blocked_navigation", "minecraft_walk_to_exact", target),
                ("unreachable_block", "minecraft_mine_block", {"position": {"x": x, "y": y + 12, "z": z}}),
                ("low_supplies", "minecraft_craft_item", {"item": "iron_pickaxe", "repetitions": 1}),
                ("ore_search", "minecraft_find_block", {"pattern": "deepslate_iron_ore"}),
            ):
                operands = next(row for row in schemas if row["name"] == tool)["parameters"]["properties"]
                if tool == "minecraft_walk_to_exact" and "position" in operands:
                    arguments = {"position": target}
                first, second = gateway.call(tool, arguments), gateway.call(tool, arguments)
                rejected = gateway.call(tool, arguments)
                atomic_json(out / (name + ".json"), {"first": first, "second": second, "rejected": rejected})
                receipt["checks"][name] = (rejected.get("reason") == "method_paused")
                save()
            # A resumed actor can already hold its self-crafted pickaxe. Select
            # an existing block item so this boundary still tests a wrong tool.
            held = initial["result"].get("held_item") or {}
            if held.get("name", "").endswith("_pickaxe"):
                harmless = next((item["name"] for item in initial["result"]["inventory"]
                    if item["name"] in {"dirt", "cobblestone", "birch_log", "oak_log", "birch_planks", "oak_planks"}), None)
                if harmless is None:
                    raise ValueError("The wrong-tool fixture requires an existing harmless held item.")
                equipped = gateway.call("minecraft_equip", {"item": harmless})
                if not equipped["ok"]:
                    raise RuntimeError("The fixture could not select its existing wrong-tool item.")
                atomic_json(out / "wrong-tool-setup.json", equipped)
            wrong = gateway.call("minecraft_mine_block", {"position": target})
            atomic_json(out / "wrong-tool.json", wrong)
            receipt["checks"]["wrong_tool"] = wrong["result"].get("can_harvest") is False
            if args.hazard == "none":
                second = gateway.call("minecraft_mine_block", {"position": target})
                rejected = gateway.call("minecraft_mine_block", {"position": target})
                atomic_json(out / "wrong-tool-repeated.json", {"first": wrong, "second": second, "rejected": rejected})
                receipt["checks"]["wrong_tool_paused"] = rejected.get("reason") == "method_paused"
                gateway.wait_idle()
        if args.hazard == "none":
            receipt["checks"].update(native_adaptation(app, environment, tools, target,
                {"x": x, "y": y + 12, "z": z}, out, args.timeout, args.history_fixture))
            death_lines = [line for line in (out / "services/server.log").read_text(encoding="utf-8").splitlines()
                           if any(f"{values['MINECRAFT_PLAYER']} {death}" in line
                                  for death in ("tried to swim in lava", "burned to death", "drowned", "fell from a high place",
                                                "hit the ground too hard", "suffocated in a wall", "was slain by"))]
            atomic_json(out / "life-audit.json", {"death_events": death_lines, "survived": not death_lines})
            receipt["checks"]["native_survived_adaptation"] = not death_lines
            receipt["status"] = "passed" if all(receipt["checks"].values()) else "failed"
            save()
            if receipt["status"] != "passed":
                raise AssertionError("A native adaptation check failed.")
            return
        with BODY_LEASES.acquire(stack.mcp_url, gateway.owner):
            # The earlier protocol boundary released its epoch. Rebuild the
            # gateway from its journal before claiming the hazard epoch.
            gateway = BodyGateway(stack.mcp_url, "a13-fixture", environment, runtime, tools)
            gateway.call("minecraft_observe", {"include_image": False})
            # Prepare liquid fixtures from the current observed body.
            current = gateway.call("minecraft_observe", {"include_image": False})["result"]
            if args.hazard == "water" and args.water_placement == "pit":
                preparation = contained_water_pit(gateway, current)
                water = preparation["cell"]
                receipt["water_pit"] = preparation
            elif args.hazard == "lava" and args.lava_placement == "edge":
                body = current["camera"]["block_position"]
                water = None
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    candidate = {"x": body["x"] + dx, "y": body["y"] - 1, "z": body["z"] + dz}
                    reads = [gateway.call("minecraft_inspect_block", {"position": {**candidate, "y": candidate["y"] + offset}})
                             for offset in (0, 1, 2)]
                    if reads[0]["result"].get("block_name") not in {"grass_block", "dirt"} or not all(
                            row["result"].get("block_name") == "air" for row in reads[1:]):
                        continue
                    gateway.call("minecraft_mine_block", {"position": candidate})
                    original = gateway.call("minecraft_inspect_block", {"position": candidate})
                    if original["result"].get("block_name") == "air":
                        water = candidate
                        break
                if water is None:
                    raise ValueError("No exposed adjacent soil cell fits a lava pit with a retreat route.")
            else:
                water = {**current["camera"]["block_position"], "y": current["camera"]["block_position"]["y"] + 1}
                original = gateway.call("minecraft_inspect_block", {"position": water})
                if original["result"].get("block_name") in {"grass", "tall_grass", "fern", "large_fern"}:
                    # Clear a dry plant through an ordinary survival action.
                    gateway.call("minecraft_mine_block", {"position": water})
                    current = gateway.call("minecraft_observe", {"include_image": False})["result"]
                    water = {**current["camera"]["block_position"], "y": current["camera"]["block_position"]["y"] + 1}
                    original = gateway.call("minecraft_inspect_block", {"position": water})
                if original["result"].get("block_name") != "air":
                    raise ValueError("The liquid source needs originally empty headroom.")
            receipt["hazard_cell"] = water
            restored.append(water)
            fixture(f"setblock {water['x']} {water['y']} {water['z']} minecraft:{args.hazard}")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not gateway.gameplay.recovery:
                time.sleep(0.5)
                gateway.call("minecraft_inspect_block", {"position": water})
                gateway.call("minecraft_observe", {"include_image": False})
            if not gateway.gameplay.recovery:
                raise AssertionError("The real liquid fixture did not establish a current hazard.")
            blocked = gateway.call("minecraft_craft_item", {"item": "iron_pickaxe", "repetitions": 1})
            receipt["checks"][f"optional_work_paused_in_{args.hazard}"] = blocked.get("reason") == "recovery_required"
            save()
            gateway.wait_idle()
        task = app.supervisor.start(action_id=f"a13-{args.hazard}-recovery", workflow_name="minecraft",
            instructions=(f"Recover from the current unexpected {args.hazard} exposure before optional work. "
                "Use the retained failures to choose a safe dry landing. Do not mine optional ore or craft the original iron pickaxe. "
                "Continue until two fresh observations show full air and stable health on dry supported ground. "
                "Use camera.block_position for exact body cells; do not round floating feet coordinates. "
                "Retreat beyond adjacent recorded lava cells before claiming safety. "
                "Inspect the feet, head and floor cells at the final landing. Use actor_diagnose to bind each recovery action. "
                "Save a handoff and finish at that safe landing."),
            arguments={"specialist": "recovery", "acceptance": {"history_journals": [str(environment._path('scratch/actor-journal.jsonl'))]}})
        receipt["native_task"] = task
        save()
        deadline = time.monotonic() + args.timeout
        while app.supervisor.status(task["run_id"])["state"] in {"accepted", "running"}:
            if time.monotonic() > deadline:
                raise TimeoutError("The actual native recovery exceeded its observation deadline.")
            time.sleep(0.5)
        app.supervisor.join(task["run_id"], 30)
        report_path = Path(task["artifact_path"]) / "outbox/report.yaml"
        report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
        receipt["native_report"] = report
        final = tools["minecraft_call"]("minecraft_observe", {"include_image": False})
        atomic_json(out / "independent-final.json", final)
        p = final["result"]["player"]
        receipt["checks"][f"native_{args.hazard}_recovery"] = (report["outcome"]["status"] == "complete"
            and report["gameplay"]["recovery"] is None and p["oxygen"] == 20 and p["on_ground"] and not p["in_water"])
        native_source = str(Path(task["artifact_path"]) / "scratch/actor-journal.jsonl")
        receipt["checks"]["native_changed_method"] = any(row["source"] == native_source for row in report["gameplay"]["diagnoses"])
        receipt["checks"]["native_life_unchanged"] = (final["result"]["body_session"] == receipt["initial_observation"]["body_session"]
            and final["result"]["death_count"] == receipt["initial_observation"]["death_count"])
        if args.hazard == "water" and args.water_placement == "pit":
            from workflows.minecraft_body.journal import read_actor_journal
            rows = read_actor_journal(Path(task["artifact_path"]) / "scratch/actor-journal.jsonl")
            first = next(row for row in rows if row.get("event") == "result" and row["tool"] == "minecraft_observe")
            receipt["checks"]["native_started_in_water"] = first["result"]["player"]["in_water"]
            receipt["checks"]["native_left_water_pit"] = final["result"]["camera"]["block_position"] != water
        if args.hazard == "lava" and args.lava_placement == "edge":
            body = final["result"]["camera"]["block_position"]
            receipt["checks"]["native_retreat_from_lava_edge"] = abs(body["x"] - water["x"]) + abs(body["z"] - water["z"]) > 1
        death_lines = [line for line in (out / "services/server.log").read_text(encoding="utf-8").splitlines()
                       if any(f"{values['MINECRAFT_PLAYER']} {death}" in line
                              for death in ("tried to swim in lava", "burned to death", "drowned", "fell from a high place",
                                            "hit the ground too hard", "suffocated in a wall", "was slain by"))]
        atomic_json(out / "life-audit.json", {"death_events": death_lines, "survived": not death_lines})
        receipt["checks"]["native_survived_hazard"] = not death_lines
        receipt["status"] = "passed" if all(receipt["checks"].values()) else "failed"
        save()
        if receipt["status"] != "passed":
            raise AssertionError("A real failure or native recovery check failed.")
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
        try:
            if stack.server is not None and stack.server.poll() is None:
                for point in restored:
                    fixture(f"setblock {point['x']} {point['y']} {point['z']} minecraft:air")
        finally:
            stack.stop()
        body_log = out / "character/body.log"
        if body_log.is_file():
            import shutil
            shutil.copy2(body_log, out / "services/body.log")
        save()
        print(out / "receipt.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument("--restore-fixture", type=Path)
    parser.add_argument("--history-fixture", type=Path, help="Retain a prior native adaptation's actual partial work without a world reset.")
    parser.add_argument("--hazard", choices=("water", "lava", "none"), default="water")
    parser.add_argument("--lava-placement", choices=("edge", "head"), default="edge")
    parser.add_argument("--water-placement", choices=("pit", "head"), default="pit")
    parser.add_argument("--reset-world", action="store_true")
    run(parser.parse_args())
