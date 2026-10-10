"""Verify M3 through the generic project and the real full-history companion."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOAL = (
    "Build a small usable wooden house near the initial spawn and craft an iron pickaxe from self-mined iron. "
    "Use ordinary survival actions and no village supplies, supplied iron, supplied tools, or operator commands. "
    "Choose a surveyed stable site for a modest 5 by 5 outer footprint, supported floor, two-block-high walls, "
    "complete roof, a wooden door in a two-cell entrance, and a clear 3 by 3 interior. "
    "Keep workstations outside the reserved interior and entrance. "
    "Verify exact cells for floor, walls, roof, both door cells, interior air and entrance approach support. "
    "Demonstrate entry and exit in separate verified leaves; use the interior center for entry, "
    "with position radius 0.7 so the target has enough room for the body. "
    "Gather and craft counted materials. Craft a suitable stone pickaxe before mining identified exposed iron ore. "
    "Obtain at least three raw iron from actual iron ore mining. Smelt that same raw iron in a furnace, "
    "then craft an iron pickaxe. Use separate inventory-verifiable mining, smelting, and crafting leaves. "
    "Root acceptance must cite the accepted mining and smelting leaves and require an iron pickaxe in current inventory. "
    "Retain the ore-to-raw-iron-to-ingot-to-pickaxe evidence chain. Do not take resources from village containers. "
    "Save the home and a tested return route. Adapt failed searches through changed local coverage; do not scan buried cells. "
    "Prioritize hazard recovery and useful partial work. Do not restart completed material work after a changed method."
)

ROUTE_TEST = {"nominal_blocks": 10, "minimum_blocks": 8, "maximum_blocks": 12,
              "measurement": "Horizontal distance between actual observed feet positions, from the starting home observation.",
              "sequence": "Home observation, safe excursion observation, home return observation, then durable route note; one actor attempt and body life."}


def route_trial(start: dict, excursion: dict, returned: dict) -> bool:
    """Check an observed round trip with room for ordinary walking error.

    The gate measures horizontal displacement from the first home position.
    A block-cell target, saved waypoint, or static position cannot prove travel.
    Terrain height can change; every observation must remain grounded and dry.
    """
    distance = math.hypot(excursion["position"]["x"] - start["position"]["x"],
                          excursion["position"]["z"] - start["position"]["z"])
    return (start["safe"] and start["home"] and excursion["safe"] and returned["safe"] and returned["home"]
            and ROUTE_TEST["minimum_blocks"] <= distance <= ROUTE_TEST["maximum_blocks"]
            and start["life"] == excursion["life"] == returned["life"]
            and start["life"][0] is not None and start["life"][1] is not None)


def m3_geometry(tree) -> dict:
    """Derive the house contract from the actor's proposed roof and door.

    These coordinates are planning operands. They do not prove world effects.
    """
    from itertools import product
    errors = []
    planks, doors = set(), set()
    for node in tree.nodes:
        for condition in node.criteria:
            if condition.kind != "block" or condition.dimension != "overworld":
                continue
            cells = {(p.x, p.y, p.z) for p in ([condition.position] if condition.position else condition.positions)}
            if all(name.endswith("_planks") for name in condition.names):
                planks.update(cells)
            if all(name.endswith("_door") and name != "iron_door" for name in condition.names):
                doors.update(cells)
    roofs = []
    for y in {cell[1] for cell in planks}:
        cells = {cell for cell in planks if cell[1] == y}
        xs, zs = sorted({c[0] for c in cells}), sorted({c[2] for c in cells})
        if len(xs) == len(zs) == 5 and xs == list(range(xs[0], xs[0] + 5)) and zs == list(range(zs[0], zs[0] + 5)):
            if cells == set(product(xs, [y], zs)):
                roofs.append((y, xs, zs))
    if not roofs:
        return {"errors": ["Declare a complete exact-cell 5 by 5 wooden roof before deriving the house contract."]}
    roof_y, xs, zs = max(roofs)
    wall_y = roof_y - 2
    perimeter = {(x, y, z) for x, y, z in product(xs, [wall_y, wall_y + 1], zs)
                 if x in (xs[0], xs[-1]) or z in (zs[0], zs[-1])}
    lower = sorted(cell for cell in doors if cell[1] == wall_y and cell in perimeter
                   and not (cell[0] in (xs[0], xs[-1]) and cell[2] in (zs[0], zs[-1]))
                   and (cell[0], cell[1] + 1, cell[2]) in doors)
    if len(lower) != 1:
        return {"errors": ["Declare one two-cell wooden door on a wall, away from the corners."]}
    x, y, z = lower[0]
    door = {(x, y, z), (x, y + 1, z)}
    approach = (x + (-1 if x == xs[0] else 1 if x == xs[-1] else 0), y,
                z + (-1 if z == zs[0] else 1 if z == zs[-1] else 0))
    geometry = {
        "floor": set(product(xs, [roof_y - 3], zs)), "walls": perimeter - door,
        "roof": set(product(xs, [roof_y], zs)), "door": door,
        "interior": set(product(xs[1:-1], [wall_y, wall_y + 1], zs[1:-1])),
        "approach_air": {approach, (approach[0], y + 1, approach[2])},
        "approach_support": {(approach[0], y - 1, approach[2])},
    }
    return {"errors": errors, "bounds": {"x": [xs[0], xs[-1]], "z": [zs[0], zs[-1]], "feet_y": wall_y},
            "entry": (xs[2], wall_y, zs[2]), "exit": approach,
            "cells": {key: sorted(value) for key, value in geometry.items()}}


def m3_plan_constraints(tree) -> dict:
    """Reject missing M3 acceptance operands without changing the native tree."""
    errors = []
    root = next(node for node in tree.nodes if node.id == tree.goal_id)
    geometry = m3_geometry(tree)
    errors.extend(geometry["errors"])
    supports = {"grass_block", "dirt", "stone", "cobblestone", "deepslate", "andesite", "diorite", "granite"}
    def covers(nodes, family):
        cells = set()
        for node in nodes:
            for condition in node.criteria:
                if condition.kind != "block" or condition.dimension != "overworld":
                    continue
                names = condition.names
                allowed = (all(n.endswith("_planks") for n in names) if family in {"floor", "walls", "roof"}
                           else all(n.endswith("_door") and n != "iron_door" for n in names) if family == "door"
                           else all(n in supports or n.endswith("_planks") for n in names) if family == "approach_support"
                           else names == ["air"])
                if allowed:
                    cells.update((p.x, p.y, p.z) for p in ([condition.position] if condition.position else condition.positions))
        return cells
    if not geometry["errors"]:
        for family, cells in geometry["cells"].items():
            required = set(map(tuple, cells))
            for label, nodes in (("construction leaves", [n for n in tree.nodes if n.kind == "leaf"]), ("root fresh checks", [root])):
                missing = sorted(required - covers(nodes, family))
                if missing:
                    errors.append(f"{label}: {family} lacks {len(missing)} exact cells; first missing {missing[:3]}.")
        for label in ("entry", "exit"):
            target = tuple(geometry[label])
            matched = [n.id for n in tree.nodes if n.kind == "leaf" and any(
                c.kind == "position" and c.dimension == "overworld" and c.radius <= 0.7 and
                (c.position.x, c.position.y, c.position.z) == target for c in n.criteria)]
            if not matched:
                errors.append(f"Declare a separate verified {label} leaf at {target}, position radius at most 0.7.")
    def inventory_nodes(item, count):
        return [n.id for n in tree.nodes if n.kind == "leaf" and any(
            c.kind == "inventory" and c.names == [item] and c.minimum >= count for c in n.criteria)]
    stages = {"mining": inventory_nodes("raw_iron", 3), "smelting": inventory_nodes("iron_ingot", 3)}
    accepted = {c.node_id for c in root.criteria if c.kind == "accepted"}
    for stage, ids in stages.items():
        if not ids:
            errors.append(f"Declare an inventory-verifiable {stage} leaf for three self-mined iron units.")
        elif not accepted.intersection(ids):
            errors.append(f"Root acceptance must directly cite an accepted {stage} leaf: {ids}.")
    if not any(c.kind == "inventory" and c.names == ["iron_pickaxe"] and c.minimum >= 1 for c in root.criteria):
        errors.append("Root acceptance must freshly require iron_pickaxe >= 1 in current inventory.")
    return {"errors": errors, "requirements": {
        "scope": "Caller M3 contract operands, not evidence that the world is complete. Preserve accepted materials and tools.",
        "house": geometry, "iron_leaves": stages,
        "durable_home_and_route": "Save a home waypoint at the house or entrance. In one actor attempt, observe at home, choose a safe destination about ten horizontal blocks away, walk there and observe, then return home and observe. The gate accepts an excursion observation eight to twelve horizontal blocks from the starting home feet position. All three observations must be healthy, grounded, dry, and in the same body life. Then save the actual tested route with minecraft_remember(kind='routes'). These effects are checked before project publication.",
        "route_test": ROUTE_TEST,
    }}


def m3_project_check(root: Path, tree, initial: dict, character: Path) -> dict:
    """Check actual iron provenance and durable home and return-route effects."""
    from runtime.minecraft_actor import read_actor_journal
    from workflows.minecraft_tasks.tree import TreeStore
    store, geometry = TreeStore(root), m3_geometry(tree)
    chain = iron_chain(root, initial)
    errors = list(geometry["errors"])
    if not chain["verified"]:
        errors.append("Retain actual ordered self-mined ore, raw iron, smelting, and pickaxe crafting evidence; supplied iron cannot count.")
    attempts = [(attempt, [r for r in read_actor_journal(store.path_for(attempt["path"] + "/scratch/actor-journal.jsonl"))
                           if r.get("event") == "result"]) for attempt in store.state["attempts"]]
    waypoint_file = character / "memory/waypoints.json"
    waypoints = json.loads(waypoint_file.read_text(encoding="utf-8")) if waypoint_file.is_file() else []
    homes, routes, observed_attempts = [], [], []
    if not geometry["errors"]:
        bounds, exit_cell = geometry["bounds"], tuple(geometry["exit"])
        def at_home(position):
            x, y, z = (position[k] for k in ("x", "y", "z"))
            return abs(y - bounds["feet_y"]) < 0.8 and (
                bounds["x"][0] <= x <= bounds["x"][1] + 1 and bounds["z"][0] <= z <= bounds["z"][1] + 1
                or (x - exit_cell[0]) ** 2 + (z - exit_cell[2]) ** 2 <= 1)
        for attempt, rows in attempts:
            observations = []
            for row in rows:
                data = row["result"]
                if row["tool"] == "minecraft_add_waypoint" and row["ok"] and data.get("ok"):
                    waypoint = data["waypoint"]
                    if waypoint["dimension"].removeprefix("minecraft:") == "overworld" and at_home(waypoint["position"]):
                        if any(w["id"] == waypoint["id"] and w["position"] == waypoint["position"] for w in waypoints):
                            homes.append({"attempt": attempt["id"], "receipt": row, "file": str(waypoint_file)})
                if row["tool"] == "minecraft_observe" and row["ok"]:
                    player, position = data["player"], data["camera"]["feet_position"]
                    life = (data.get("body_session"), data.get("death_count"))
                    safe = player["health"] > 0 and player["on_ground"] and not player["in_water"] and player["dimension"] == "overworld"
                    observations.append({"row": row, "position": position, "life": life, "safe": safe, "home": at_home(position)})
                if row["tool"] == "minecraft_remember" and row["arguments"].get("kind") == "routes" and row["ok"] and data.get("ok"):
                    file = (character / data["file"]).resolve()
                    if not file.is_relative_to(character.resolve()) or not file.is_file():
                        continue
                    notes = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.startswith("{")]
                    if not any(note["note_id"] == data["note_id"] and note["markdown"] == row["arguments"]["markdown"] for note in notes):
                        continue
                    trials = [(start, a, b) for i, start in enumerate(observations)
                              for j, a in enumerate(observations[i + 1:], i + 1) for b in observations[j + 1:]
                              if route_trial(start, a, b)]
                    if trials:
                        start, a, b = trials[-1]
                        distance = math.hypot(a["position"]["x"] - start["position"]["x"], a["position"]["z"] - start["position"]["z"])
                        routes.append({"attempt": attempt["id"], "start": start["row"], "excursion": a["row"], "return": b["row"],
                                       "horizontal_distance": distance, "note": row, "file": str(file)})
            if observations:
                observed_attempts.append({"attempt": attempt["id"], "leaf_id": attempt["leaf_id"],
                                          "observations": [{key: value for key, value in item.items() if key != "row"}
                                                           | {"seq": item["row"]["seq"]} for item in observations]})
    if not homes:
        errors.append("Save an actual home waypoint at the finished house or entrance; its original receipt and durable waypoint must match.")
    if not routes:
        errors.append("The tested return route is still missing. In one actor attempt and body life, observe at home, walk about ten horizontal blocks away and observe eight to twelve blocks from that starting feet position, then return home and observe. All three observations must be healthy, grounded, and dry. Save the tested route with minecraft_remember(kind='routes'); its original receipt must match the durable note. Static safety checks or a saved note alone cannot replace this trip.")
    return {"errors": errors, "checks": {"iron_chain": chain["verified"], "saved_home": bool(homes), "tested_saved_route": bool(routes)},
            "requirements": {"route_test": ROUTE_TEST, "recent_attempt_observations": observed_attempts[-3:]},
            "iron": chain, "homes": homes, "routes": routes}


def recovered_smelt_matches(pending: dict, placed_furnaces: list[dict], furnace: dict, recovered_count: int) -> bool:
    """Match observed retained output to an uncertain, self-supplied iron batch.

    Only a native furnace placement before that dispatch can qualify. The
    observed remaining raw input and recovered output must equal that batch.
    This does not certify an arbitrary container withdrawal as ore mining.
    """
    slots = {slot["slot"]: slot for slot in furnace.get("slots", [])}
    raw, output = slots.get(0, {}), slots.get(2, {})
    return (furnace.get("position") in placed_furnaces and raw.get("item") == "raw_iron"
        and output.get("item") == "iron_ingot" and output.get("count") == recovered_count
        and raw.get("count", 0) + recovered_count == pending["arguments"].get("input_count", 1))


def iron_chain(root: Path, initial: dict) -> dict:
    """Audit positive mining, smelting, and crafting deltas in original order.

    Fresh world and inventory evidence exclude supplied iron. Container
    withdrawals remain separate effects and cannot establish ore provenance.
    """
    from runtime.minecraft_actor import read_actor_journal
    from workflows.minecraft_tasks.tree import TreeStore
    store = TreeStore(root)
    rows = [row for attempt in store.state["attempts"]
            for row in read_actor_journal(store.path_for(attempt["path"] + "/scratch/actor-journal.jsonl"))
            if row.get("event") in {"result", "uncertain"}]
    mined, smelted, crafted, chain = 0, 0, 0, []
    forbidden = []
    placed_furnaces, pending_smelt = [], None
    for row in rows:
        if row["event"] == "uncertain":
            if row["tool"] == "minecraft_smelt_item" and row["arguments"].get("input") == "raw_iron":
                pending_smelt = {"receipt": row, "furnaces": list(placed_furnaces)}
            continue
        data, tool = row["result"], row["tool"]
        valid_recovery = False
        delta = {entry["item"]: entry["delta"] for entry in (data.get("inventory_delta") or {}).get("changes", [])}
        raw_gain = max(0, delta.get("raw_iron", 0))
        block = data.get("block")
        ore = block.get("block_name") if isinstance(block, dict) else block
        if tool == "minecraft_build" and row["ok"]:
            placed_furnaces.extend(cell["position"] for cell in data.get("placed", []) if cell["block_name"] == "furnace")
        if tool in {"minecraft_mine_block", "minecraft_mine_vein"} and raw_gain:
            if ore not in {"iron_ore", "deepslate_iron_ore"}:
                forbidden.append("A mining inventory gain lacks an identified iron-ore target.")
                continue
            mined += raw_gain
            chain.append({"stage": "mine", "receipt": row})
        if tool == "minecraft_smelt_item" and data.get("input_item") == "raw_iron":
            count = max(0, delta.get("iron_ingot", 0))
            recovered = data.get("recovered_output") or {}
            recovered_count = recovered.get("count", 0) if recovered.get("name") == "iron_ingot" else 0
            if recovered_count:
                # Only reconcile an uncertain native batch in a newly placed
                # furnace. Existing village output cannot establish provenance.
                furnace = data.get("furnace_before") or {}
                valid_recovery = pending_smelt is not None and recovered_smelt_matches(
                    pending_smelt["receipt"], pending_smelt["furnaces"], furnace, recovered_count)
                if not valid_recovery:
                    forbidden.append("Recovered furnace output lacks a matching uncertain native batch in a newly placed furnace.")
                else:
                    chain.append({"stage": "reconcile_smelt", "uncertain_receipt": pending_smelt["receipt"], "furnace_before": furnace})
                    pending_smelt = None
            new_output = data.get("output") or {}
            new_count = new_output.get("count", 0) if new_output.get("name") == "iron_ingot" else 0
            if count and count != new_count + recovered_count:
                forbidden.append("The iron inventory gain differs from explicit new and recovered furnace output.")
            if data.get("input_consumed", 0) > mined - smelted:
                forbidden.append("Smelting consumed raw iron without preceding mining evidence.")
            smelted += count
            chain.append({"stage": "smelt", "receipt": row})
        if tool == "minecraft_craft_item" and data.get("resolved_item") == "iron_pickaxe" and row["ok"]:
            crafted += max(0, delta.get("iron_pickaxe", 0))
            if smelted < 3 or delta.get("iron_ingot", 0) > -3:
                forbidden.append("Pickaxe crafting lacks three preceding smelted ingots.")
            chain.append({"stage": "craft", "receipt": row})
        if raw_gain and tool not in {"minecraft_mine_block", "minecraft_mine_vein"}:
            # Recovering unused input is not another mining gain. Its exact
            # quantity must follow the independently matched interrupted batch.
            furnace = data.get("furnace_before") or {}
            retained_raw = sum(slot["count"] for slot in furnace.get("slots", [])
                               if slot["slot"] == 0 and slot.get("item") == "raw_iron")
            returned_raw = max(0, retained_raw - data.get("input_consumed", 0))
            if tool != "minecraft_smelt_item" or not valid_recovery or raw_gain != returned_raw:
                forbidden.append("Raw iron entered inventory outside mining or a matched interrupted smelt.")
        if delta.get("iron_ingot", 0) > 0 and tool != "minecraft_smelt_item":
            forbidden.append("Iron ingots entered inventory outside smelting.")
    initial_iron = [item for item in initial.get("inventory", []) if item["name"] in {"raw_iron", "iron_ingot", "iron_pickaxe"}]
    return {"verified": not initial_iron and not forbidden and mined >= 3 and smelted >= 3 and crafted >= 1,
            "mined_raw_iron": mined, "smelted_ingots": smelted, "crafted_pickaxes": crafted,
            "initial_iron": initial_iron, "forbidden": forbidden, "chain": chain}


def run(args) -> None:
    """Retain the generic project receipt even when M3 or provenance fails."""
    from scripts.alpha_a11_minecraft_gate import run as run_project
    args.item, args.goal, args.plan_only, args.source_tree = "A13", GOAL, False, None
    args.plan_constraints, args.project_check = m3_plan_constraints, m3_project_check
    args.out = args.out.resolve()
    if args.out.exists():
        raise FileExistsError("Use a fresh evidence directory; retained receipts cannot be overwritten.")
    try:
        run_project(args)
    finally:
        path = args.out / "receipt.json"
        if path.is_file():
            receipt = json.loads(path.read_text(encoding="utf-8"))
            task_root = receipt.get("task_root")
            if task_root and (Path(task_root) / "planning/state.json").is_file():
                from workflows.minecraft_tasks.tree import TreeStore
                store = TreeStore(Path(task_root))
                chain = iron_chain(Path(task_root), receipt.get("initial_observation", {}))
                (args.out / "iron-chain.json").write_text(json.dumps(chain, indent=2), encoding="utf-8")
                receipt["checks"]["self_mined_smelted_crafted_iron"] = chain["verified"]
                if not chain["verified"]:
                    receipt["status"] = "failed"
                if store.state["revision"]:
                    character = Path(receipt["character_root"])
                    contract = m3_plan_constraints(store.load())
                    completion = m3_project_check(Path(task_root), store.load(), receipt.get("initial_observation", {}), character)
                    (args.out / "m3-evidence.json").write_text(json.dumps({"plan_contract": contract, "completion": completion}, indent=2), encoding="utf-8")
                    receipt["checks"].update(m3_contract_coverage=not contract["errors"], **completion["checks"])
                    if contract["errors"] or completion["errors"]:
                        receipt["status"] = "failed"
            receipt["milestone"] = "M3"
            path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if receipt["status"] != "passed":
        raise AssertionError("M3 or its independent iron provenance remains unestablished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--server", type=Path, default=Path("C:/source/minecraft-server-real"))
    parser.add_argument("--reset-world", action="store_true")
    parser.add_argument("--resume-from", type=Path, help="Resume a stopped M3 project without a world reset or a new inventory baseline.")
    parser.add_argument("--allow-model-change", action="store_true", help="Record an explicit model transition while preserving task, world, checkpoint, and embeddings.")
    parser.add_argument("--decision-generation", type=Path, help="JSON generation settings for direct decisions; result schema and routing stay fixed.")
    parser.add_argument("--timeout", type=float, default=10800)
    run(parser.parse_args())
