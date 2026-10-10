"""Read independent world evidence and recompute generic acceptance conditions."""
from __future__ import annotations

import fnmatch
import json
import math
import time
from pathlib import Path
from pm_workflows.python_role import RoleContext

from runtime.body_lease import BODY_LEASES
from workflows.minecraft_tasks.schemas import Condition, Node
from workflows.minecraft_tasks.tree import TreeStore


def observe_conditions(context: RoleContext, conditions: list[Condition]) -> list[dict]:
    """Collect independent world reads and guarded prior leaf certificates.

    World reads use serialized body command boundaries. A verifier has no
    movement or construction tools. Earlier achievements retain their original
    specification and attempt identity. Actor statements cannot supply proof.
    """
    runtime = context.runtime
    endpoint = runtime.tools["minecraft_endpoint"]()
    reads = []
    with BODY_LEASES.command(endpoint, runtime.stop_event):
        observation = runtime.tools["minecraft_call"]("minecraft_observe", {"include_image": False})
    observed_dimension = observation["result"].get("player", {}).get("dimension") if observation["ok"] else None
    for condition in conditions:
        runtime.check_cancelled()
        if condition.kind == "accepted":
            from workflows.minecraft_tasks.tree import TreeStore, digest
            store = TreeStore(runtime.environment._path("."))
            record = store.state["acceptances"].get(condition.node_id)
            valid = store.valid(condition.node_id)
            source = json.loads(store.path_for(record["verification"]).read_text(encoding="utf-8")) if valid else {}
            specification = next((node.model_dump(mode="json") for node in store.load().nodes if node.id == condition.node_id), None)
            reads.append({"criterion_id": condition.id, "tool": "accepted_task",
                          "arguments": {"node_id": condition.node_id}, "time": time.time(),
                          "ok": valid, "result": {
                              "node_id": condition.node_id, "verified": valid,
                              "certificate_digest": digest(record) if record else "",
                              "verification": record["verification"] if record else "",
                              "source_specification": specification, "source_identity": source.get("identity"),
                              "source_checks": source.get("checks"),
                              "source_observed_times": [row["time"] for row in source.get("reads", [])]}})
            continue
        if condition.kind == "block" and condition.positions:
            for point in condition.positions:
                arguments = {"position": point.model_dump(), "include_image": False}
                with BODY_LEASES.command(endpoint, runtime.stop_event):
                    reply = runtime.tools["minecraft_call"]("minecraft_inspect_block", arguments)
                reads.append({"criterion_id": condition.id, "tool": "minecraft_inspect_block", "arguments": arguments,
                              "result": reply["result"], "ok": reply["ok"], "time": time.time(),
                              "observed_dimension": observed_dimension})
            continue
        if condition.kind in {"inventory", "position", "alive", "food", "health"}:
            tool, arguments = "minecraft_observe", {"include_image": False}
        elif condition.kind == "block":
            tool, arguments = "minecraft_inspect_block", {"position": condition.position.model_dump(), "include_image": False}
        else:
            tool, arguments = "minecraft_analyze_area", {"start": condition.position.model_dump(), "end": condition.end.model_dump()}
        if tool == "minecraft_observe" and observation is not None:
            reply = observation
        else:
            with BODY_LEASES.command(endpoint, runtime.stop_event):
                reply = runtime.tools["minecraft_call"](tool, arguments)
            if tool == "minecraft_observe":
                observation = reply
        reads.append({"criterion_id": condition.id, "tool": tool, "arguments": arguments,
                      "result": reply["result"], "ok": reply["ok"], "time": time.time(),
                      "observed_dimension": observed_dimension})
    runtime.check_cancelled()
    return reads


def evaluate(conditions: list[Condition], reads: list[dict], *, certificates: dict | None = None) -> dict[str, bool]:
    """Require a matching original read for every condition; fail closed."""
    checks = {}
    for condition in conditions:
        candidates = [row for row in reads if row.get("criterion_id") == condition.id]
        if condition.kind == "block" and condition.positions:
            positions = [point.model_dump() for point in condition.positions]
            checks[condition.id] = (len(candidates) == len(positions)
                and {json.dumps(row.get("arguments", {}).get("position"), sort_keys=True) for row in candidates}
                    == {json.dumps(point, sort_keys=True) for point in positions}
                and all(evaluate([Condition.model_validate({**condition.model_dump(), "position": row["arguments"]["position"], "positions": []})], [row])[condition.id]
                        for row in candidates))
            continue
        if len(candidates) != 1:
            checks[condition.id] = False
            continue
        row, passed = candidates[0], False
        data = row["result"]
        if condition.kind == "accepted" and row["tool"] == "accepted_task" and row["ok"]:
            from workflows.minecraft_tasks.tree import digest
            record = (certificates or {}).get(condition.node_id)
            passed = (record is not None and row["arguments"] == {"node_id": condition.node_id}
                      and data.get("node_id") == condition.node_id and data.get("verified") is True
                      and data.get("certificate_digest") == digest(record)
                      and data.get("verification") == record["verification"])
        elif condition.kind == "inventory" and row["tool"] == "minecraft_observe" and row["ok"]:
            count = sum(item["count"] for item in data.get("inventory", [])
                        if any(fnmatch.fnmatchcase(item["name"], name) for name in condition.names))
            passed = count >= condition.minimum
        elif condition.kind == "alive" and row["tool"] == "minecraft_observe" and row["ok"]:
            passed = data.get("player", {}).get("health", 0) > 0
        elif condition.kind in {"food", "health"} and row["tool"] == "minecraft_observe" and row["ok"]:
            player = data.get("player", {})
            value = player.get(condition.kind)
            passed = (isinstance(value, (int, float)) and not isinstance(value, bool)
                      and condition.minimum <= value <= 20
                      and player.get("dimension") == condition.dimension)
        elif condition.kind == "position" and row["tool"] == "minecraft_observe" and row["ok"]:
            point = data.get("camera", {}).get("feet_position")
            passed = bool(point) and data["player"]["dimension"] == condition.dimension and math.dist(
                [point[axis] for axis in ("x", "y", "z")],
                [getattr(condition.position, axis) for axis in ("x", "y", "z")]) <= condition.radius
        elif condition.kind == "block" and row["tool"] == "minecraft_inspect_block":
            position = condition.position.model_dump()
            passed = (row["arguments"].get("position") == position and data.get("position") == position
                      and data.get("block_name") in condition.names
                      and (row["ok"] or data.get("reason") == "not_found" and data.get("block_name") == "air"))
        elif condition.kind == "region" and row["tool"] == "minecraft_analyze_area" and row["ok"]:
            start, end = condition.position.model_dump(), condition.end.model_dump()
            volume = math.prod(abs(start[axis] - end[axis]) + 1 for axis in start)
            # The MCP omits empty cells from blocks and combines all three
            # air variants in air_cells. That count cannot prove one variant.
            air_names = {"air", "cave_air", "void_air"}
            aggregate_air = air_names.issubset(condition.names)
            count = sum(block["count"] for block in data.get("blocks", [])
                        if block["block_name"] in condition.names
                        and not (aggregate_air and block["block_name"] in air_names))
            air_cells = data.get("air_cells")
            air_known = not aggregate_air or type(air_cells) is int and 0 <= air_cells <= volume
            if aggregate_air and air_known:
                count += air_cells
            bounds = {"min": {axis: min(start[axis], end[axis]) for axis in start},
                      "max": {axis: max(start[axis], end[axis]) for axis in start}}
            passed = (air_known and row["arguments"] == {"start": start, "end": end} and data.get("unseen_cells") == 0
                      and data.get("bounds") == bounds and data.get("total_cells") == volume and count >= condition.minimum)
        if condition.kind in {"block", "region"}:
            passed = passed and row.get("observed_dimension") == condition.dimension
        checks[condition.id] = bool(passed)
    return checks


def accepted_actor_matches(store: TreeStore, node: Node, report: dict) -> bool:
    """Bind retained actor calls to their original acceptance identity.

    Exact specification content and unchanged files must pass ``store.valid``.
    Native restoration can change the current revision counter without changing
    that content. The original certificate owns the actor's revision counters.
    This function sends no world calls and changes no evidence.
    """
    if not store.valid(node.id):
        return False
    identity = store.state["acceptances"][node.id]["identity"]
    if report["identity"] != identity:
        return False
    actor = next((attempt for attempt in store.state["attempts"]
                  if attempt["id"] == identity["actor_attempt"]), None)
    if (actor is None or actor["leaf_id"] != node.id
            or actor["spec_revision"] != identity["spec_revision"]
            or actor["tree_revision"] != identity["tree_revision"]):
        return False
    import yaml
    outcome = yaml.safe_load(store.path_for(actor["report"]).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in store.path_for(actor["journal"]).read_text(encoding="utf-8").splitlines()]
    dispatches = {row["seq"]: row for row in rows if row["event"] == "dispatch"}
    results = [row for row in rows if row["event"] == "result"]
    reference = outcome.get("task_reference", {})
    return (outcome["outcome"]["status"] == "complete" and bool(results)
            and all(reference.get(key) == identity[key]
                    for key in ("task_id", "tree_revision", "leaf_id", "spec_revision", "actor_attempt"))
            and len(dispatches) == len(results) and len({row["seq"] for row in results}) == len(results)
            and all(row["seq"] in dispatches and dispatches[row["seq"]]["tool"] == row["tool"]
                    and dispatches[row["seq"]]["arguments"] == row["arguments"] for row in results))


def verified_project(root: Path) -> dict:
    """Recompute retained evidence for public claims without issuing world calls.

    This establishes verified attainment at the recorded time. Current world
    truth requires a new observation. Changed files invalidate the claim.
    """
    store = TreeStore(root)
    if not store.state["revision"]:
        return {"verified": False, "reason": "No durable task specification."}
    tree = store.load()
    if not store.valid(tree.goal_id):
        return {"verified": False, "reason": "Acceptance evidence or prerequisite guards changed."}
    for node in tree.nodes:
        if not store.valid(node.id):
            return {"verified": False, "reason": "A task has no valid acceptance."}
        record = store.state["acceptances"][node.id]
        report = json.loads((store.root / record["verification"]).read_text(encoding="utf-8"))
        checks = evaluate(node.criteria, report["reads"], certificates=store.state["acceptances"])
        if (not checks or not all(checks.values()) or report["review"]["status"] != "accepted"
            or set(report["review"]["criterion_ids"]) != {row.id for row in node.criteria}):
            return {"verified": False, "reason": "Independent observations do not meet acceptance."}
        if node.kind == "leaf" and not accepted_actor_matches(store, node, report):
            return {"verified": False, "reason": "Original actor calls do not support this completion candidate."}
    return {"verified": True, "goal_id": tree.goal_id, "tree_revision": tree.revision,
            "accepted_nodes": list(store.state["acceptances"]), "scope": "retained independent world observations"}
