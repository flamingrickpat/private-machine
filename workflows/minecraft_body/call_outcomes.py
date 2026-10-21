"""Derive gameplay decisions from original calls without replaying mutations."""
from __future__ import annotations

import hashlib
import json
import math


SERVICE_REASONS = frozenset({"not_connected", "body_unavailable", "connection_error", "service_unavailable"})
READ_FAILURE_EXCEPTIONS = frozenset({"minecraft_find_block", "minecraft_find_path"})
DANGEROUS_CELLS = frozenset({"water", "lava", "bubble_column", "fire", "soul_fire", "powder_snow", "magma_block", "cactus"})
WALK_TOOLS = frozenset({"minecraft_walk_to_exact", "minecraft_walk_to_visible", "minecraft_walk_to_surface"})
MOTION_PROGRESS_BLOCKS = 0.5


def safe_landing(floor: dict, feet: dict, head: dict) -> bool:
    """Require support and dry body space from typed collision observations.

    Dry plants can have empty collision bounds. Their names do not imply
    an obstruction. Missing collision evidence cannot establish safe space.
    """
    return (floor.get("is_solid") is True and floor.get("block_name") not in DANGEROUS_CELLS
            and all(cell.get("is_solid") is False and cell.get("block_name")
                    and cell["block_name"] not in DANGEROUS_CELLS for cell in (feet, head)))


def signature(tool: str, arguments: dict) -> str:
    """Identify a method by its actual operands, excluding presentation limits."""
    operands = {key: value for key, value in arguments.items() if key not in {"include_image", "limit"}}
    return hashlib.sha256(json.dumps([tool, operands], sort_keys=True).encode()).hexdigest()


def motion_action(tool: str, arguments: dict) -> bool:
    """Identify requested translation while allowing input-release controls."""
    return tool in WALK_TOOLS or (tool == "minecraft_fine_control" and any(
        arguments.get(key) is True for key in ("forward", "back", "left", "right", "jump")))


def classify(row: dict, *, movement: float | None = None) -> dict:
    """Keep external effects separate from action success and leaf acceptance.

    Positive quantities establish partial progress even when the action fails.
    A successful tool call supplies a candidate, never a completion certificate.
    Unknown result fields cannot establish a quantity or a resource absence.
    Observed movement uses blocks. A half-block displacement counts as useful
    walking progress; smaller displacement remains a recorded world effect.
    Identical held-item identity and durability establish an equipment no-op.
    A new held item records an equipment change, without proving harvesting.
    """
    data = row.get("result", {})
    fine_motion = row["tool"] == "minecraft_fine_control" and motion_action(row["tool"], row.get("arguments", {}))
    if fine_motion:
        before, after = data.get("pose_before"), data.get("pose_after")
        movement = math.dist([before[key] for key in ("x", "y", "z")],
                             [after[key] for key in ("x", "y", "z")]) if before and after else None
    reason = data.get("reason") or (data.get("diagnostics") or {}).get("cause")
    quantities = {key: data[key] for key in ("logs_collected", "mined_count", "placed_count", "blocks_placed", "blocks_dug", "climbed", "depth_achieved", "repetitions_crafted", "input_consumed", "moved_count")
                  if isinstance(data.get(key), (int, float))}
    delta = data.get("inventory_delta") or {}
    made_progress = any(value > 0 for value in quantities.values()) or any(
        entry.get("delta", 0) > 0 for entry in delta.get("changes", []))
    changed = made_progress or bool(delta.get("changes"))
    if movement is not None:
        # Fine control makes intentional steps smaller than a walking hop.
        # Its input acknowledgement cannot prove movement through an obstacle.
        made_progress |= movement > 0 if fine_motion else movement >= MOTION_PROGRESS_BLOCKS
        changed |= movement > 0
    if row["tool"] == "minecraft_mine_block" and row.get("ok") and data.get("block"):
        changed = made_progress = True
    unsuccessful = not row.get("ok", False)
    if fine_motion and (movement is None or movement == 0):
        unsuccessful, reason = True, "motion_unobserved" if movement is None else "no_motion"
    if data.get("can_harvest") is False and reason not in {"unsafe_support", "target_changed"}:
        unsuccessful, reason = True, "wrong_harvesting_tool"
    if row["tool"] in {"minecraft_equip", "minecraft_equip_best_tool"} and row.get("ok"):
        previous, equipped = data.get("previous_held"), data.get("equipped")
        # The endpoint can succeed without changing equipment. Repeated
        # acknowledgements cannot acquire tools or establish useful progress.
        facts = lambda item: tuple(item.get(key) for key in ("name", "durability", "max_durability")) if item else None
        if facts(previous) == facts(equipped):
            unsuccessful, reason = True, "equipment_unchanged"
        else:
            changed = made_progress = True
            quantities["equipment_changed"] = 1
    if quantities and not any(value > 0 for value in quantities.values()):
        unsuccessful = True
        reason = reason or "zero_output"
    if row["tool"] == "minecraft_find_block" and data.get("total_matches") == 0:
        unsuccessful, reason = True, reason or "local_search_empty"
    if row["tool"] == "minecraft_find_path" and data.get("state") in {"blocked", "unknown"}:
        unsuccessful, reason = True, data["state"]
    if movement is not None:
        quantities["moved_blocks"] = movement
    died = action_death(data)
    if died:
        unsuccessful, reason, made_progress = True, "player_died", False
    kind = ("failed_recovery" if died else "transport_failure" if reason in SERVICE_REASONS else
            "partial_progress" if changed else "unsuccessful_action" if unsuccessful else "completion_candidate")
    return {"kind": kind, "unsuccessful": unsuccessful, "reason": reason,
            "quantities": quantities, "inventory_delta": delta,
            "made_progress": made_progress, "completion_verified": False}


def action_death(data: dict) -> dict | None:
    """Read an explicit same-body death transition from an action receipt.

    Missing lifecycle fields cannot establish death or survival.
    The original result remains intact for later inspection.
    """
    session = data.get("body_session")
    before, after = data.get("death_count_before"), data.get("death_count_after")
    if (isinstance(session, str) and session and isinstance(before, int) and not isinstance(before, bool)
            and isinstance(after, int) and not isinstance(after, bool) and 0 <= before < after):
        return {"body_session": session, "death_count": after}
    return None


def missing_inventory(before: list[dict], after: list[dict]) -> list[dict]:
    """Record missing observed quantities without claiming material consumption.

    A death can drop items. These counts do not establish their location,
    recovery, destruction, or placement as blocks.
    """
    counts = {row["name"]: row["count"] for row in after}
    return [{"item": row["name"], "before": row["count"], "after": counts.get(row["name"], 0),
             "missing": row["count"] - counts.get(row["name"], 0)} for row in before
            if row["count"] > counts.get(row["name"], 0)]


def lava_near_body(body: dict, lava: dict) -> bool:
    """Recognize a typed lava cell beside the floor, feet, or head.

    Nearby summaries do not locate every lava cell. Exact inspections and
    their typed neighbors can establish this immediate exposure.
    """
    return (abs(body["x"] - lava["x"]) + abs(body["z"] - lava["z"]) <= 1
            and -1 <= lava["y"] - body["y"] <= 1)


class GameplayEvidence:
    """Rebuild method history, search coverage, and sticky recovery from calls.

    The actor journal owns the facts. This projection never sends world calls.
    Private diagnosis records can permit one concrete changed method.
    Recovery ends only after fresh vitals and exact dry landing inspections.
    """

    def __init__(self, tool_schemas: list[dict] | tuple = ()) -> None:
        # Compare omitted options with their actual public defaults. Preserve
        # original arguments in journals and outcomes for later review.
        self.argument_defaults = {row["name"]: {name: field["default"]
            for name, field in row.get("parameters", {}).get("properties", {}).items() if "default" in field}
            for row in tool_schemas}
        self.observation: dict = {}
        self.observed_at = 0.0
        self.methods: dict[str, dict] = {}
        self.searches: list[dict] = []
        self.outcomes: list[dict] = []
        self.reads: list[dict] = []
        self.recovery: dict | None = None
        self.service_wait: dict | None = None
        self.permit: dict | None = None
        self.diagnoses: list[dict] = []
        self.stable_observations = 0
        self.epoch_started_at = 0.0
        self._source = ""
        self.position: dict | None = None
        self.position_at = 0.0
        self.navigation: dict = {"ineffective": 0, "last": None}
        self.harvesting: dict = {"ineffective": 0, "last_failure": None}
        self.life: dict | None = None
        self.failed_recoveries: list[dict] = []

    def method_key(self, tool: str, arguments: dict) -> str:
        """Compare action operands with defaults from the actual tool schema.

        An explicit default and its omission name the same method. A changed
        nondefault value stays distinct. Original receipts remain unchanged.
        """
        return signature(tool, {**self.argument_defaults.get(tool, {}), **arguments})

    def consume(self, row: dict, source: str) -> None:
        """Apply one original journal event in source order.

        Sequence numbers belong to one journal. Source paths prevent a new
        actor's sequence from colliding with retained calls from another actor.
        """
        from workflows.minecraft_body.gateway import READ_TOOLS, observed_target
        self._source = source
        if row.get("event") == "epoch_acquired":
            self.permit = None
            self.stable_observations = 0
            self.epoch_started_at = row["time"]
            return
        if row.get("event") == "diagnosis":
            self.diagnoses.append({**row, "source": source})
            self.permit = row
            return
        if row.get("event") == "uncertain":
            self.service_wait = {"source": source, "seq": row["seq"], "error": row["error"], "observed_at": row["time"]}
            self.outcomes.append({"kind": "transport_failure", "unsuccessful": True, "source": source,
                                  "seq": row["seq"], "time": row["time"], "tool": row["tool"],
                                  "arguments": row["arguments"], "reason": row["error"], "completion_verified": False})
            return
        if row.get("event") != "result":
            return
        data, tool = row["result"], row["tool"]
        position = data.get("camera", {}).get("feet_position") or data.get("final_position") or data.get("pose_after")
        displacement = (math.dist([self.position[key] for key in ("x", "y", "z")],
                                  [position[key] for key in ("x", "y", "z")])
                        if self.position and position and row["time"] >= self.position_at else None)
        if position and row["time"] >= self.position_at:
            self.position, self.position_at = position, row["time"]
        if (tool == "minecraft_observe" and row["ok"] and data.get("player", {}).get("on_ground") is True
                and displacement is not None and displacement >= MOTION_PROGRESS_BLOCKS):
            self.navigation["ineffective"] = 0
        if (self.permit and self.permit["next_tool"] == tool
                and self.method_key(tool, self.permit["next_arguments"]) == self.method_key(tool, row["arguments"])):
            self.permit = None
        if tool in READ_TOOLS:
            usable = row["ok"] or (tool == "minecraft_inspect_block" and observed_target(row, row["arguments"]["position"]))
            if usable:
                self.reads.append({**row, "source": source})
        if tool == "minecraft_observe" and row["ok"]:
            if row["time"] < self.observed_at:
                return
            self.observed_at = row["time"]
            self.service_wait = None
            session, deaths = data.get("body_session"), data.get("death_count")
            life = ({"body_session": session, "death_count": deaths}
                    if isinstance(session, str) and session and isinstance(deaths, int)
                    and not isinstance(deaths, bool) and deaths >= 0 else None)
            previous_life = self.life
            if previous_life and life and previous_life["body_session"] == session and deaths < previous_life["death_count"]:
                raise ValueError("A body's observed death counter cannot decrease within the same session.")
            self.life = life
            if previous_life and life and previous_life["body_session"] == session and deaths > previous_life["death_count"]:
                failure = {"kind": "failed_recovery", "reason": "player_died", "unsuccessful": True,
                           "made_progress": False, "completion_verified": False, "quantities": {}, "inventory_delta": {},
                           "source": source, "seq": row["seq"], "time": row["time"], "tool": tool,
                           "arguments": row["arguments"], "previous_life": previous_life, "current_life": life,
                           "failed_episode": self.recovery,
                           "inventory_missing_after_death": missing_inventory(self.observation.get("inventory", []), data.get("inventory", [])),
                           "inventory_scope": "Missing after observed death. These quantities do not establish consumption, block placement, destruction, drop location, or recovered items."}
                self.failed_recoveries.append(failure)
                self.outcomes.append(failure)
                self.recovery, self.permit = None, None
                self.start_recovery(["respawn_reconciliation"], row, source)
            elif self.recovery and life and self.recovery.get("life") != life:
                # A body restart cannot prove that the old hazard was survived.
                # Reconcile the new life with fresh geometry and retain that limit.
                self.recovery.update(life=life, started_at=row["time"], prior_life_unverified=True)
                self.stable_observations = 0
            previous = self.observation.get("player", {})
            self.observation = data
            player = data.get("player", {})
            causes = []
            # Full air does not establish a dry body. Unexpected water still
            # requires a supported landing before the original work resumes.
            if player.get("in_water") is True:
                causes.append("water")
            if player.get("oxygen", 20) < 20:
                causes.append("water_air")
            if player.get("health", 20) <= player.get("max_health", 20) / 4:
                causes.append("low_health")
            if player.get("food", 20) <= 6:
                causes.append("low_supplies")
            if previous and player.get("health", 20) < previous.get("health", 20):
                causes.append("damage")
            if causes:
                self.start_recovery(causes, row, source)
            safe = (player.get("health", 0) > player.get("max_health", 20) / 4
                    and player.get("oxygen") == 20 and player.get("on_ground") is True
                    and player.get("in_water") is False and player.get("food", 0) > 6
                    and player.get("health", 0) >= previous.get("health", 0))
            self.stable_observations = self.stable_observations + 1 if safe else 0
        diagnostics = data.get("diagnostics") or {}
        if diagnostics.get("head_block") in {"water", "lava"} or diagnostics.get("feet_block") == "lava":
            self.start_recovery(["liquid"], row, source)
        if tool == "minecraft_inspect_block":
            body = self.observation.get("camera", {}).get("block_position")
            if body:
                if data.get("block_name") == "water" and data.get("position") == {**body, "y": body["y"] + 1}:
                    self.start_recovery(["water"], row, source)
                for cell in [data, *data.get("neighbors", [])]:
                    point = cell.get("position")
                    if cell.get("block_name") == "lava" and point and lava_near_body(body, point):
                        self.start_recovery(["lava"], row, source)
                        known = self.recovery.setdefault("lava_cells", [])
                        if point not in known:
                            known.append(point)
        if self.recovery and self.recovery_proved():
            self.recovery = None
        # Routine reads stay in the journal, outside the meaningful outcome stream.
        if tool in READ_TOOLS and tool not in READ_FAILURE_EXCEPTIONS and data.get("reason") not in SERVICE_REASONS:
            return
        outcome = {**classify(row, movement=displacement if tool in WALK_TOOLS else None),
                   "source": source, "seq": row["seq"], "time": row["time"],
                   "tool": tool, "arguments": row["arguments"]}
        self.outcomes.append(outcome)
        if tool == "minecraft_mine_block" and data.get("can_harvest") is False and data.get("reason") == "unharvestable":
            self.harvesting["ineffective"] += 1
            self.harvesting["last_failure"] = {"source": source, "seq": row["seq"],
                "time": row["time"], "block": data.get("block"), "tool_used": data.get("tool_used")}
        dead_life = action_death(data)
        if dead_life:
            failure = {**outcome, "previous_life": self.life, "current_life": dead_life,
                       "failed_episode": self.recovery,
                       "inventory_missing_after_death": [{"item": change["item"], "before": change["before"],
                           "after": change["after"], "missing": -change["delta"]}
                           for change in data.get("inventory_delta", {}).get("changes", []) if change["delta"] < 0],
                       "inventory_scope": "Missing during an action with observed death. Negative deltas do not prove material consumption or block placement."}
            self.failed_recoveries.append(failure)
            self.life, self.recovery, self.permit = dead_life, None, None
            self.start_recovery(["respawn_reconciliation"], row, source)
            return
        if outcome["kind"] == "transport_failure":
            self.service_wait = {"source": source, "seq": row["seq"], "reason": outcome["reason"], "observed_at": row["time"]}
            return
        if motion_action(tool, row["arguments"]):
            # New coordinates alone cannot bypass a stalled local walk. Retain
            # measured partial travel, but require diagnosis after two failures
            # without a half-block displacement, even across walker types.
            if outcome["made_progress"]:
                self.navigation["ineffective"] = 0
            elif outcome["unsuccessful"]:
                self.navigation["ineffective"] += 1
            self.navigation["last"] = outcome
        key = self.method_key(tool, row["arguments"])
        method = self.methods.setdefault(key, {"tool": tool, "arguments": row["arguments"], "ineffective": 0})
        method.update(last=outcome, state=self.preconditions())
        method["ineffective"] = method["ineffective"] + 1 if outcome["unsuccessful"] and not outcome["made_progress"] else 0
        if tool == "minecraft_find_block":
            self.searches.append({"source": source, "seq": row["seq"], "observed_at": row["time"],
                "world": "configured_endpoint", "dimension": self.observation.get("player", {}).get("dimension", "unknown"),
                "method": "surface_exposed_loaded_cells", "pattern": row["arguments"].get("pattern"),
                "coverage": data.get("coverage"), "total_matches": data.get("total_matches"),
                "available": outcome["kind"] != "transport_failure" and (row["ok"] or data.get("reason") == "not_found"),
                "uncertainty": "Unloaded and buried cells remain unknown. This result applies only to the recorded coverage."})

    def start_recovery(self, causes: list[str], row: dict, source: str) -> None:
        """Latch a present hazard instead of changing back to the resource goal."""
        if self.recovery is None:
            self.recovery = {"causes": [], "source": source, "seq": row["seq"], "started_at": row["time"], "life": self.life}
        self.recovery["causes"] = sorted(set(self.recovery["causes"] + causes))
        self.stable_observations = 0

    def preconditions(self) -> dict:
        """Compare held tools, supplies, and the observed body location.

        View direction and clock changes cannot unlock a failed method.
        Only current-source reads can authorize a new external command.
        """
        camera = self.observation.get("camera", {})
        return {"position": camera.get("block_position"), "dimension": self.observation.get("player", {}).get("dimension"),
                "held_item": self.observation.get("held_item"), "inventory": self.observation.get("inventory", [])}

    def recovery_proved(self) -> bool:
        """Require stable vitals plus recent dry feet, headroom, and support.

        A movement success cannot prove an escape. Two fresh observations
        must show full air, ground support, sufficient food, and stable health.
        """
        if (not self.recovery or self.stable_observations < 2 or self.life is None
                or self.recovery.get("life") != self.life or self.current_failed_recovery()):
            return False
        point = self.observation.get("camera", {}).get("block_position")
        if point is None:
            return False
        cells = self.landing_cells(max(self.recovery["started_at"], self.epoch_started_at))
        x, y, z = (point[key] for key in ("x", "y", "z"))
        if not safe_landing(*(cells.get((x, y + offset, z), {}).get("result", {}) for offset in (-1, 0, 1))):
            return False
        # Dry body space alone cannot prove retreat from an adjacent lava edge.
        # A fresh changed cell can also prove that the recorded exposure ended.
        for lava in self.recovery.get("lava_cells", []):
            if lava_near_body(point, lava):
                read = cells.get(tuple(lava[key] for key in ("x", "y", "z")), {}).get("result", {})
                if not read.get("block_name") or read["block_name"] == "lava":
                    return False
        return True

    def landing_cells(self, since: float) -> dict:
        """Index original exact inspections and their typed adjacent cells.

        Neighbor positions and collision fields belong to the same receipt.
        They establish local geometry without inventing another tool call.
        Only reads from the current actor epoch can authorize recovery.
        """
        cells = {}
        for read in self.reads:
            if read["source"] != self._source or read["tool"] != "minecraft_inspect_block" or read["time"] < since:
                continue
            data = read["result"]
            observed = [data] + [{**neighbor, "is_solid": neighbor.get("solid")} for neighbor in data.get("neighbors", [])]
            for cell in observed:
                point = cell.get("position")
                if point:
                    cells[tuple(point[key] for key in ("x", "y", "z"))] = {**read, "result": cell}
        return cells

    def current_failed_recovery(self) -> dict | None:
        """Return an observed death in this actor epoch, without inferring survival."""
        return next((row for row in reversed(self.failed_recoveries)
                     if row["source"] == self._source and row["time"] >= self.epoch_started_at), None)

    def guard(self, tool: str, arguments: dict) -> dict | None:
        """Reject another unchanged failed method or optional hazardous work.

        The rejection sends no world call. A diagnosis must cite a fresh read
        and bind one exact next action. It cannot erase prior failures.
        After two wrong-tool failures, even changed mining coordinates need
        a fresh true harvestability read for that exact target. A diagnosis
        cannot override false or missing harvesting metadata.
        """
        from workflows.minecraft_body.gateway import READ_TOOLS
        failure = self.current_failed_recovery()
        if failure and tool not in READ_TOOLS and tool != "minecraft_stop":
            return {"ok": False, "reason": "death_requires_replan", "failed_recovery": failure,
                    "message": "No mutation was sent. The player died in this actor epoch. Return replan; respawn cannot prove escape or food recovery. Reconcile lost inventory and revise the leaf in a new attempt."}
        route = self.ready_route() if self.recovery else None
        if self.recovery and route and tool in {"minecraft_distance", "minecraft_relative", "minecraft_find_path"}:
            return {"ok": False, "reason": "recovery_action_ready", "message":
                    "No extra preview was sent. A reachable dry landing is already observed. "
                    "Observe current vitals, use actor_diagnose, then execute a concrete escape action.", "route": route}
        if self.service_wait and tool not in READ_TOOLS:
            return {"ok": False, "reason": "service_wait", "message":
                    "No action was sent. Restore the service and obtain a fresh body observation before mutation."}
        if tool == "minecraft_mine_block" and self.harvesting["ineffective"] >= 2:
            failure = self.harvesting["last_failure"]
            target = arguments.get("position")
            reads = [row for row in self.reads if row["source"] == self._source and row["time"] > failure["time"]]
            harvestable = None
            for row in reads:
                if row["tool"] == "minecraft_inspect_block" and row["result"].get("position") == target:
                    harvestable = row["result"].get("can_harvest_with_held")
                elif row["tool"] == "minecraft_find_block":
                    for cell in row["result"].get("results", []):
                        if cell.get("position") == target:
                            harvestable = cell.get("harvestable_with_held")
            if harvestable is not True:
                return {"ok": False, "reason": "harvesting_prerequisite",
                    "message": "No dig was sent. Changing coordinates cannot fix the wrong tool. "
                    "Inspect this exact target after equipping a suitable owned tool; can_harvest_with_held must be true. "
                    "Use best_tools and current recipe_search to rebuild missing prerequisites. "
                    "An accepted historical tool certificate does not mean you still own that tool. "
                    "After inventory loss, craft a table from four owned planks, place it outside your body, "
                    "then craft the required wooden/stone tools through their actual recipes. "
                    "Bare-hand wood and dirt still require a fresh true harvestability read.",
                    "target": target, "can_harvest_with_held": harvestable, "last_failure": failure}
        method = self.methods.get(self.method_key(tool, arguments))
        paused = method is not None and method["ineffective"] >= 2
        stalled = motion_action(tool, arguments) and self.navigation["ineffective"] >= 2
        recovery_action = self.recovery is not None and tool not in READ_TOOLS and tool != "minecraft_stop"
        if paused or stalled or recovery_action:
            if (self.permit and self.permit["next_tool"] == tool
                    and self.method_key(tool, self.permit["next_arguments"]) == self.method_key(tool, arguments)):
                self.permit = None
                return None
            return {"ok": False, "reason": "recovery_required" if recovery_action else "method_paused",
                    "message": "No action was sent. Do not repeat this call. Call actor_diagnose using diagnosis_call, "
                    "and add your diagnosis text explaining the cause and concrete method change. "
                    "An unchanged paused method still needs changed observed preconditions. "
                    "Stalled motion requires a fresh body observation.",
                    "equipment_guidance": "equipment_unchanged means the same item remains held, or no owned upgrade exists. "
                    "Repeating equip cannot gather or craft supplies. Inspect target harvestability; if the required pickaxe is missing, "
                    "find *_log, harvest reachable logs from a supported side, and use actual recipes to rebuild the missing workbench and tools.",
                    "recovery": self.recovery_guidance(),
                    "navigation": {**self.navigation, "position": self.position},
                    "paused_methods": self.paused_methods(tool, arguments),
                    "diagnosis_call": self.diagnosis_call(tool, arguments),
                    "evidence": "scratch/decision-evidence.json",
                    "full_history": "scratch/gameplay-evidence.json"}
        self.permit = None
        return None

    def paused_methods(self, tool: str | None = None, arguments: dict | None = None) -> list[dict]:
        """Return decision facts without repeating inventory and outcome history.

        Full preconditions and original outcomes remain in the disk projection.
        Repeated tool replies must leave context available for the next action.
        With a tool and arguments, return only that exact paused method.
        Without a tool, return every paused method.
        """
        methods = list(self.methods.values())
        if tool is not None:
            method = self.methods.get(self.method_key(tool, arguments or {}))
            methods = [method] if method else []
        return [{"tool": method["tool"], "arguments": method["arguments"],
                 "ineffective": method["ineffective"], "reason": method["last"]["reason"],
                 "source": method["last"]["source"], "seq": method["last"]["seq"]}
                for method in methods if method["ineffective"] >= 2]

    def diagnosis_call(self, tool: str, arguments: dict) -> dict:
        """Describe the private correction call without granting an action.

        Suggested sequence IDs name existing usable reads in this actor epoch.
        The actor must supply its own diagnosis. The diagnosis validator still
        checks freshness, changed preconditions, recovery purpose, and geometry.
        Missing reads require another observation before that private call.
        """
        failed_at = max((row["time"] for row in self.outcomes if row["unsuccessful"]), default=0)
        fresh = [row for row in self.reads if row["source"] == self._source and row["time"] > failed_at]
        body = next((row for row in reversed(fresh) if row["tool"] == "minecraft_observe"), None)
        chosen = ([body] if body else []) + fresh[-7:]
        sequences = list(dict.fromkeys(row["seq"] for row in chosen))
        return {"tool": "actor_diagnose", "arguments": {"evidence_sequences": sequences,
                "next_tool": tool, "next_arguments": arguments, "recovery_action": self.recovery is not None},
                "required_text": "Add diagnosis with the observed cause and how this action changes the method or reaches safety.",
                "read_first": "minecraft_observe" if not sequences else None}

    def decision_snapshot(self) -> dict:
        """Expose current safety facts without replaying old preconditions.

        The complete projection and original journals retain every method,
        search, and outcome. This artifact guides the next actor decision.
        It cannot establish world effects or completion.
        """
        return {"schema": "pm.gameplay-actor-decision.v1", "recovery": self.recovery_guidance(),
                "player": self.observation.get("player", {}), "position": self.position,
                "held_item": self.observation.get("held_item"), "inventory": self.observation.get("inventory", []),
                "service_wait": self.service_wait, "failed_recovery": self.current_failed_recovery(),
                "navigation": {**self.navigation, "position": self.position},
                "harvesting": self.harvesting,
                "paused_methods": self.paused_methods(), "recent_outcomes": self.outcomes[-5:],
                "full_history": "scratch/gameplay-evidence.json", "journal": "scratch/actor-journal.jsonl"}

    def ready_route(self) -> dict | None:
        """Find a reachable landing with original dry body-space inspections.

        Known safe geometry makes repeated numeric previews ineffective.
        This hint sends no action and cannot establish actual arrival.
        """
        for read in reversed(self.reads):
            if read["source"] != self._source or read["tool"] != "minecraft_find_path" or read["result"].get("state") != "reachable":
                continue
            point = read["arguments"]
            if not all(key in point for key in ("x", "y", "z")):
                continue
            import math
            x, y, z = (math.floor(point[key]) for key in ("x", "y", "z"))
            cells = self.landing_cells(self.epoch_started_at)
            floor, feet, head = (cells.get((x, y + offset, z), {}) for offset in (-1, 0, 1))
            if safe_landing(*(cell.get("result", {}) for cell in (floor, feet, head))):
                return {"target": point, "evidence_sequences": [floor["seq"], feet["seq"], head["seq"], read["seq"]]}
        return None

    def recovery_guidance(self) -> dict | None:
        """Expose current recovery facts beside every live tool result.

        Current server rules can distinguish food from health recovery.
        These hints cannot prove healing or replace fresh body observations.
        """
        if self.recovery is None:
            return None
        body = self.observation.get("camera", {}).get("block_position")
        body_cells = ({name: {**body, "y": body["y"] + offset}
                       for name, offset in (("floor", -1), ("feet", 0), ("head", 1))} if body else None)
        rules = next((row["result"].get("server_rules", "") for row in reversed(self.reads)
                      if row["source"] == self._source and row["tool"] == "minecraft_info"), "")
        health = self.observation.get("player", {}).get("health")
        healing = ({"health": health, "food": self.observation.get("player", {}).get("food"),
                    "natural_regeneration": False,
                    "next_step": "Ordinary food, full hunger, saturation, dandelions, and sleep cannot heal damage with natural regeneration off. "
                    "Nutrition penalties do not establish a healing effect. Obtain an actual healing effect, such as golden_apple regeneration, "
                    "through survival actions and current recipe evidence. Recovery can require gathering and crafting its prerequisites. "
                    "Keep a supported route. Prove increased HEALTH through fresh observations; absorption is not restored health."}
                   if "natural regen off" in rules and "low_health" in self.recovery.get("causes", []) else None)
        return {**self.recovery, "current_body_cells": body_cells, "health_recovery": healing,
                "next_step": "Use current_body_cells exactly; do not round the floating feet position. "
                "Observe current vitals. Inspect one reachable dry landing's floor, feet, and head. "
                "Use actor_diagnose with current journal sequences and recovery_action=true, then execute its exact escape action. "
                "Use the feet cell above the floor as the destination. Use walk_to_exact for that dry landing; "
                "walk_to_visible can stop beside it in water. A route preview cannot prove dry body space. "
                "Retreat beyond adjacent recorded lava cells or obtain fresh evidence that those cells changed. "
                "More numeric previews cannot end recovery.",
                "observed_route": self.ready_route()}

    def diagnosis(self, diagnosis: str, evidence_sequences: list[int], next_tool: str,
                  next_arguments: dict, recovery_action: bool) -> dict:
        """Validate a targeted diagnosis and one relevant next method.

        Evidence must come from this actor after the latest ineffective call.
        Unchanged operands also require changed observed preconditions.
        During recovery, the caller must explain how the action supports safety.
        """
        if self.current_failed_recovery():
            raise ValueError("The player died in this actor epoch. Return replan; another diagnosis cannot prove survival.")
        if not diagnosis.strip() or not evidence_sequences:
            raise ValueError("A diagnosis needs a cause and fresh read sequences.")
        method = self.methods.get(self.method_key(next_tool, next_arguments))
        last_failure = max((row["time"] for row in self.outcomes if row["unsuccessful"]), default=0)
        fresh_reads = [row for row in self.reads if row["source"] == self._source and row["time"] > last_failure]
        reads = [row for row in fresh_reads if row["seq"] in evidence_sequences]
        if {row["seq"] for row in reads} != set(evidence_sequences):
            invalid = sorted(set(evidence_sequences) - {row["seq"] for row in reads})
            available = [{"seq": row["seq"], "tool": row["tool"]} for row in fresh_reads[-8:]]
            raise ValueError(f"All evidence sequences must cite fresh usable reads after the latest ineffective action. "
                             f"Invalid sequences: {invalid}. Available fresh reads: {available}. "
                              "Describe older failures in diagnosis text; cite current read sequences as evidence.")
        if (motion_action(next_tool, next_arguments) and self.navigation["ineffective"] >= 2
                and not any(row["tool"] == "minecraft_observe" for row in reads)):
            raise ValueError("Stalled walking needs a fresh body observation after the latest ineffective action. "
                             "This also applies to stalled fine control. Cite its sequence, inspect the local route, and bind a changed method.")
        if method and method["ineffective"] >= 2 and method["state"] == self.preconditions():
            raise ValueError("The same operands and observed preconditions remain paused. "
                             "Observe current equipment and inventory after a change, or select a changed method.")
        if self.recovery is not None and not recovery_action:
            raise ValueError("Recovery remains active. The next action must support recovery.")
        if self.recovery is not None and next_tool in {"minecraft_walk_to_exact", "minecraft_walk_to_visible"}:
            import math
            point = next_arguments.get("position", next_arguments)
            x, y, z = (math.floor(point[key]) for key in ("x", "y", "z"))
            cells = self.landing_cells(max(self.recovery["started_at"], self.epoch_started_at))
            if not safe_landing(*(cells.get((x, y + offset, z), {}).get("result", {}) for offset in (-1, 0, 1))):
                raise ValueError("Inspect the destination's solid floor and dry feet/head cells before recovery navigation. "
                                 "Target the feet cell above the floor, not the occupied floor cell.")
        return {"event": "diagnosis", "diagnosis": diagnosis, "evidence_sequences": evidence_sequences,
                "next_tool": next_tool, "next_arguments": next_arguments, "recovery_action": recovery_action}

    def snapshot(self) -> dict:
        """Supply cumulative evidence with original journal references.

        Failed methods and search coverage remain across actor replacements.
        The latest twenty outcomes give context; full history stays on disk.
        """
        return {"schema": "pm.gameplay-evidence.v1", "recovery": self.recovery_guidance(), "service_wait": self.service_wait,
                "life": self.life, "failed_recoveries": self.failed_recoveries,
                "navigation": {**self.navigation, "position": self.position, "observed_at": self.position_at},
                "harvesting": self.harvesting,
                "methods": list(self.methods.values()), "searches": self.searches,
                "outcome_count": len(self.outcomes), "recent_outcomes": self.outcomes[-20:],
                "diagnoses": self.diagnoses, "completion_verified": False}
