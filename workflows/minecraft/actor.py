"""Run a sustained pm-coder specialist and retain independently read effects."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict
from typing import Literal
from pm_workflows.drivers import PmCoderDriver

from runtime.minecraft_actor import ActorGateway, READ_TOOLS
from runtime.mcp_services import serve_services
from workflows.minecraft.schemas import MinecraftRequest
from workflows.minecraft.steps import build_world

RESOURCES = Path(__file__).with_name("resources")


class ActorOutcome(BaseModel):
    """A model completion candidate. Tool receipts establish actual effects."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["complete", "yield", "blocked", "replan"]
    summary: str
    handoff: str


def verify_wood_return(receipts: list[dict], initial: dict, final: dict, acceptance: dict) -> dict:
    """Require distinct harvested trees, retained logs, travel, and return.

    Only actual tool results count. A model statement cannot add a tree.
    Coordinates and inventory come from fresh observations.
    """
    trees = {}
    for receipt in receipts:
        data = receipt["result"]
        if receipt["tool"] == "minecraft_harvest_tree" and receipt["ok"] and data.get("logs_collected", 0) > 0:
            base = data.get("base")
            if base:
                # A partial harvest can expose a higher base in the same trunk.
                # Recovery must not count that column as another physical tree.
                trees[(base["x"], base["z"])] = base
    def logs(data):
        return sum(item["count"] for item in data.get("inventory", [])
                   if item["name"].endswith(("_log", "_wood", "_stem", "_hyphae")))
    home, end = initial["camera"]["feet_position"], final["camera"]["feet_position"]
    def distance(point):
        return math.dist([home[key] for key in ("x", "y", "z")], [point[key] for key in ("x", "y", "z")])
    traveled = max([distance(row["result"]["camera"]["feet_position"]) for row in receipts
                    if row["result"].get("camera", {}).get("feet_position")] or [0])
    return_distance = distance(end)
    checks = {"five_distinct_trees": len(trees) >= acceptance.get("trees", 5),
              "retained_logs": logs(final) > logs(initial),
              "nontrivial_travel": traveled >= acceptance.get("minimum_travel", 16),
              "returned": return_distance <= acceptance.get("return_radius", 3),
              "same_dimension": final["player"]["dimension"] == initial["player"]["dimension"],
              "alive_on_ground": final["player"]["health"] > 0 and final["player"]["on_ground"]
                                 and not final["player"].get("in_water", False)}
    return {"verified": all(checks.values()), "checks": checks, "trees": len(trees),
            "tree_bases": [[base[key] for key in ("x", "y", "z")] for base in trees.values()], "initial_logs": logs(initial),
            "final_logs": logs(final), "maximum_travel": traveled, "return_distance": return_distance}


def leaf_execution_context(request: MinecraftRequest) -> dict:
    """Keep current leaf operands separate from historical actor summaries.

    The task reference identifies this attempt. Earlier thresholds and memory
    claims cannot replace these current criteria or independent verification.
    """
    current = {"identity": request.task_reference, "instruction": request.goal or request.instruction,
               "criteria": request.acceptance.get("criteria", [])}
    observations = request.acceptance.get("current_criterion_observations")
    if observations:
        current["current_criterion_observations"] = observations
    constraints = request.acceptance.get("project", {}).get("caller_constraints")
    if constraints is not None:
        current["caller_constraints"] = constraints
    check = request.acceptance.get("project", {}).get("latest_project_check")
    if check is not None:
        current["latest_project_check"] = check
    return current


def run_actor(context) -> dict:
    """Hold one body lease through the native session and external cleanup.

    The configured model session preserves pm-coder context discovery and
    automatic compaction. The actor never sends a public companion reply.
    """
    runtime, environment = context.runtime, context.runtime.environment
    payload = yaml.safe_load(environment.read_text("inbox/request.yaml"))
    request = MinecraftRequest.model_validate({key: value for key, value in payload.items()
                                             if not key.startswith("_public_")})
    endpoint = runtime.tools["minecraft_endpoint"]()
    gateway = ActorGateway(endpoint, context.task_id, environment, runtime, dict(runtime.tools))
    request.goal = runtime.tools["managed_instructions"]()["instruction"] or request.goal or request.instruction
    current_leaf = leaf_execution_context(request)
    environment.write_text("inbox/leaf-specification.json", json.dumps(current_leaf, ensure_ascii=False, indent=2))
    with runtime.tools["minecraft_lease"]():
        try:
            initial = gateway.call("minecraft_observe", {"include_image": request.include_image})
            if not initial["ok"]:
                raise RuntimeError("The actor has no usable initial observation.")
            server_info = gateway.call("minecraft_info", {})
            if not server_info["ok"]:
                raise RuntimeError("The actor cannot read current server rules.")
            environment.write_text("inbox/server-rules.json", json.dumps(server_info["result"], ensure_ascii=False, indent=2))
            previous = request.acceptance.get("previous_attempt", {})
            sources = [Path(path) for path in previous.get("journals", [])]
            history_sources = [Path(path) for path in request.acceptance.get("history_journals", [])]
            reconciliation = gateway.reconcile(sources, history_sources=history_sources)
            # Retained receipts define the original wood-return home across epochs.
            original = next((row for row in gateway.receipts if row["tool"] == "minecraft_observe" and row["ok"]), None)
            baseline = original["result"] if original else initial["result"]
            reconciliation["original_observation"] = baseline
            environment.write_text("inbox/reconciliation.json", json.dumps(reconciliation, ensure_ascii=False, indent=2))
            schemas = gateway.tool_schemas
            allowed = set(request.allowed_tools) if request.allowed_tools else {row["name"] for row in schemas}
            # Scripts can hide multiple actions from the instruction boundary.
            # Operator resets are not part of the configured survival action space.
            allowed -= {"minecraft_execute_typescript", "minecraft_suicide"}
            schemas = [row for row in schemas if row["name"] in allowed]
            # Keep observed deployment behavior beside the callable tool.
            # A long specialist resource can be overlooked during selection.
            environment.write_text("inbox/original-tool-schemas.json", json.dumps(schemas, ensure_ascii=False, indent=2))
            schemas = [dict(row) for row in schemas]
            for row in schemas:
                if row["name"] == "minecraft_inspect_block":
                    row["description"] = row.get("description", "") + (
                        "\nAn explicitly observed air cell returns ok=false, reason=not_found, "
                        "block_name=air, and the requested position. This proves that exact cell is air. "
                        "It is not a transport failure. Missing position or block_name does not prove air.")
                if row["name"] == "minecraft_analyze_area":
                    row["description"] = row.get("description", "") + (
                        "\nstart and end are absolute world block coordinates, with inclusive bounds. "
                        "air_cells combines air, cave_air, and void_air; unseen_cells must be zero for complete coverage. "
                        "Use an exact block inspection when a criterion requires one specific air variant.")
                if row["name"] == "minecraft_mine_block":
                    row["description"] = row.get("description", "") + (
                        "\nInspect the exact target before digging and skip air. "
                        "The current body supports digging grass and tall_grass. "
                        "is_solid describes collision bounds; plants with empty bounds can still be dug. "
                        "Observe the resulting cell and preserve required support below it.")
            environment.write_text("inbox/tool-schemas.json", json.dumps(schemas, ensure_ascii=False, indent=2))
            environment.write_text("inbox/persona.yaml", yaml.safe_dump(request.persona_context, allow_unicode=True))
            resources = [RESOURCES / "shared.md", RESOURCES / "navigator.md", RESOURCES / f"{request.specialist}.md"]
            skill = "\n\n".join(path.read_text(encoding="utf-8") for path in dict.fromkeys(resources))
            skill += ("\n\nYou are a private specialist of the same companion. Do not address the user. "
                      "Read inbox/persona.yaml and inbox/request.yaml. Tools use the live MCP schemas. "
                      "Read inbox/server-rules.json before choosing recovery. It contains current server rules, not assumed vanilla mechanics. "
                      "If natural regeneration is off, full food and waiting cannot heal damage. Check enabled nutrition effects. "
                      "Ordinary food and saturation do not heal with natural regeneration off. Nutrition penalties are not healing effects. "
                      "Golden apples supply a regeneration effect. Check their actual recipe and safely gather and craft its prerequisites. "
                      "A flower alone is not food or healing. Prove increased health; absorption does not satisfy a health threshold. "
                      "Obtain ordinary healing items through supported survival actions and actual recipe evidence when health remains low. "
                      "Read inbox/leaf-specification.json for this attempt's exact current instruction and criteria. "
                      "Re-read that file before completion. Check every criterion's current operands, not an earlier threshold or memory claim. "
                      "Perform only this leaf's remaining outcomes and required safety actions. "
                      "Retain already correct world state instead of rebuilding it. "
                      "Do not add travel or occupy a target cell unless a criterion or necessary physical action requires it. "
                      "An explicit air inspection with matching position and block_name=air is usable evidence even when ok=false and reason=not_found. "
                      "After current reads support every required outcome, save the handoff and return a completion candidate. "
                      "The parent verifier obtains its own fresh reads and can reject that candidate. "
                      "Missing support, unseen cells, and failed exact reads require actual repair or replan. "
                      "Accepted references in acceptance.criteria cite parent-verified earlier leaves. "
                      "Read their original specification, identity, and checks in acceptance.prerequisites. "
                      "Those certificates record past attainment. Perform this leaf's remaining world outcomes. "
                      "Read acceptance.project before choosing work locations; its ancestor criteria reserve future construction cells. "
                      "Its caller_constraints supplies original caller operands and publication audits. "
                      "Use the requirements relevant to this leaf; they do not authorize unrelated project work or prove completion. "
                      "Walk targets identify block cells. Compare actual returned feet positions with world-point criteria and caller spatial bounds. "
                      "Keep workbenches outside reserved floors, interiors, and entrance clearance. "
                      "A previous_attempt handoff records earlier work. Current specifications and fresh observations take precedence. "
                      "Read inbox/reconciliation.json. It retains earlier calls and fresh reads after interruption. "
                      "Its opening uncertain_world_effects lists unresolved calls across all earlier actors. "
                      "Inspect their known targets before replacing apparently missing supplies. Inventory absence does not prove loss. "
                      "An interrupted smelt can retain input, fuel, and output inside its furnace. "
                      "Candidate furnace positions come from earlier native placements; inspect their current block before approaching. "
                      "Use minecraft_use_block to inspect its window slots. Resume only remaining input with enough fuel; one log smelts 1.5 items. "
                      "Furnace output uses minecraft_smelt_item, not chest withdrawal. Set input_count to the remaining raw input in slot 0. "
                      "That tool collects earlier output before loading the new batch. Pass fuel_count sufficient for all remaining input. "
                      "Its original_observation records the original home and inventory baseline across restarts. "
                      "Never repeat an uncertain mutation from its old plan. Choose work from the current state. "
                      "Read scratch/decision-evidence.json for current safety and next-action facts. "
                      "acceptance.gameplay_history supplies selected earlier methods and searches. "
                      "Full history remains in inbox/gameplay-history.json and scratch/gameplay-evidence.json. Read only relevant older entries when needed. "
                      "The compact history uses method_fields and outcome_fields to label its rows; "
                      "resolve journal source_index through source_paths, method state_index through method_states, and tool_index through tool_names. "
                      "In v4, resolve each state's inventory_indexes and held_item_index through inventory_items; null means no held item. "
                      "Resolve outcome kind_index through outcome_kinds; additional_fields retains changed last-result operands. "
                      "In planning history v3 and v4, recent_outcomes method_index references that methods row's exact last outcome. "
                      "Search entries inherit search_defaults, with explicit entry values taking precedence. "
                      "Zip method_state_fields with each method_states row when supplied. "
                      "Search coverage_index selects search_coverages: zip each [field_set_index, values] row with search_coverage_field_sets[field_set_index]; a null index keeps its raw value. "
                      "With search_coverage_vector_fields, zip its axes with each coverage center, bounds.min and bounds.max array to restore exact coordinates. "
                      "After two ineffective attempts, inspect the cause and use actor_diagnose with fresh journal_sequence values. "
                      "Bind the diagnosis to a concrete changed next_tool and next_arguments. A different explanation cannot unlock unchanged work. "
                      "If a call returns diagnosis_call, call actor_diagnose with those operands and add your own diagnosis text. "
                      "The supplied sequence IDs cite existing current-epoch reads. Do not repeat the rejected call before diagnosis succeeds. "
                      "next_tool uses the upstream Minecraft name, such as minecraft_craft_item; the client prefix minecraft_actor_ is normalized. "
                      "Every evidence_sequences value must name a fresh usable read after the latest failure. "
                      "Describe earlier failed calls in diagnosis text, then cite only the new read sequences. "
                      "Two stalled walks or zero-motion fine controls pause movement across tool and destination changes. "
                      "Fine-control success acknowledges inputs; compare pose_before and pose_after. Small actual steps count as progress. "
                      "Observe fresh body position and vitals, inspect the local route, then bind a diagnosed changed method. "
                      "Before excavation or movement, identify floor, feet, and head from camera.block_position. "
                      "Never mine the floor beneath yourself to approach ore. Clear the next feet and head cells from a supported side. "
                      "Never place a block inside your current feet or head cells. Step aside before constructing walls or tables. "
                      "After death, accepted tool certificates are historical evidence; inspect current owned inventory. "
                      "If tools or workstations are missing, rebuild their recipe prerequisites before approaching ore. "
                      "Inspect target harvestability with the currently held item. New coordinates do not fix a wrong tool. "
                      "Use owned dirt or planks for scaffolding; mining stone and replacing that same floor adds no route. "
                      "A bridge floor belongs one block BELOW the destination feet, with two dry clear body cells above it. "
                      "Preserve a return step and inspect the landing before entering a cave opening. "
                      "Prefer short supported steps over fine controls toward an unknown drop. "
                      "A guessed coordinate at the same obstruction does not establish a new route. "
                      "Recipes show a concrete ingredient variant. When planks are missing but logs are owned, "
                      "query recipe_search for *_planks with craftable_now=true, craft the matching planks, then query the required item again. "
                      "Count all required planks for the table, sticks, and pickaxe before converting logs or crafting sticks. "
                      "When gameplay.recovery is active, pause optional work. Call actor_diagnose before each recovery mutation, "
                      "set recovery_action=true, and explain how it reaches safety. Continue recovery through changed methods. "
                      "If gameplay.failed_recovery reports a death, return replan immediately. Respawn cannot prove escape or food recovery. "
                      "Recovery clears only after two fresh observations show full oxygen, stable health, ground support, "
                      "and food above six, plus fresh inspections of dry feet/head cells and a solid safe floor. "
                      "Use camera.block_position for exact body cells. Do not round floating feet coordinates. "
                      "Recovery guidance supplies current_body_cells for floor, feet, and head. "
                      "An inspected adjacent lava edge also requires retreat or fresh evidence that the lava cell changed. "
                      "Oxygen uses the live 0..20 bubble scale. Low health is one quarter of max_health. "
                      "A negative search applies only to its recorded loaded exposed-cell coverage. "
                      "Nearby-block summaries can include buried cells and do not identify exposed ore. "
                      "Locate ore through exposed-cell find_block results, then inspect its actual exposed_faces before mining. "
                      "Do not probe buried ore coordinates from the nearby-block summary. "
                      "This configured actor uses text evidence. It cannot interpret screenshot pixels. "
                      "Images are retained as separate artifacts. After an instruction_changed result, "
                      "discard the pending plan and obey the new instruction. Call actor_instructions at milestones. "
                      "Write a concise task card to scratch/task-card.md and a durable handoff to outbox/handoff.md. "
                      "Continue until the requested leaf succeeds, yields, needs replanning, or meets a real blocker. "
                      "Return only JSON with status (complete, yield, blocked, replan), summary, and handoff. "
                      "A wood-return check requires distinct trunk columns (x,z) with positive harvest_tree logs, travel, and return. "
                      "Repeated partial harvests at higher bases in the same column count as ONE physical tree. "
                      "Do not finish after merely locating trees. Save starting coordinates for the return.")
            skill_path = context.kernel_data / "actor-skill.md"
            skill_path.write_text(skill, encoding="utf-8")
            if runtime.classification_session is None and runtime.agent_model is None:
                raise ValueError("A Minecraft actor needs a configured model session.")
            from contextlib import nullcontext
            session = (nullcontext(runtime.agent_model) if runtime.agent_model is not None else
                       runtime.classification_session("Default", False, runtime.stop_event))
            with session as model, serve_services({"minecraft_actor": gateway.server(schemas)}) as servers:
                # The shared classification session disables thinking by default.
                # Sustained execution needs its own model copy with thinking enabled.
                model = replace(model, enable_thinking=True)
                # Cover the body's 600-second action budget and cleanup reads.
                # The native loader must honor this per-server timeout.
                servers["minecraft_actor"]["requestTimeoutMs"] = 660000
                config = context.kernel_data / "actor-mcp.json"
                config.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
                agent_runtime = replace(runtime, agent_model=model, mcp_servers=servers)
                driver = PmCoderDriver(log_root=context.kernel_data / "actor-agent")
                result = driver.run_session(run_id=context.task_id + "-actor", attempt=1,
                    skill=str(skill_path), prompt="Execute the Minecraft leaf in inbox/request.yaml. "
                    "Current leaf operands, which take precedence over retained notes: "
                    + json.dumps(current_leaf, ensure_ascii=False, separators=(",", ":")) + "\n"
                    "Unresolved world effects from retained actors; reconcile these before replacing missing resources: "
                    + json.dumps(reconciliation["uncertain_world_effects"], ensure_ascii=False, separators=(",", ":")) + "\n"
                    "Current server rules and recovery facts, which also take precedence over retained notes: "
                    + json.dumps({"server_rules": server_info["result"].get("server_rules", ""),
                                  "recovery": gateway.gameplay.recovery_guidance()}, ensure_ascii=False, separators=(",", ":")) + "\n"
                    "Re-read inbox/leaf-specification.json before declaring completion. "
                    "Read the fresh initial observation in scratch/actor-journal.jsonl. Use the Minecraft MCP tools. "
                    "Your final JSON must include exactly status, summary, and handoff. "
                    "The required handoff string can name your written outbox/handoff.md.",
                    work_dir=context.workspace, runtime=agent_runtime, mcp_config=config,
                    trace_file=context.kernel_data / "actor-agent-trace.jsonl")
            runtime.check_cancelled()
            outcome = ActorOutcome.model_validate(result.result_json)
            final_instruction = runtime.tools["managed_instructions"](close=True)
            if final_instruction["revision"] != gateway.revision:
                outcome.status = "yield"
                outcome.summary = "Instructions changed before completion. Resume from the retained evidence."
            gateway.revision = final_instruction["revision"]
            request.goal = final_instruction["instruction"] or request.goal
            final = gateway.call("minecraft_observe", {"include_image": request.include_image}, close=True)
            if outcome.status == "complete" and (gateway.gameplay.recovery or gateway.gameplay.current_failed_recovery()):
                outcome.status = "replan"
                outcome.summary = "Recovery is active or failed through death. Add or revise a concrete recovery leaf before optional work. " + outcome.summary
            verification = {"verified": False, "reason": "Independent generic leaf verification belongs to A11."}
            if request.acceptance.get("kind") == "wood_return" and final["ok"]:
                verification = verify_wood_return(gateway.receipts, baseline, final["result"], request.acceptance)
            accepted = outcome.status == "complete" and verification["verified"]
            image_paths = [frame["path"] for row in gateway.receipts for frame in row["images"]]
            world = build_world(request, final, None, image_paths)
            world["outcome"] = world["episodes"][0]["outcome"] = "success" if accepted else outcome.status
            world["episodes"][0]["action_count"] = sum(row["tool"] not in READ_TOOLS for row in gateway.receipts)
            gameplay = gateway.gameplay.snapshot()
            world["episodes"][0]["gameplay"] = gameplay
            report = {"schema": "pm.minecraft-actor-report.v1", "accepted": accepted,
                      "goal": final_instruction["instruction"] or request.goal or request.instruction,
                      "body_id": request.body_id, "summary": outcome.summary,
                      "outcome": outcome.model_dump(), "verification": verification,
                      "acceptance": request.acceptance, "verified": accepted,
                      "task_reference": request.task_reference,
                      "agent_session": result.session_ref, "instruction_revision": gateway.revision,
                      "call_count": len(gateway.receipts), "visual_interpretation": False,
                      "initial_observation": baseline, "final_observation": final["result"],
                      "execution_epoch": gateway.token["epoch"], "reconciliation": reconciliation,
                      "gameplay": gameplay,
                      "journal": "scratch/actor-journal.jsonl", "images": image_paths,
                      "rejection_reason": "" if accepted else "The leaf completion was not independently verified."}
            runtime.check_cancelled()
            # The native outcome may reference its already written handoff.
            # Preserve that document instead of replacing it with its path.
            handoff_path = environment._path("outbox/handoff.md")
            if not handoff_path.is_file() or not handoff_path.read_text(encoding="utf-8").strip():
                environment.write_text("outbox/handoff.md", outcome.summary + "\n\n" + outcome.handoff + "\n")
            environment.write_text("outbox/world.yaml", yaml.safe_dump(world, allow_unicode=True, sort_keys=False))
            environment.write_text("outbox/report.yaml", yaml.safe_dump(report, allow_unicode=True, sort_keys=False))
            return {"status": "done", "summary": report["summary"]}
        finally:
            gateway.wait_idle()
