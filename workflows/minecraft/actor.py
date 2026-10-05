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
                trees[tuple(base[key] for key in ("x", "y", "z"))] = data["logs_collected"]
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
            "tree_bases": [list(base) for base in trees], "initial_logs": logs(initial),
            "final_logs": logs(final), "maximum_travel": traveled, "return_distance": return_distance}


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
    with runtime.tools["minecraft_lease"]():
        try:
            initial = gateway.call("minecraft_observe", {"include_image": request.include_image})
            if not initial["ok"]:
                raise RuntimeError("The actor has no usable initial observation.")
            schemas = runtime.tools["minecraft_tool_schemas"]()
            allowed = set(request.allowed_tools) if request.allowed_tools else {row["name"] for row in schemas}
            # Scripts can hide multiple actions from the instruction boundary.
            # Operator resets are not part of the configured survival action space.
            allowed -= {"minecraft_execute_typescript", "minecraft_suicide"}
            schemas = [row for row in schemas if row["name"] in allowed]
            environment.write_text("inbox/tool-schemas.json", json.dumps(schemas, ensure_ascii=False, indent=2))
            environment.write_text("inbox/persona.yaml", yaml.safe_dump(request.persona_context, allow_unicode=True))
            resources = [RESOURCES / "shared.md", RESOURCES / "navigator.md", RESOURCES / f"{request.specialist}.md"]
            skill = "\n\n".join(path.read_text(encoding="utf-8") for path in dict.fromkeys(resources))
            skill += ("\n\nYou are a private specialist of the same companion. Do not address the user. "
                      "Read inbox/persona.yaml and inbox/request.yaml. Tools use the live MCP schemas. "
                      "This configured actor uses text evidence. It cannot interpret screenshot pixels. "
                      "Images are retained as separate artifacts. After an instruction_changed result, "
                      "discard the pending plan and obey the new instruction. Call actor_instructions at milestones. "
                      "Write a concise task card to scratch/task-card.md and a durable handoff to outbox/handoff.md. "
                      "Continue until the requested leaf succeeds, yields, needs replanning, or meets a real blocker. "
                      "Return only JSON with status (complete, yield, blocked, replan), summary, and handoff. "
                      "A wood-return check requires distinct harvest_tree bases with positive logs, travel, and return. "
                      "Do not finish after merely locating trees. Save starting coordinates for the return.")
            skill_path = context.kernel_data / "actor-skill.md"
            skill_path.write_text(skill, encoding="utf-8")
            if runtime.classification_session is None and runtime.agent_model is None:
                raise ValueError("A Minecraft actor needs a configured model session.")
            from contextlib import nullcontext
            session = (nullcontext(runtime.agent_model) if runtime.agent_model is not None else
                       runtime.classification_session("Default", False, runtime.stop_event))
            with session as model, serve_services({"minecraft_actor": gateway.server(schemas)}) as servers:
                config = context.kernel_data / "actor-mcp.json"
                config.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
                agent_runtime = replace(runtime, agent_model=model, mcp_servers=servers)
                driver = PmCoderDriver(log_root=context.kernel_data / "actor-agent")
                result = driver.run_session(run_id=context.task_id + "-actor", attempt=1,
                    skill=str(skill_path), prompt="Execute the Minecraft leaf in inbox/request.yaml. "
                    "Read the fresh initial observation in scratch/actor-journal.jsonl. Use the Minecraft MCP tools.",
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
            verification = {"verified": False, "reason": "Independent generic leaf verification belongs to A11."}
            if request.acceptance.get("kind") == "wood_return" and final["ok"]:
                verification = verify_wood_return(gateway.receipts, initial["result"], final["result"], request.acceptance)
            accepted = outcome.status == "complete" and verification["verified"]
            image_paths = [frame["path"] for row in gateway.receipts for frame in row["images"]]
            world = build_world(request, final, None, image_paths)
            world["outcome"] = world["episodes"][0]["outcome"] = "success" if accepted else outcome.status
            world["episodes"][0]["action_count"] = sum(row["tool"] not in READ_TOOLS for row in gateway.receipts)
            report = {"schema": "pm.minecraft-actor-report.v1", "accepted": accepted,
                      "goal": final_instruction["instruction"] or request.goal or request.instruction,
                      "body_id": request.body_id, "summary": outcome.summary,
                      "outcome": outcome.model_dump(), "verification": verification,
                      "acceptance": request.acceptance, "verified": accepted,
                      "agent_session": result.session_ref, "instruction_revision": gateway.revision,
                      "call_count": len(gateway.receipts), "visual_interpretation": False,
                      "initial_observation": initial["result"], "final_observation": final["result"],
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
