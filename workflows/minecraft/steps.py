"""Minecraft body role.

Entry point: ``run(context)``. The default runs a sustained native pm-coder
actor. The explicit boundary-check mode retains the earlier one-action path.
Both paths hold the body lease and write observations, screenshots, world
evidence, and a final report.

Screenshot pixels are written as files, never into a prompt. The tool result
base64 is the source of that file.
"""
from __future__ import annotations
from workflows.minecraft.runtime import require_minecraft_runtime
from workflows.minecraft_body.world_text import observation_text
from pm.runs.managed_boundary import BoundaryReadSettings

import asyncio
import base64
import json
from pathlib import Path
from typing import Any

from pm_workflows.python_role import RoleContext
from pm_workflows.working_target import WorkingEnvironment

from pm.utils import yaml_util

from workflows.minecraft.schemas import MinecraftReport, MinecraftRequest

WORKFLOW_DIR = Path(__file__).resolve().parent


def load_prompts(path: Path | None = None) -> dict:
    return yaml_util.safe_load((path or WORKFLOW_DIR / "prompts.yaml").read_text(encoding="utf-8"))


def summarize(call_result: dict) -> str:
    """One bounded, human-readable line from one tool result."""
    data = call_result.get("result") or {}
    if not isinstance(data, dict):
        data = {}
    parts = [f"tool={call_result.get('tool')} ok={call_result.get('ok')}"]
    if not call_result.get("ok"):
        parts.append(f"reason={data.get('reason')} message={data.get('message')}")
    camera = data.get("camera") or {}
    feet = camera.get("feet_position") or {}
    parts.append(
        "position=({x},{y},{z})".format(
            x=feet.get("x"), y=feet.get("y"), z=feet.get("z"),
        )
    )
    player = data.get("player") or {}
    parts.append(
        f"biome={player.get('biome')} dimension={player.get('dimension')} "
        f"health={player.get('health')} food={player.get('food')}"
    )
    blocks = data.get("nearby_blocks") or []
    summary = ", ".join(
        f"{block.get('block_name')}x{block.get('count')}"
        for block in blocks[:8] if isinstance(block, dict)
    )
    parts.append("nearby_blocks=" + (summary or "(none)"))
    return " | ".join(parts)


def parse_plan(text: str) -> tuple[str, dict]:
    """Parse ``<tool> <json>`` from a planner reply. Malformed raises."""
    text = (text or "").strip()
    if not text:
        raise ValueError("The planner returned no action.")
    head, _, tail = text.partition(" ")
    if not tail.strip():
        return head.strip(), {}
    arguments = json.loads(tail.strip())
    if not isinstance(arguments, dict):
        raise ValueError("The planner arguments must be a JSON object.")
    return head.strip(), arguments


def write_images(environment: WorkingEnvironment, images: list[dict[str, Any]]) -> list[str]:
    """Write screenshot pixels to real files and return their paths.

    An environment without binary writes keeps the base64 payload in a text
    file beside the screenshots. A frame that already names a path is retained
    as it is, so a replayed call cannot duplicate the bytes.
    """
    paths: list[str] = []
    for index, image in enumerate(images):
        if image.get("path"):
            paths.append(image["path"])
            continue
        payload = image.get("base64")
        if not payload:
            continue
        media_type = str(image.get("media_type", "image/png"))
        suffix = ".png" if "png" in media_type else ".jpg"
        raw = base64.b64decode(payload)
        path = f"screenshots/frame-{index:02d}{suffix}"
        if hasattr(environment, "write_binary"):
            environment.write_binary(path, raw)
            paths.append(path)
        else:
            encoded = f"screenshots/frame-{index:02d}{suffix}.b64"
            environment.write_text(encoded, payload)
            paths.append(encoded)
    return paths


def build_world(
    request: MinecraftRequest,
    observation: dict,
    action: dict | None,
    image_paths: list[str],
) -> dict:
    """One structured spatial record from the real tool results."""
    data = observation.get("result") or {}
    if not isinstance(data, dict):
        data = {}
    camera = data.get("camera") or {}
    feet = camera.get("feet_position") or {}
    player = data.get("player") or {}
    dimension = str(player.get("dimension", "unknown"))
    blocks = [block for block in (data.get("nearby_blocks") or []) if isinstance(block, dict)]
    outcome = "observed"
    action_count = 0
    if action is not None:
        action_count = 1
        outcome = "success" if action.get("ok") else "failed"
    return {
        "body_id": request.body_id,
        "goal": request.goal,
        "outcome": outcome,
        "places": [{
            "body_id": request.body_id,
            "place_key": "current",
            "content": f"Position near {player.get('biome', 'unknown')} biome",
            "position": feet,
            "dimension": dimension,
            "landmarks": [str(player.get("biome", "unknown"))],
            "confidence": 0.9,
        }],
        "objects": [{
            "body_id": request.body_id,
            "object_key": str(block.get("block_name", "")),
            "object_kind": "block",
            "content": f"Nearby {block.get('display_name', block.get('block_name'))}",
            "position": block.get("nearest") or {},
            "dimension": dimension,
            "confidence": 0.7,
        } for block in blocks[:8]],
        "episodes": [{
            "body_id": request.body_id,
            "goal": request.goal,
            "outcome": outcome,
            "action_count": action_count,
            "images": list(image_paths),
        }],
    }


def _run_boundary_check(context: RoleContext) -> dict[str, Any]:
    """Run the narrow explicit observation/action boundary check.

    The observation is published before the optional instruction wait, so a
    cancellation still retains real world evidence. A new instruction revision
    invalidates a previously supplied action.
    """
    environment = context.runtime.environment
    payload = yaml_util.safe_load(environment.read_text("inbox/request.yaml"))
    request = MinecraftRequest.model_validate({key: value for key, value in payload.items()
                                             if not key.startswith("_public_")})
    prompts = load_prompts()
    from workflows.minecraft_body.gateway import BodyGateway
    runtime = require_minecraft_runtime(context.runtime)
    endpoint = runtime.minecraft_transport.settings.endpoint
    gateway = BodyGateway(endpoint, runtime.body_lease.owner, environment, runtime)
    initial = runtime.instructions()
    request.goal = initial["instruction"] or request.goal or request.instruction
    context.runtime.check_cancelled()
    observation = gateway.call("minecraft_observe", {"include_image": request.include_image})
    observations = [summarize(observation)]
    images = list(observation.get("images") or [])

    # Publish the observation before the optional instruction wait. Cancellation
    # retains this evidence even when no action has started.
    image_paths = write_images(environment, images)
    environment.write_text("outbox/world.yaml", yaml_util.safe_dump(
        build_world(request, observation, None, image_paths), sort_keys=False,
    ))
    environment.write_text("scratch/observations.md", observations[0] + "\n")
    revision = runtime.instructions(BoundaryReadSettings(wait=request.wait_for_update, close=True))
    gateway.revision = revision["revision"]
    request.goal = revision["instruction"] or request.goal
    context.runtime.check_cancelled()

    action_tool = request.action_tool
    action_arguments = dict(request.action_arguments)
    if revision["revision"] != initial["revision"]:
        # A new instruction invalidates a previously supplied action.
        action_tool, action_arguments = "", {}
    action = None
    final_observation = observation
    if observation["ok"]:
        if not action_tool and request.goal:
            schemas = runtime.minecraft_transport.tool_schemas()
            allowed = set(request.allowed_tools) if request.allowed_tools else {schema.name for schema in schemas}
            schemas = [schema for schema in schemas if schema.name in allowed]
            plan = runtime.minecraft_transport.planner.choose_action(
                observation_text(observation["result"]), tuple(schemas), request.goal)
            action_tool, action_arguments = parse_plan(plan)
            if action_tool not in {schema.name for schema in schemas}:
                raise ValueError(f"The Minecraft planner selected an unavailable action: {action_tool}")
        if action_tool:
            if request.allowed_tools and action_tool not in request.allowed_tools:
                raise ValueError(f"The Minecraft action is not granted: {action_tool}")
            context.runtime.check_cancelled()
            action = gateway.call(action_tool, action_arguments)
            observations.append(summarize(action))
            images.extend(action.get("images") or [])
            context.runtime.check_cancelled()
            final_observation = gateway.call("minecraft_observe", {"include_image": request.include_image})
            observations.append(summarize(final_observation))
            images.extend(final_observation.get("images") or [])

    image_paths = write_images(environment, images)
    world = build_world(request, final_observation, action, image_paths)
    accepted = observation["ok"] and final_observation["ok"] and (action is None or action["ok"])
    if not accepted:
        world["outcome"] = world["episodes"][0]["outcome"] = "failed"
    environment.write_text(
        "scratch/observations.md",
        f"# {prompts['observation_header'].format(body_id=request.body_id)}\n\n"
        f"Goal: {request.goal}\nRevision: {revision['revision']}\n\n"
        + "\n".join(f"- {line}" for line in observations) + "\n",
    )
    environment.write_text("outbox/world.yaml", yaml_util.safe_dump(world, sort_keys=False, allow_unicode=True))
    # Image bytes belong only in image files and evidence metadata.
    report_action = {key: value for key, value in action.items() if key != "images"} if action else None
    report = MinecraftReport(
        goal=request.goal, body_id=request.body_id, observations=observations,
        action=report_action, images=[{"path": path} for path in image_paths],
        world=world, accepted=accepted,
        rejection_reason="" if accepted else "the Minecraft tool reported a failure",
    )
    environment.write_text("scratch/report.yaml", yaml_util.safe_dump(
        report.model_dump(mode="json"), sort_keys=False, allow_unicode=True,
    ))
    environment.move("scratch/report.yaml", "outbox/report.yaml")
    if request.wait_for_cancel:
        assert runtime.managed_boundary is not None
        runtime.managed_boundary.wait_for_stop()
        runtime.check_cancelled()
    return {"status": "done", "summary": "minecraft body finished with recorded tool evidence"}


def run(context: RoleContext) -> dict[str, Any]:
    """Select sustained work or the retained explicit W8 boundary check."""
    payload = yaml_util.safe_load(context.runtime.environment.read_text("inbox/request.yaml"))
    # Existing explicit actions and wait fixtures retain their narrow contract.
    narrow = payload.get("mode") == "boundary_check" or payload.get("action_tool") or payload.get("wait_for_cancel") or payload.get("wait_for_update")
    if narrow:
        runtime = require_minecraft_runtime(context.runtime)
        with runtime.body_lease.acquire():
            try:
                return _run_boundary_check(context)
            finally:
                if runtime.body_lease.ownership is None:
                    token = runtime.minecraft_transport.leases.token(runtime.minecraft_transport.settings.endpoint, runtime.body_lease.owner)
                    runtime.minecraft_transport.release(token)
    from workflows.minecraft.specialist_session import run_specialist_session
    return run_specialist_session(context)


__all__ = [
    "build_world",
    "load_prompts",
    "parse_plan",
    "run",
    "summarize",
    "write_images",
]
