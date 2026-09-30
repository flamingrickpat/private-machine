"""Minecraft body role.

Entry point: ``run(context)``. The role observes the real world through the
injected ``minecraft_call`` tool, chooses one bounded action (explicitly or
through the optional ``minecraft_plan`` model tool), executes it, and writes
real evidence: text observations, a structured ``outbox/world.yaml``, the
screenshot files, and a final report.

Screenshot pixels are written as files, never into a prompt. The tool result
base64 is the source of that file.
"""
from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import yaml

from workflows.minecraft.schemas import MinecraftReport, MinecraftRequest

WORKFLOW_DIR = Path(__file__).resolve().parent


def load_prompts(path: Path | None = None) -> dict:
    return yaml.safe_load((path or WORKFLOW_DIR / "prompts.yaml").read_text(encoding="utf-8"))


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


def write_images(environment, images: list[dict]) -> list[str]:
    """Write screenshot pixels to real files and return their paths."""
    paths: list[str] = []
    for index, image in enumerate(images):
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


def run(context) -> dict:
    environment = context.runtime.environment
    request = MinecraftRequest.model_validate(
        yaml.safe_load(environment.read_text("inbox/request.yaml"))
    )
    prompts = load_prompts()
    call = context.runtime.tools["minecraft_call"]

    observation = call("minecraft_observe", {"include_image": request.include_image})
    observations = [summarize(observation)]

    action_tool = request.action_tool
    action_arguments = dict(request.action_arguments)
    if not action_tool and request.allowed_tools and "minecraft_plan" in context.runtime.tools:
        plan = context.runtime.tools["minecraft_plan"](
            observations[-1], request.allowed_tools, request.goal,
        )
        action_tool, action_arguments = parse_plan(plan)

    action = None
    if action_tool:
        action = call(action_tool, action_arguments)
        observations.append(summarize(action))

    images = list(observation.get("images") or [])
    if action is not None:
        images.extend(action.get("images") or [])
    image_paths = write_images(environment, images)

    world = build_world(request, observation, action, image_paths)
    environment.write_text(
        "scratch/observations.md",
        f"# {prompts['observation_header'].format(body_id=request.body_id)}\n\n"
        + "\n".join(f"- {line}" for line in observations) + "\n",
    )
    environment.write_text(
        "outbox/world.yaml",
        yaml.safe_dump(world, sort_keys=False, allow_unicode=True),
    )

    if request.wait_for_cancel:
        # Hold the run at a declared safe boundary so a caller can update or
        # terminate it. The evidence above is already durable on disk.
        context.runtime.stop_event.wait()
        context.runtime.check_cancelled()

    accepted = bool(observation.get("ok")) and (action is None or bool(action.get("ok")))
    report = MinecraftReport(
        goal=request.goal,
        body_id=request.body_id,
        observations=observations,
        action=action,
        images=[{"path": path} for path in image_paths],
        world=world,
        accepted=accepted,
        rejection_reason="" if accepted else "the Minecraft tool reported a failure",
    )
    environment.write_text(
        "scratch/report.yaml",
        yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
    )
    environment.move("scratch/report.yaml", "outbox/report.yaml")
    return {
        "status": "done",
        "summary": f"minecraft body observed and {'acted' if action else 'waited'}",
    }


__all__ = [
    "build_world",
    "load_prompts",
    "parse_plan",
    "run",
    "summarize",
    "write_images",
]
