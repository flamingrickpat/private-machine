"""Read current domain capabilities without moving the Minecraft character."""
from __future__ import annotations

import asyncio
import os
import socket
from datetime import datetime, timezone


def capability_snapshot(ghost, workflow_names) -> dict:
    """Discover MCP tools and game transport. No world action runs here.

    Discovery has a five-second deadline. A connection proves availability,
    not perception, successful gameplay, or visual interpretation.
    """
    from runtime.operator_services import environment_values
    from pm.records.mental_state import StateEmotions
    values, _ = environment_values()
    url = os.environ.get("MINECRAFT_MCP_URL") or values.get("MINECRAFT_MCP_URL") or "http://127.0.0.1:6767/mcp"
    result = {"status": "disconnected", "endpoint": url, "tools": [],
              "visual_interpretation": False, "observed_at": datetime.now(timezone.utc).isoformat()}

    async def discover():
        from fastmcp import Client
        async with Client(url) as client:
            return [item.name for item in await client.list_tools()]

    try:
        result["tools"] = asyncio.run(asyncio.wait_for(discover(), timeout=5))
        with socket.create_connection((values.get("PM_MINECRAFT_HOST", "127.0.0.1"),
                                       int(values.get("PM_MINECRAFT_PORT", "12345"))), timeout=2):
            pass
        result["status"] = "connected"
    except (OSError, TimeoutError, ExceptionGroup) as error:
        result["error_type"] = type(error).__name__
    except Exception as error:
        # MCP exposes transport faults through several client exception types.
        result["error_type"] = type(error).__name__
    config = ghost.ghost_config
    return {"text": True, "persistent_memory": True, "human_senses": [], "human_body": False,
            "configured": list(config.supported_capabilities),
            "unavailable": list(map(str, config.unsupported_capabilities)),
            "notes": list(map(str, config.capability_notes)),
            "managed_workflows": sorted(workflow_names), "minecraft": result,
            "implemented_emotions": list(StateEmotions.model_fields)}


__all__ = ["capability_snapshot"]
