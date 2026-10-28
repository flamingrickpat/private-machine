"""Read current domain capabilities without moving the Minecraft character."""
from __future__ import annotations

from pm import vclock

import asyncio
import os
import socket
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import anyio
import httpx
from mcp.shared.exceptions import McpError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pm.memory.store import MemoryStore


def _transport_failure(error: BaseException) -> bool:
    """Recognize client transport failures without hiding programming errors."""
    if isinstance(error, BaseExceptionGroup):
        return all(_transport_failure(item) for item in error.exceptions)
    if isinstance(error, (OSError, TimeoutError, httpx.HTTPError, McpError,
                          anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream)):
        return True
    # FastMCP wraps closed sessions and initialization timeouts in RuntimeError.
    return isinstance(error, RuntimeError) and error.__cause__ is not None and _transport_failure(error.__cause__)


def capability_snapshot(ghost: MemoryStore, workflow_names: Iterable[str]) -> dict[str, Any]:
    """Discover MCP tools and game transport. No world action runs here.

    Discovery has a five-second deadline. A connection proves availability,
    not perception, successful gameplay, or visual interpretation.
    """
    from pm.app.operator_config import environment_values
    from pm.records.mental_state import StateEmotions
    values, _ = environment_values()
    url = os.environ.get("MINECRAFT_MCP_URL") or values.get("MINECRAFT_MCP_URL") or "http://127.0.0.1:6767/mcp"
    result = {"status": "disconnected", "endpoint": url, "tools": [],
              "visual_interpretation": False, "observed_at": vclock.now_aware().isoformat()}

    async def discover() -> list[str]:
        """Return the tool names that the body endpoint advertises."""
        from fastmcp import Client
        async with Client(url) as client:
            return [item.name for item in await client.list_tools()]

    try:
        result["tools"] = asyncio.run(asyncio.wait_for(discover(), timeout=5))
        with socket.create_connection((values.get("PM_MINECRAFT_HOST", "127.0.0.1"),
                                       int(values.get("PM_MINECRAFT_PORT", "12345"))), timeout=2):
            pass
        result["status"] = "connected"
    except Exception as error:
        if not _transport_failure(error):
            raise
        result["error_type"] = type(error).__name__
    config = ghost.ghost_config
    return {"text": True, "persistent_memory": True, "human_senses": [], "human_body": False,
            "configured": list(config.supported_capabilities),
            "unavailable": list(map(str, config.unsupported_capabilities)),
            "notes": list(map(str, config.capability_notes)),
            "managed_workflows": sorted(workflow_names), "minecraft": result,
            "implemented_emotions": list(StateEmotions.model_fields)}


__all__ = ["capability_snapshot"]
