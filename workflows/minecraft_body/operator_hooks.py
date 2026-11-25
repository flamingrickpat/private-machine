"""Probe the configured Minecraft body for the operator console."""
from __future__ import annotations

import asyncio
import time

from pm.app.operator_config import ConnectionRow
from pm.llm.config import ModelConfig
from workflows.minecraft_body.mcp_client import probe_body


async def probes(configuration: ModelConfig) -> list[ConnectionRow]:
    """Ask the enabled Minecraft MCP endpoint for its tools."""
    body = configuration.minecraft
    if body is None:
        return []
    started = time.perf_counter()
    try:
        await asyncio.to_thread(probe_body, body.mcp_url, timeout_seconds=body.probe_timeout_seconds)
        state = "reachable"
    except Exception as error:
        state = type(error).__name__
    return [ConnectionRow(name="Minecraft MCP", endpoint=body.mcp_url, model="",
                          state=state, seconds=round(time.perf_counter() - started, 3))]
