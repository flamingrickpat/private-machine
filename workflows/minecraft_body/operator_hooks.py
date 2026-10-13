"""Operator-console hooks of the Minecraft body: endpoint defaults and health probes.

The console loads this module by name, so the runtime holds no Minecraft
endpoint knowledge.
"""
from __future__ import annotations

import asyncio
import os
import socket


def prepare_environment(environment: dict) -> None:
    """Export the MCP endpoint before the assembly builds its tools."""
    os.environ.setdefault("MINECRAFT_MCP_URL", environment.get("MINECRAFT_MCP_URL") or
                          f"http://127.0.0.1:{environment.get('PM_MC_MCP_PORT', '6767')}/mcp")


async def probes(values: dict) -> list[dict]:
    """Probe MCP tool discovery and the Minecraft server TCP port."""
    rows = []
    mcp = values.get("MINECRAFT_MCP_URL") or f"http://127.0.0.1:{values.get('PM_MC_MCP_PORT', '6767')}/mcp"
    try:
        from fastmcp import Client
        async def discover():
            async with Client(mcp) as client:
                return [t.name for t in await client.list_tools()]
        tools = await asyncio.wait_for(discover(), timeout=5)
        rows.append({"name": "Minecraft MCP", "endpoint": mcp, "state": "ready", "tools": tools})
    except Exception as exc:
        rows.append({"name": "Minecraft MCP", "endpoint": mcp, "state": "not ready", "error_type": type(exc).__name__})
    host, port = values.get("PM_MINECRAFT_HOST", "127.0.0.1"), int(values.get("PM_MINECRAFT_PORT", "12345"))
    def tcp_probe():
        try:
            with socket.create_connection((host, port), timeout=3):
                return "TCP reachable"
        except OSError:
            return "unreachable"
    rows.append({"name": "Minecraft server", "endpoint": f"{host}:{port}", "state": await asyncio.to_thread(tcp_probe)})
    return rows
