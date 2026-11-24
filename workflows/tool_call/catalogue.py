"""The tool list of the configured MCP servers, in words for the route prompt.

Only the route phase reads this text: the decide phase stays in character and
never sees tool names (decision D25). Discovery lists the tools of each
server with a short deadline (``TOOL_DISCOVERY_SECONDS``), as the Minecraft
capability discovery does. A successful listing is cached until the next refresh. A server that does not answer is shown as unavailable now and is
asked again after ``TOOL_DISCOVERY_RETRY_SECONDS``. Discovery calls no tool.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import threading
import time

import httpx
from mcp.shared.exceptions import McpError
from pydantic import BaseModel, ConfigDict

from pm.constants import TOOL_DISCOVERY_RETRY_SECONDS, TOOL_DISCOVERY_SECONDS
from pm.llm.mcp_config import McpServerConfig, ToolCallConfig


class ToolEntry(BaseModel):
    """One tool of one server: its name and the first line of its description."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""


def _first_line(text: str | None) -> str:
    """Return the first non-empty line of a tool description."""
    return next((line.strip() for line in (text or "").splitlines() if line.strip()), "")


async def _list_tools(server: McpServerConfig) -> list[ToolEntry]:
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    transport = StreamableHttpTransport(server.url, headers=server.resolved_headers())
    async with Client(transport) as client:
        return [ToolEntry(name=item.name, description=_first_line(item.description))
                for item in await client.list_tools() if server.name != "minecraft" or
                (not item.name.startswith("minecraft_body_") and item.name not in
                 {"minecraft_disconnect", "minecraft_connect", "minecraft_execute_typescript", "minecraft_suicide"})]


@dataclass(frozen=True, slots=True)
class ToolDiscoveryFailure:
    error_type: str
    observed_at: float


class ToolCatalogue:
    """Discover and describe the tools of every configured MCP server.

    The object is shared by the route calls of one application. A lock keeps
    the cache consistent when decisions route in parallel.
    """

    def __init__(self, config: ToolCallConfig) -> None:
        """Retain the MCP configuration and the clock used for retry deadlines."""
        self.config = config
        self._tools: dict[str, list[ToolEntry]] = {}
        #: Server name -> (error type, clock time of the failed listing).
        self._failures: dict[str, ToolDiscoveryFailure] = {}
        self._checked: dict[str, float] = {}
        self._lock = threading.Lock()

    def tools(self, server: McpServerConfig) -> list[ToolEntry] | str:
        """Return the tools of one server, or the error type when the server does not answer now."""
        with self._lock:
            if server.name in self._tools and time.monotonic() - self._checked[server.name] < TOOL_DISCOVERY_RETRY_SECONDS:
                return self._tools[server.name]
            if (server.name in self._failures
                    and time.monotonic() - self._failures[server.name].observed_at < TOOL_DISCOVERY_RETRY_SECONDS):
                return self._failures[server.name].error_type
        try:
            found = asyncio.run(asyncio.wait_for(_list_tools(server), timeout=TOOL_DISCOVERY_SECONDS))
        except (OSError, TimeoutError, httpx.HTTPError, McpError, RuntimeError) as error:
            # An unreachable server is an external state, not our bug. The
            # route then knows that this server cannot help now.
            with self._lock:
                self._failures[server.name] = ToolDiscoveryFailure(type(error).__name__, time.monotonic())
            return type(error).__name__
        with self._lock:
            self._checked[server.name] = time.monotonic()
            self._tools[server.name] = found
        return found

    def describe(self, *, compact: bool = False) -> str:
        """Return the servers and their tools as indented lines for the workflow list."""
        lines: list[str] = []
        for server in self.config.mcp_servers:
            found = self.tools(server)
            if isinstance(found, str):
                lines.append(f"  - server {server.name}: unavailable now ({found})")
                continue
            lines.append(f"  - server {server.name}:")
            if compact:
                # The agent receives all schemas when it enters the session. The chooser needs orientation.
                entries: list[str] = []
                remaining = 800
                for tool in found:
                    label = tool.name.removeprefix(server.name + "_")
                    if len(found) <= 8 and tool.description:
                        label += " (" + tool.description[:64] + ")"
                    if len(label) + 2 > remaining:
                        break
                    entries.append(label)
                    remaining -= len(label) + 2
                omitted = len(found) - len(entries)
                if omitted:
                    entries.append(f"{omitted} more in session")
                lines.append(f"    {len(found)} tools: " + ", ".join(entries))
            else:
                for tool in found:
                    lines.append(f"    - {tool.name}: {tool.description[:160]}")

        return "\n".join(lines)


__all__ = ["ToolCatalogue", "ToolEntry"]
