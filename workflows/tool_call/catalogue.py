"""The tool list of the configured MCP servers, in words for the route prompt.

Only the route phase reads this text: the decide phase stays in character and
never sees tool names (decision D25). Discovery lists the tools of each
server with a short deadline (``TOOL_DISCOVERY_SECONDS``), as the Minecraft
capability discovery does. A successful listing is cached for the life of the
application. A server that does not answer is shown as unavailable now and is
asked again after ``TOOL_DISCOVERY_RETRY_SECONDS``. Discovery calls no tool.
"""
from __future__ import annotations

import asyncio
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
                for item in await client.list_tools()]


class ToolCatalogue:
    """Discover and describe the tools of every configured MCP server.

    The object is shared by the route calls of one application. A lock keeps
    the cache consistent when decisions route in parallel.
    """

    def __init__(self, config: ToolCallConfig, clock=time.monotonic) -> None:
        self.config = config
        self.clock = clock
        self._tools: dict[str, list[ToolEntry]] = {}
        #: Server name -> (error type, clock time of the failed listing).
        self._failures: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def tools(self, server: McpServerConfig) -> list[ToolEntry] | str:
        """Return the tools of one server, or the error type when the server does not answer now."""
        with self._lock:
            if server.name in self._tools:
                return self._tools[server.name]
            if (server.name in self._failures
                    and self.clock() - self._failures[server.name][1] < TOOL_DISCOVERY_RETRY_SECONDS):
                return self._failures[server.name][0]
        try:
            found = asyncio.run(asyncio.wait_for(_list_tools(server), timeout=TOOL_DISCOVERY_SECONDS))
        except (OSError, TimeoutError, httpx.HTTPError, McpError, RuntimeError) as error:
            # An unreachable server is an external state, not our bug. The
            # route then knows that this server cannot help now.
            with self._lock:
                self._failures[server.name] = (type(error).__name__, self.clock())
            return type(error).__name__
        with self._lock:
            self._tools[server.name] = found
        return found

    def describe(self) -> str:
        """Return the servers and their tools as indented lines for the workflow list."""
        lines = []
        for server in self.config.mcp_servers:
            found = self.tools(server)
            if isinstance(found, str):
                lines.append(f"  - server {server.name}: unavailable now ({found})")
                continue
            lines.append(f"  - server {server.name}:")
            lines += [f"    - {tool.name}: {tool.description}" if tool.description else f"    - {tool.name}"
                      for tool in found]
        return "\n".join(lines)


__all__ = ["ToolCatalogue", "ToolEntry"]
