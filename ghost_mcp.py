"""Scoped Ghost tools, exposed both as plain functions and as an MCP server.

The tools read the invocation's visibility snapshot from the working
environment. They never reach back into the live Ghost, so an omitted record
stays omitted for both Bash and MCP access.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from fastmcp import FastMCP


def _visible_knoxels(environment) -> dict[int, dict]:
    return getattr(environment, "visible_knoxels", {})


def _visible_gwt(environment) -> dict[int, dict]:
    return getattr(environment, "visible_gwt", {})


def build_tools(environment) -> Mapping[str, Callable[..., Any]]:
    """Return the three bound tool functions for one invocation."""

    def ghost_get(knoxel_id: int) -> dict:
        record = _visible_knoxels(environment).get(knoxel_id)
        if record is None:
            raise KeyError(knoxel_id)
        return record

    def ghost_search(query: str) -> list[dict]:
        needle = query.casefold()
        matches: list[dict] = []
        for knoxel_id in sorted(_visible_knoxels(environment)):
            record = _visible_knoxels(environment)[knoxel_id]
            content = str(record.get("data", {}).get("content", ""))
            if needle in content.casefold():
                matches.append(record)
        return matches

    def ghost_gwt(tick_id: int) -> dict:
        record = _visible_gwt(environment).get(tick_id)
        if record is None:
            raise KeyError(tick_id)
        return record

    return {
        "ghost_get": ghost_get,
        "ghost_search": ghost_search,
        "ghost_gwt": ghost_gwt,
    }


def build_mcp(environment) -> FastMCP:
    """Register the same three scoped tools on a FastMCP server."""
    tools = build_tools(environment)
    server = FastMCP("ghost")
    for fn in tools.values():
        server.add_tool(fn)
    return server


__all__ = ["build_tools", "build_mcp"]