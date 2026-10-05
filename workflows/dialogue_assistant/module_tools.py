"""Module-level tool factory for Assistant dialogue.

The application supplies one lazily-activated model, one managed-run
supervisor, and the scoped Ghost tools. The same management functions are
registered on the scoped MCP server supplied by ``mcp_services_factory``, so
the role can call them over MCP or directly with one implementation.
"""
from __future__ import annotations

from typing import Callable

from ghost_mcp import build_mcp, build_tools
from runtime.management_tools import build_management_tools, management_mcp


def build_assistant_tools(model, supervisor) -> Callable[[object, object], dict]:
    """Return ``tools_factory(environment, batch)`` for Assistant dialogue."""

    def factory(environment, batch) -> dict:
        tools = {"assistant_complete": model.build_tool_completion()}
        tools["public_complete"] = model.build_async_tool()
        tools.update(build_management_tools(supervisor))
        tools.update(build_tools(environment, batch))
        return tools

    return factory


def build_assistant_mcp_factory(supervisor) -> Callable[[object], dict]:
    """Return ``mcp_services_factory(environment)`` for one Assistant run."""

    def factory(environment) -> dict:
        return {
            "workflow_management": management_mcp(supervisor),
            "ghost": build_mcp(environment),
        }

    return factory


__all__ = ["build_assistant_mcp_factory", "build_assistant_tools"]
