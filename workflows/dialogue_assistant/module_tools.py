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
        from runtime.minecraft_tools import build_minecraft_tools, DEFAULT_MCP_URL
        from runtime.body_lease import BODY_LEASES
        import os
        endpoint = os.environ.get("MINECRAFT_MCP_URL", DEFAULT_MCP_URL)
        minecraft = build_minecraft_tools(endpoint)
        def minecraft_observe(include_image: bool = False):
            """Read current numeric state. The public text model gets no pixels."""
            with BODY_LEASES.command(endpoint):
                result = minecraft["minecraft_call"]("minecraft_observe", {"include_image": False})
                owner = BODY_LEASES.owner(endpoint)
                active = supervisor.receipts().get(owner)
            return {**{key: value for key, value in result.items() if key != "images"},
                    "body_owner": owner, "managed_run": active}
        tools["minecraft_observe"] = minecraft_observe
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
