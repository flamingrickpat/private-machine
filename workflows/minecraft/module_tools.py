"""Managed-run tools for the Minecraft body workflow."""

from __future__ import annotations

import os
from typing import Callable

from runtime.minecraft_tools import DEFAULT_MCP_URL, build_minecraft_tools
from runtime.body_lease import BODY_LEASES


def build_minecraft_module_tools(plan_model=None, context_measure=None) -> Callable[[str, object], dict]:
    """Return a ``tools_factory(run_id, environment)`` for managed runs.

    ``MINECRAFT_MCP_URL`` selects the real pm-minecraft MCP endpoint.
    """
    url = os.environ.get("MINECRAFT_MCP_URL", DEFAULT_MCP_URL)

    def factory(run_id: str, environment) -> dict:
        return {**build_minecraft_tools(url, plan_model=plan_model),
                "minecraft_reason": plan_model,
                "minecraft_context_measure": context_measure,
                "minecraft_endpoint": lambda: url,
                "minecraft_lease": lambda: BODY_LEASES.acquire(url, run_id)}

    return factory


__all__ = ["build_minecraft_module_tools"]
