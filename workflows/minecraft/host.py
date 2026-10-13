"""Managed-run tools for the Minecraft body workflow."""

from __future__ import annotations

import os
from typing import Callable

from workflows.minecraft_body.mcp_client import DEFAULT_MCP_URL, build_minecraft_tools
from workflows.minecraft_body.lease import BODY_LEASES


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


def persona_context(store) -> dict:
    """Return the companion identity that body workflows receive in their request.

    The supervisor calls this under the memory-store lock at each launch.
    """
    from workflows.public_speech.public_context import build_public_context
    public = build_public_context(store, {}).model_dump(mode="json")
    return {"persona_context": {
        "companion_name": store.get_companion_name(),
        "user_name": store.get_user_name(),
        "character_card": store.get_simple_character_story_block(),
        "accepted_context": public,
    }}
