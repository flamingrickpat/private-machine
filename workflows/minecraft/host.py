"""Managed-run tools for the Minecraft body workflow."""

from __future__ import annotations

import os
from typing import Callable

from workflows.minecraft_body.mcp_client import DEFAULT_MCP_URL, build_minecraft_tools
from workflows.minecraft_body.lease import BODY_LEASES
from pm.memory.agent_tools import MemoryToolConfig, MemoryView


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


def project_memory_config(store, cancel, embeddings=None) -> MemoryToolConfig:
    """Grant the project planner accepted in-world history and its sources.

    Query words affect relevance only. Prior preferences, corrections, and
    project outcomes remain visible. The native child connects to this accepted
    owner through MCP. It never reopens the owner's writable checkpoint.
    """
    def select() -> MemoryView:
        with store.lock:
            return MemoryView(record_ids=frozenset(store.revisions),
                              scopes=frozenset({(-1, "in_world")}),
                              frontier=(store.current_tick_id, 2**63 - 1))

    return MemoryToolConfig(store, select(), semantic=True, lexical=True, graph=True,
                            embeddings=embeddings, cancel=cancel, refresh_view=select)


def persona_context(store) -> dict:
    """Return the companion identity that body workflows receive in their request.

    The supervisor calls this under the memory-store lock at each launch.
    """
    from workflows.public_speech.public_context import build_public_context
    from pm.records.mental_state import FullMentalState, create_empty_state
    public = build_public_context(store, {}).model_dump(mode="json")
    state = FullMentalState.init_from_list(store.states[-1].mental_state_latent_vec) if store.states else create_empty_state()
    return {"persona_context": {
        "companion_name": store.get_companion_name(),
        "user_name": store.get_user_name(),
        "character_card": store.get_simple_character_story_block(),
        "accepted_context": public,
        "accepted_emotions": state.state_emotions.model_dump(),
        "accepted_relationship": state.state_relationship.model_dump() if state.state_relationship else None,
    }}
