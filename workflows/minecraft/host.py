"""Managed-run tools for the Minecraft body workflow.

This module owns the tool factories and request context that a Minecraft
managed run receives: the scoped body transport, the planning reasoner, the
body lease, the persona context, and the project memory grant. The body
workflow itself lives in ``workflows/minecraft`` and
``workflows/minecraft_project``; the transport lives in
``workflows/minecraft_body``. The assembly resolves the endpoint and passes it
here, so a real run needs no environment variable.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Callable

from workflows.minecraft_body.mcp_client import DEFAULT_MCP_URL, build_minecraft_tools
from workflows.minecraft_body.lease import BODY_LEASES
from pm.memory.agent_tools import MemoryToolConfig, MemoryView

if TYPE_CHECKING:
    from pm.memory.embeddings import EmbeddingService
    from pm.memory.store import MemoryStore


def build_minecraft_module_tools(
    plan_model: Callable[..., str] | None = None,
    context_measure: Callable[..., object] | None = None,
    mcp_url: str | None = None,
) -> Callable[[str, object], dict[str, object]]:
    """Return a ``tools_factory(run_id, environment)`` for managed runs.

    ``mcp_url`` is the body endpoint that the assembly resolved from the
    ``minecraft`` section of the model configuration. A real run therefore
    needs no environment variable. When it is None, the old
    ``MINECRAFT_MCP_URL`` override and then the default endpoint apply, which
    is what a unit run that never calls a world tool relies on.
    """
    url = mcp_url or os.environ.get("MINECRAFT_MCP_URL", DEFAULT_MCP_URL)

    def factory(run_id: str, environment: object) -> dict[str, object]:
        """Return the body tools and request hooks of one managed run."""
        return {**build_minecraft_tools(url, plan_model=plan_model),
                "minecraft_reason": plan_model,
                "minecraft_context_measure": context_measure,
                "minecraft_endpoint": lambda: url,
                "minecraft_lease": lambda: BODY_LEASES.acquire(url, run_id)}

    return factory


__all__ = ["build_minecraft_module_tools"]


def project_memory_config(store: MemoryStore, cancel: object,
                         embeddings: EmbeddingService | None = None) -> MemoryToolConfig:
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


def workflow_shard_story(store: MemoryStore, workflow: str = "minecraft") -> str:
    """Render the workspace shard of ``workflow`` from the newest cycle state, or an empty text.

    The story uses the words of the main cognition story prefix: the shard
    items and the tail of the newest workflow results, without record ids.
    The caller holds the memory-store lock.
    """
    from pm.records.workspace import ShardKey
    from workflows.main_cognition.schemas import ActionRequest, RequestShard
    from workflows.public_speech.shard_context import shard_story
    key = ShardKey(kind="workflow", ref=workflow)
    if not store.states or not store.states[-1].workspace.has_shard(key):
        return ""
    shard = store.states[-1].workspace.shard(key)
    request = ActionRequest(shards=[RequestShard(name=workflow, shard=shard)])
    return shard_story(store.snapshot(), request, store.get_companion_name())


def persona_context(store: MemoryStore) -> dict[str, Any]:
    """Return the companion identity that body workflows receive in their request.

    ``workspace`` is the story of the Minecraft workflow shard of the newest
    cycle (:func:`workflow_shard_story`). The supervisor calls this under the
    memory-store lock at each launch.
    """
    from workflows.public_speech.public_context import build_public_context
    from pm.records.mental_state import create_empty_state
    public = build_public_context(store).model_dump(mode="json")
    state = store.states[-1].mental_state if store.states else create_empty_state()
    return {"persona_context": {
        "companion_name": store.get_companion_name(),
        "user_name": store.get_user_name(),
        "character_card": store.get_simple_character_story_block(),
        "accepted_context": public,
        "workspace": workflow_shard_story(store),
        "accepted_emotions": state.state_emotions.model_dump(),
        "accepted_relationship": state.state_relationship.model_dump() if state.state_relationship else None,
    }}
