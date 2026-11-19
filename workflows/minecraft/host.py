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

from dataclasses import dataclass, field
from pathlib import Path
from workflows.minecraft_project.services import ProjectAuditOwner
from workflows.minecraft.persona import MinecraftPersona
from threading import Event
from typing import TYPE_CHECKING
from pydantic import BaseModel
from pm.utils.operating_system import OperatingSystem
from pm.llm.stage_model import LazyStageModel
from pm.runs.role_runtime import PMWorkflowRuntime
from workflows.minecraft_body.mcp_client import DEFAULT_MCP_URL, MinecraftTransport, MinecraftTransportSettings
from workflows.minecraft.runtime import MinecraftBodyLease, MinecraftWorkflowRuntime
from pm.memory.agent_tools import MemoryToolConfig, MemoryView, MemoryRefreshScope

if TYPE_CHECKING:
    from pm.memory.embeddings import EmbeddingService
    from pm.memory.store import MemoryStore


@dataclass(frozen=True, slots=True)
class MinecraftRuntimeOwner:
    transport: MinecraftTransport
    project_audit: ProjectAuditOwner | None = None
    evidence_sources: dict[str, Path] = field(default_factory=dict[str, Path])

    def build_runtime(self, runtime: PMWorkflowRuntime, run_id: str) -> MinecraftWorkflowRuntime:
        return MinecraftWorkflowRuntime(environment=runtime.environment, stop_event=runtime.stop_event,
            tools=runtime.tools, propagate_python_errors=runtime.propagate_python_errors,
            agent_model=runtime.agent_model, mcp_servers=runtime.mcp_servers, child_factory=runtime.child_factory,
            classification_models=runtime.classification_models, classification_session=runtime.classification_session,
            model_session=runtime.model_session, embedding_service=runtime.embedding_service,
            managed_boundary=runtime.managed_boundary, tool_call_config=runtime.tool_call_config,
            memory_store=runtime.memory_store, memory_context=runtime.memory_context,
            minecraft_transport=self.transport, body_lease=MinecraftBodyLease(self.transport.settings.endpoint, run_id),
            project_audit=self.project_audit, evidence_sources=self.evidence_sources.copy())


def build_minecraft_module_tools(plan_model: LazyStageModel | None = None,
                                 mcp_url: str | None = None) -> MinecraftRuntimeOwner:
    url = mcp_url or OperatingSystem.environment.get("MINECRAFT_MCP_URL", DEFAULT_MCP_URL)
    return MinecraftRuntimeOwner(MinecraftTransport(MinecraftTransportSettings(endpoint=url), plan_model))


@dataclass(frozen=True, slots=True)
class ProjectMemoryOwner:
    embeddings: EmbeddingService | None

    def create(self, store: MemoryStore, cancel: Event) -> MemoryToolConfig:
        return project_memory_config(store, cancel, self.embeddings)


def project_memory_config(store: MemoryStore, cancel: Event,
                         embeddings: EmbeddingService | None = None) -> MemoryToolConfig:
    with store.lock:
        view = MemoryView(record_ids=frozenset(store.revisions), scopes=frozenset({(-1, "in_world")}),
            frontier=(store.current_tick_id, 2**63 - 1))
    return MemoryToolConfig(store, view, semantic=True, lexical=True, graph=True,
        embeddings=embeddings, cancel=cancel, refresh_scope=MemoryRefreshScope.AcceptedInWorld)


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



class MinecraftRequestContext(BaseModel):
    persona_context: MinecraftPersona


@dataclass(frozen=True, slots=True)
class MinecraftContextOwner:
    def capture(self, store: MemoryStore) -> MinecraftRequestContext:
        return MinecraftRequestContext(persona_context=MinecraftPersona(companion_name=store.get_companion_name(),
            user_name=store.get_user_name(), character_card=store.get_simple_character_story_block(),
            workspace=workflow_shard_story(store)))
