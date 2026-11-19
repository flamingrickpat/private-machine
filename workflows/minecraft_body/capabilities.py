"""Read current domain capabilities without moving the Minecraft character."""
from __future__ import annotations

from pm.utils.operating_system import OperatingSystem
from pm import vclock

import asyncio
import socket
from datetime import datetime
from typing import TYPE_CHECKING, Literal, cast
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field
from pm.records.mental_state import StateEmotions
from workflows.inner_thought.request_views import CapabilityInspection

import anyio
import httpx
from mcp.shared.exceptions import McpError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pm.memory.store import MemoryStore
    from pm.runs.supervisor import ManagedRunSupervisor
    from pm.host.application import Application


def _transport_failure(error: BaseException) -> bool:
    """Recognize client transport failures without hiding programming errors."""
    if isinstance(error, BaseExceptionGroup):
        group = cast(BaseExceptionGroup[BaseException], error)
        return all(_transport_failure(item) for item in group.exceptions)
    if isinstance(error, (OSError, TimeoutError, httpx.HTTPError, McpError,
                          anyio.ClosedResourceError, anyio.BrokenResourceError, anyio.EndOfStream)):
        return True
    # FastMCP wraps closed sessions and initialization timeouts in RuntimeError.
    return isinstance(error, RuntimeError) and error.__cause__ is not None and _transport_failure(error.__cause__)


class MinecraftCapabilityState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["unknown", "disconnected", "connected"] = "unknown"
    endpoint: str = ""
    tools: tuple[str, ...] = ()
    visual_interpretation: bool = False
    observed_at: datetime | None = None
    error_type: str = ""


class CapabilityEvidenceLimit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: int = 48
    record_characters: int = 1200
    omitted_ids: tuple[int, ...] = ()


class CapabilitySnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: bool = True
    persistent_memory: bool = True
    human_senses: tuple[str, ...] = ()
    human_body: bool = False
    configured: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    managed_workflows: tuple[str, ...] = ()
    minecraft: MinecraftCapabilityState = Field(default_factory=MinecraftCapabilityState)
    implemented_emotions: tuple[str, ...] = ()
    self_model: CapabilityInspection = Field(default_factory=CapabilityInspection)
    emotion_state: StateEmotions | None = None
    evidence_limit: CapabilityEvidenceLimit = Field(default_factory=CapabilityEvidenceLimit)


async def discover_body_tools(endpoint: str) -> tuple[str, ...]:
    from fastmcp import Client
    async with Client(endpoint) as client:
        return tuple(item.name for item in await client.list_tools())


def capability_snapshot(ghost: MemoryStore, workflow_names: Iterable[str]) -> CapabilitySnapshot:
    from pm.app.operator_config import environment_values
    values, _ = environment_values()
    url = OperatingSystem.environment.get("MINECRAFT_MCP_URL") or values.get("MINECRAFT_MCP_URL") or "http://127.0.0.1:6767/mcp"
    minecraft = MinecraftCapabilityState(status="disconnected", endpoint=url, observed_at=vclock.now_aware())
    try:
        minecraft.tools = asyncio.run(asyncio.wait_for(discover_body_tools(url), timeout=5))
        with socket.create_connection((values.get("PM_MINECRAFT_HOST", "127.0.0.1"),
                                       int(values.get("PM_MINECRAFT_PORT", "12345"))), timeout=2):
            pass
        minecraft.status = "connected"
    except Exception as error:
        if not _transport_failure(error):
            raise
        minecraft.error_type = type(error).__name__
    config = ghost.ghost_config
    return CapabilitySnapshot(configured=tuple(config.supported_capabilities),
        unavailable=tuple(str(item) for item in config.unsupported_capabilities),
        notes=tuple(str(item) for item in config.capability_notes),
        managed_workflows=tuple(sorted(workflow_names)), minecraft=minecraft,
        implemented_emotions=tuple(StateEmotions.model_fields))


@dataclass(frozen=True, slots=True)
class CapabilitySnapshotOwner:
    store: MemoryStore
    supervisor: ManagedRunSupervisor
    application: Application

    def snapshot(self) -> CapabilitySnapshot:
        from workflows.introspection.inspection_owner import capability_inspection
        snapshot = capability_snapshot(self.store, self.supervisor.workflows)
        snapshot.self_model = capability_inspection(self.application.capability_overlay)
        return snapshot


__all__ = ["capability_snapshot", "CapabilitySnapshot", "CapabilitySnapshotOwner", "MinecraftCapabilityState", "CapabilityEvidenceLimit"]
