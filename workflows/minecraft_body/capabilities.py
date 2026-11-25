"""Read current domain capabilities without moving the Minecraft character."""
from __future__ import annotations

from pm import vclock
from workflows.minecraft_body.lease import BODY_LEASES
from workflows.minecraft_body.mcp_client import MinecraftTransport

import asyncio

from datetime import datetime
from typing import TYPE_CHECKING, Literal, cast
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field
from pm.records.mental_state import StateEmotions
from pm.runs.run_receipt import ACTIVE_STATES
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
    runtime_orientation: str = ""
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


class BodyConnection(BaseModel):
    connected: bool = False
    spawned: bool = False
    reason: str | None = None


async def inspect_body(endpoint: str) -> MinecraftCapabilityState:
    from fastmcp import Client
    state = MinecraftCapabilityState(endpoint=endpoint, observed_at=vclock.now_aware())
    async with Client(endpoint) as client:
        state.tools = tuple(tool.name for tool in await client.list_tools())
        info = await client.call_tool("minecraft_info", {}, raise_on_error=False)
        connection = BodyConnection.model_validate(info.structured_content)
        if connection.reason == "not_connected" or not info.is_error:
            state.status = "connected" if connection.connected and connection.spawned else "disconnected"
        else:
            state.error_type = connection.reason or "tool_error"
    return state


def capability_snapshot(ghost: MemoryStore, workflow_names: Iterable[str],
                        endpoint: str | None = None) -> CapabilitySnapshot:
    minecraft = MinecraftCapabilityState()
    if endpoint:
        try:
            minecraft = asyncio.run(asyncio.wait_for(inspect_body(endpoint), timeout=8))
        except Exception as error:
            if not _transport_failure(error):
                raise
            minecraft = MinecraftCapabilityState(endpoint=endpoint, observed_at=vclock.now_aware(),
                                                 error_type=type(error).__name__)
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

    endpoint: str | None = None
    suites: tuple[str, ...] = ()

    transport: MinecraftTransport | None = None

    def suite_status(self, name: str) -> str:
        tools = self.supervisor.workflows.get("tool_call")
        if tools and name in tools.reserved_suites and self.endpoint:
            spec = self.supervisor.workflows.get("minecraft_project")
            if spec and spec.reservation:
                owner = spec.reservation.owner()
            else:
                leases = self.transport.leases if self.transport is not None else BODY_LEASES
                owner = leases.owner(self.endpoint)
            if owner:
                return f"{name} busy with {owner} (status, wait, yield or terminate)"
        return f"{name} available"

    def orientation(self, snapshot: CapabilitySnapshot) -> str:
        from pm.introspection.runtime_state import RuntimeSelfState, machine_identity
        active = tuple(f"{run['workflow_name']} {run['run_id']} ({run['state']})"
                       for run in self.store.read_run_receipts().values() if run["state"] in ACTIVE_STATES)
        body = snapshot.minecraft.status if self.endpoint else "no assigned body"
        if self.transport is not None and self.endpoint:
            with self.transport.read_lock:
                if self.transport.latest is not None:
                    body += ". " + self.transport.latest.summary()
        return RuntimeSelfState(machine_identity(), snapshot.configured,
            ("dialogue_story", "self_management", "focus_management", "theory_of_mind", *snapshot.managed_workflows),
            tuple(self.suite_status(name) for name in self.suites), body, active).render()

    def snapshot(self) -> CapabilitySnapshot:
        from workflows.introspection.inspection_owner import capability_inspection
        snapshot = capability_snapshot(self.store, self.supervisor.workflows)
        if self.transport is not None and self.endpoint:
            with self.transport.read_lock:
                reading = self.transport.latest
                connected = self.transport.connected
                snapshot.minecraft = MinecraftCapabilityState(endpoint=self.endpoint, tools=self.transport.tools,
                    status="unknown" if connected is None else ("connected" if connected else "disconnected"),
                    observed_at=reading.observed_at if reading else None)

        snapshot.runtime_orientation = self.orientation(snapshot)
        snapshot.self_model = capability_inspection(self.application.capability_overlay)
        return snapshot


__all__ = ["capability_snapshot", "CapabilitySnapshot", "CapabilitySnapshotOwner", "MinecraftCapabilityState", "CapabilityEvidenceLimit"]
