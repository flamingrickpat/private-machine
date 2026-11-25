"""Transport for a configured Minecraft MCP endpoint.

The companion shares one transport across body workflows, polling and dialogue.
It records the time of successful observations and invalidates cached body state
when the live server reports a change of life or a connection failure.
"""
from __future__ import annotations

from datetime import datetime, timezone
from threading import RLock
from workflows.minecraft_body.world_text import BodyReading, WorldObservation

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import TypedDict
from workflows.minecraft_body.lease import BodyEpoch, BODY_LEASES, BodyLeases
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import ContentBlock, ImageContent
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter
from workflows.minecraft_body.planning_prompt import MinecraftActionTool

import httpx
from pm.constants import MINECRAFT_REQUEST_SECONDS
from pm.runs.errors import TaskServiceError
from pm.llm.stage_model import LazyStageModel
from pm.llm.session import ModelSession

DEFAULT_MCP_URL = "http://127.0.0.1:6767/mcp"
MCP_REQUEST_TIMEOUT_SECONDS = MINECRAFT_REQUEST_SECONDS
#: A Minecraft body endpoint must offer this direct read tool. The server
#: exposes its body actions under their real names; ``minecraft_call`` is the
#: local wrapper, not a remote tool. Another MCP server on the same port does
#: not pass the startup check.
BODY_TOOL = "minecraft_observe"


@asynccontextmanager
async def _client(mcp_url: str, *, headers: dict[str, str] | None = None, timeout: float = MCP_REQUEST_TIMEOUT_SECONDS) -> AsyncGenerator[Client[StreamableHttpTransport], None]:
    """Translate transport failure without hiding cancellation or logic errors.

    FastMCP wraps connection errors in RuntimeError. Keep the upstream cause
    for diagnostics. A lost response cannot establish the command's outcome.
    """
    try:
        async with Client(StreamableHttpTransport(mcp_url, headers=headers), timeout=timeout) as client:
            yield client
    except (httpx.TransportError, TimeoutError) as error:
        raise TaskServiceError("Minecraft transport failed. The task outcome is unverified.") from error
    except RuntimeError as error:
        if not isinstance(error.__cause__, (httpx.TransportError, TimeoutError)):
            raise
        raise TaskServiceError("Minecraft connection failed. The task outcome is unverified.") from error



class MinecraftImageWire(TypedDict):
    media_type: str
    base64: str


class MinecraftToolReply(TypedDict):
    tool: str
    arguments: dict[str, JsonValue]
    ok: bool
    result: JsonValue
    images: list[MinecraftImageWire]
    observed_at: str


def _extract_images(content: list[ContentBlock]) -> list[MinecraftImageWire]:
    images: list[MinecraftImageWire] = []
    for block in content:
        if isinstance(block, ImageContent):
            images.append(MinecraftImageWire(media_type=block.mimeType, base64=block.data))
    return images


def probe_body(mcp_url: str, *, timeout_seconds: float = 5.0) -> list[str]:
    """Return the tool names of the Minecraft body at ``mcp_url``, or raise.

    The assembly calls this once at startup, before any module thread runs.
    The check runs the real MCP handshake and lists tools, so a port that
    answers with something else fails too. ``TaskServiceError`` names the
    endpoint and says what to do, so a wrong ``mcp_url`` is not mistaken for a
    model or task failure later in the session.
    """
    transport = MinecraftTransport(MinecraftTransportSettings(endpoint=mcp_url), None)
    return list(transport.probe(timeout_seconds))


class MinecraftTransportSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    endpoint: str = DEFAULT_MCP_URL
    request_timeout_seconds: int = MCP_REQUEST_TIMEOUT_SECONDS


class BodyCallSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    epoch: BodyEpoch | None = None


class MinecraftTransport:
    def __init__(self, settings: MinecraftTransportSettings, model: LazyStageModel | None,
                 *, leases: BodyLeases = BODY_LEASES) -> None:
        self.settings = settings
        self.leases: BodyLeases = leases
        self.planner = MinecraftPlanner(model)
        self.headers: dict[str, str] = {}
        self.read_lock = RLock()
        self.latest: BodyReading | None = None
        self.connected: bool | None = None
        self.tools: tuple[str, ...] = ()


    async def call_async(self, name: str, arguments: dict[str, JsonValue], settings: BodyCallSettings = BodyCallSettings()) -> MinecraftToolReply:
        try:
            async with _client(self.settings.endpoint, headers=self.headers, timeout=self.settings.request_timeout_seconds) as client:
                dispatch_name = name
                dispatch_arguments = arguments
                token = settings.epoch
                if token is not None:
                    dispatch_name = "minecraft_body_call"
                    dispatch_arguments = {"owner": token.owner, "epoch": token.epoch, "session": token.session,
                        "tool": name, "arguments": arguments}
                result = await client.call_tool(dispatch_name, dispatch_arguments)
                data = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(
                    result.data if result.data is not None else result.structured_content)
                observed_at = datetime.now(timezone.utc)
                with self.read_lock:
                    if data.get("reason") in {"not_connected", "not_spawned"} or name == "minecraft_disconnect" and data["ok"]:
                        self.connected = False
                        self.latest = None
                    elif name != BODY_TOOL and self.latest is not None:
                        previous = self.latest.observation
                        if (data.get("death_count_after", previous.death_count) != previous.death_count
                                or data.get("body_session", previous.body_session) != previous.body_session
                                or data.get("died") is True or data.get("respawned") is True
                                or name == "minecraft_respawn" and data["ok"]):
                            self.latest = None
                    elif name == BODY_TOOL and data["ok"]:
                        self.latest = BodyReading(observed_at=observed_at,
                                                  observation=WorldObservation.model_validate(data))
                        self.connected = True
                return {"tool": name, "arguments": arguments, "ok": not result.is_error and data["ok"] is True,
                    "result": data, "images": _extract_images(result.content), "observed_at": observed_at.isoformat()}
        except TaskServiceError:
            with self.read_lock:
                self.latest = None
                self.connected = None
            raise

    def call(self, name: str, arguments: dict[str, JsonValue]) -> MinecraftToolReply:
        return asyncio.run(self.call_async(name, arguments))

    def fenced_call(self, name: str, arguments: dict[str, JsonValue], token: BodyEpoch) -> MinecraftToolReply:
        return asyncio.run(self.call_async(name, arguments, BodyCallSettings(epoch=token)))

    def claim(self, token: BodyEpoch) -> MinecraftToolReply:
        return self.call("minecraft_body_claim", {"owner": token.owner, "epoch": token.epoch, "session": token.session})

    def release(self, token: BodyEpoch) -> MinecraftToolReply:
        return self.call("minecraft_body_release", {"owner": token.owner, "epoch": token.epoch, "session": token.session})

    def probe(self, timeout_seconds: float) -> tuple[str, ...]:
        try:
            self.tools = tuple(asyncio.run(self.list_tools_async(timeout_seconds)))
        except TaskServiceError as error:
            raise TaskServiceError(f"No Minecraft body answers at {self.settings.endpoint}.") from error
        if BODY_TOOL not in self.tools:
            raise TaskServiceError(f"{self.settings.endpoint} offers no {BODY_TOOL} tool.")
        return self.tools

    async def list_tools_async(self, timeout_seconds: float | None = None) -> list[str]:
        timeout = self.settings.request_timeout_seconds if timeout_seconds is None else timeout_seconds
        async with _client(self.settings.endpoint, headers=self.headers, timeout=timeout) as client:
            return [tool.name for tool in await client.list_tools()]

    def list_tools(self) -> list[str]:
        return asyncio.run(self.list_tools_async())

    async def tool_schemas_async(self) -> tuple[MinecraftActionTool, ...]:
        async with _client(self.settings.endpoint, headers=self.headers, timeout=self.settings.request_timeout_seconds) as client:
            return tuple(MinecraftActionTool(name=tool.name, description=tool.description,
                parameters=TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(tool.inputSchema))
                for tool in await client.list_tools() if not tool.name.startswith("minecraft_body_")
                and tool.name not in {"minecraft_connect", "minecraft_disconnect", "minecraft_execute_typescript", "minecraft_suicide"})

    def tool_schemas(self) -> tuple[MinecraftActionTool, ...]:
        return asyncio.run(self.tool_schemas_async())

    def tool_schema_wire(self) -> list[dict[str, JsonValue]]:
        return [tool.model_dump(mode="json") for tool in self.tool_schemas()]


class MinecraftPlanner:
    def __init__(self, model: LazyStageModel | None) -> None:
        self.model = model

    def choose_action(self, observation: str, allowed_tools: tuple[MinecraftActionTool, ...], goal: str) -> str:
        from workflows.minecraft_body.planning_prompt import MinecraftActionRenderInput, render_minecraft_action
        from pm.context_render.prompt import RendererCapture, RendererInput
        from pm.llm.generation import GenerationSettings
        if self.model is None:
            raise RuntimeError("Minecraft planning needs a configured model.")
        input = MinecraftActionRenderInput(goal, observation, allowed_tools)
        session = ModelSession(self.model)
        settings = session.render_settings()
        with RendererCapture(session.capture_settings("minecraft.action"), RendererInput("minecraft.action", input, settings)) as capture:
            prompt = render_minecraft_action(input, settings)
            capture.save_output(prompt)
        return session.complete(prompt.turns, settings=GenerationSettings(enable_thinking=False))


__all__ = ["DEFAULT_MCP_URL", "MinecraftTransport", "MinecraftTransportSettings", "probe_body"]
