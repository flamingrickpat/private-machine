"""Scoped Minecraft body tools backed by the real pm-minecraft MCP server.

The application owns the connection. A Python role receives a synchronous
``minecraft_call(name, arguments)`` closure that calls the configured MCP over
HTTP, keeps structured results, and extracts screenshot pixels as base64 with
their media type and frame id. No base64 enters a text prompt.

A test or a unit run can inject a deterministic ``minecraft_call`` instead of a
live server; the workflow contract is identical.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import TypedDict
from workflows.minecraft_body.lease import BodyEpoch
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
async def _client(mcp_url: str, *, timeout: int = MCP_REQUEST_TIMEOUT_SECONDS) -> AsyncGenerator[Client[StreamableHttpTransport], None]:
    """Translate transport failure without hiding cancellation or logic errors.

    FastMCP wraps connection errors in RuntimeError. Keep the upstream cause
    for diagnostics. A lost response cannot establish the command's outcome.
    """
    try:
        async with Client(StreamableHttpTransport(mcp_url), timeout=timeout) as client:
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
    try:
        names = MinecraftTransport(MinecraftTransportSettings(endpoint=mcp_url, request_timeout_seconds=int(timeout_seconds)), None).list_tools()
    except TaskServiceError as error:
        raise TaskServiceError(
            f"No Minecraft body answers at {mcp_url}. Start the MCP server, or fix "
            f"minecraft.mcp_url in the model config.") from error
    if BODY_TOOL not in names:
        raise TaskServiceError(f"{mcp_url} answers but offers no {BODY_TOOL} tool. "
                               f"It is not a Minecraft body endpoint.")
    return names


class MinecraftTransportSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    endpoint: str = DEFAULT_MCP_URL
    request_timeout_seconds: int = MCP_REQUEST_TIMEOUT_SECONDS


class BodyCallSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    epoch: BodyEpoch | None = None


class MinecraftTransport:
    def __init__(self, settings: MinecraftTransportSettings, model: LazyStageModel | None) -> None:
        self.settings = settings
        self.planner = MinecraftPlanner(model)

    async def call_async(self, name: str, arguments: dict[str, JsonValue], settings: BodyCallSettings = BodyCallSettings()) -> MinecraftToolReply:
        async with _client(self.settings.endpoint, timeout=self.settings.request_timeout_seconds) as client:
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
            return {"tool": name, "arguments": arguments, "ok": not result.is_error and data["ok"] is True,
                "result": data, "images": _extract_images(result.content)}

    def call(self, name: str, arguments: dict[str, JsonValue]) -> MinecraftToolReply:
        return asyncio.run(self.call_async(name, arguments))

    def fenced_call(self, name: str, arguments: dict[str, JsonValue], token: BodyEpoch) -> MinecraftToolReply:
        return asyncio.run(self.call_async(name, arguments, BodyCallSettings(epoch=token)))

    def claim(self, token: BodyEpoch) -> MinecraftToolReply:
        return self.call("minecraft_body_claim", {"owner": token.owner, "epoch": token.epoch, "session": token.session})

    def release(self, token: BodyEpoch) -> MinecraftToolReply:
        return self.call("minecraft_body_release", {"owner": token.owner, "epoch": token.epoch, "session": token.session})

    async def list_tools_async(self) -> list[str]:
        async with _client(self.settings.endpoint, timeout=self.settings.request_timeout_seconds) as client:
            return [tool.name for tool in await client.list_tools()]

    def list_tools(self) -> list[str]:
        return asyncio.run(self.list_tools_async())

    async def tool_schemas_async(self) -> tuple[MinecraftActionTool, ...]:
        async with _client(self.settings.endpoint, timeout=self.settings.request_timeout_seconds) as client:
            return tuple(MinecraftActionTool(name=tool.name, description=tool.description,
                parameters=TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_python(tool.inputSchema))
                for tool in await client.list_tools() if not tool.name.startswith("minecraft_body_"))

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
