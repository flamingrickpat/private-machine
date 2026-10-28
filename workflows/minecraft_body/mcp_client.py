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
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
from pm.constants import MINECRAFT_REQUEST_SECONDS
from pm.runs.errors import TaskServiceError

DEFAULT_MCP_URL = "http://127.0.0.1:6767/mcp"
MCP_REQUEST_TIMEOUT_SECONDS = MINECRAFT_REQUEST_SECONDS
#: A Minecraft body endpoint must offer this direct read tool. The server
#: exposes its body actions under their real names; ``minecraft_call`` is the
#: local wrapper, not a remote tool. Another MCP server on the same port does
#: not pass the startup check.
BODY_TOOL = "minecraft_observe"


@asynccontextmanager
async def _client(mcp_url: str, *, timeout: int = MCP_REQUEST_TIMEOUT_SECONDS) -> AsyncIterator[Any]:
    """Translate transport failure without hiding cancellation or logic errors.

    FastMCP wraps connection errors in RuntimeError. Keep the upstream cause
    for diagnostics. A lost response cannot establish the command's outcome.
    """
    from fastmcp import Client
    try:
        async with Client(mcp_url, timeout=timeout) as client:
            yield client
    except (httpx.TransportError, TimeoutError) as error:
        raise TaskServiceError("Minecraft transport failed. The task outcome is unverified.") from error
    except RuntimeError as error:
        if not isinstance(error.__cause__, (httpx.TransportError, TimeoutError)):
            raise
        raise TaskServiceError("Minecraft connection failed. The task outcome is unverified.") from error


def _extract_images(content: list[Any]) -> list[dict[str, str]]:
    """Return the image blocks of one MCP result as media type and base64 data."""
    images: list[dict[str, str]] = []
    for block in content or []:
        if getattr(block, "type", None) != "image":
            continue
        images.append({
            "media_type": getattr(block, "mimeType", None) or "image/png",
            "base64": getattr(block, "data", "") or "",
        })
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
        names = build_minecraft_tools(mcp_url, request_timeout_seconds=int(timeout_seconds))["minecraft_list_tools"]()
    except TaskServiceError as error:
        raise TaskServiceError(
            f"No Minecraft body answers at {mcp_url}. Start the MCP server, or fix "
            f"minecraft.mcp_url in the model config.") from error
    if BODY_TOOL not in names:
        raise TaskServiceError(f"{mcp_url} answers but offers no {BODY_TOOL} tool. "
                               f"It is not a Minecraft body endpoint.")
    return names


def build_minecraft_tools(
    mcp_url: str = DEFAULT_MCP_URL,
    *,
    plan_model: Any = None,
    request_timeout_seconds: int = MCP_REQUEST_TIMEOUT_SECONDS,
) -> dict[str, Callable[..., Any]]:
    """Return scoped transport tools with a request deadline in seconds.

    Managed runs use the long action deadline. Periodic observations can select
    a shorter deadline. A transport error leaves external outcomes unverified.
    """

    async def _call(name: str, arguments: dict[str, Any], token: dict[str, str] | None = None) -> dict[str, Any]:
        async with _client(mcp_url, timeout=request_timeout_seconds) as client:
            dispatch_name, dispatch_arguments = name, arguments or {}
            if token is not None:
                dispatch_name = "minecraft_body_call"
                dispatch_arguments = {**{key: token[key] for key in ("owner", "epoch", "session")},
                                      "tool": name, "arguments": arguments or {}}
            result = await client.call_tool(dispatch_name, dispatch_arguments)
            data = result.data if result.data is not None else result.structured_content
            return {
                "tool": name,
                "arguments": dict(arguments or {}),
                "ok": not result.is_error and data["ok"] is True,
                "result": data,
                "images": _extract_images(result.content),
            }

    def minecraft_call(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Call one body tool and return the retained structured result."""
        return asyncio.run(_call(name, dict(arguments or {})))

    def minecraft_fenced_call(name: str, arguments: dict[str, Any], token: dict[str, str]) -> dict[str, Any]:
        """Send the pinned epoch to the server with the original tool schema."""
        return asyncio.run(_call(name, arguments, token))

    def minecraft_claim(token: dict[str, str]) -> dict[str, Any]:
        """Wait for old server work before activating this local lease."""
        return minecraft_call("minecraft_body_claim", {key: token[key] for key in ("owner", "epoch", "session")})

    def minecraft_release(token: dict[str, str]) -> dict[str, Any]:
        """Release remote ownership after external cleanup. Retain its epoch."""
        return minecraft_call("minecraft_body_release", {key: token[key] for key in ("owner", "epoch", "session")})

    def minecraft_list_tools() -> list[str]:
        async def _list() -> list[str]:
            async with _client(mcp_url, timeout=request_timeout_seconds) as client:
                return [tool.name for tool in await client.list_tools()]
        return asyncio.run(_list())

    def minecraft_tool_schemas() -> list[dict[str, Any]]:
        """Return the server's public tool schemas, without body bookkeeping tools."""

        async def _list() -> list[dict[str, Any]]:
            async with _client(mcp_url, timeout=request_timeout_seconds) as client:
                return [{"name": tool.name, "description": tool.description,
                         "parameters": tool.inputSchema} for tool in await client.list_tools()
                        if not tool.name.startswith("minecraft_body_")]
        return asyncio.run(_list())

    tools: dict[str, Callable[..., Any]] = {
        "minecraft_call": minecraft_call,
        "minecraft_fenced_call": minecraft_fenced_call,
        "minecraft_claim": minecraft_claim,
        "minecraft_release": minecraft_release,
        "minecraft_list_tools": minecraft_list_tools,
        "minecraft_tool_schemas": minecraft_tool_schemas,
        "minecraft_plan": _build_plan_tool(plan_model),
    }
    # These capabilities stay on the granted transport closure. A role cannot
    # obtain another connection or invent a token through its public tool list.
    if isinstance(mcp_url, str):
        minecraft_call.claim = minecraft_claim
        minecraft_call.release = minecraft_release
        minecraft_call.fenced_call = minecraft_fenced_call
    return tools


def _build_plan_tool(plan_model: Callable[..., str] | None) -> Callable[..., str]:
    """Let a model choose one allowed action, with a deterministic fallback."""

    def minecraft_plan(observation: str, allowed_tools: list[dict[str, Any]], goal: str) -> str:
        if plan_model is None:
            raise RuntimeError("Minecraft planning needs a configured model or an explicit action_tool.")
        import json
        tool_list = json.dumps(allowed_tools, ensure_ascii=False)
        messages = [
            {
                "role": "system",
                "content": (
                    "You choose exactly one Minecraft action from the allowed "
                    "tool list. Reply with only the tool name and a JSON argument "
                    "object on one line: <tool> <json>."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Goal: {goal}\nAllowed tools: {tool_list}\n"
                    f"Observation:\n{observation}\n\nChoose one action."
                ),
            },
        ]
        return plan_model(messages)

    return minecraft_plan


__all__ = ["DEFAULT_MCP_URL", "build_minecraft_tools"]
