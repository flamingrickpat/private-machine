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
from typing import Any, Callable

DEFAULT_MCP_URL = "http://127.0.0.1:6767/mcp"


def _extract_images(content: list) -> list[dict]:
    images: list[dict] = []
    for block in content or []:
        if getattr(block, "type", None) != "image":
            continue
        images.append({
            "media_type": getattr(block, "mimeType", None) or "image/png",
            "base64": getattr(block, "data", "") or "",
        })
    return images


def build_minecraft_tools(
    mcp_url: str = DEFAULT_MCP_URL,
    *,
    plan_model: Any = None,
) -> dict[str, Callable[..., Any]]:
    """Return the Minecraft tools for one managed run."""

    async def _call(name: str, arguments: dict) -> dict:
        from fastmcp import Client

        async with Client(mcp_url) as client:
            result = await client.call_tool(name, arguments or {})
            data = result.data if result.data is not None else result.structured_content
            return {
                "tool": name,
                "arguments": dict(arguments or {}),
                "ok": not result.is_error and data["ok"] is True,
                "result": data,
                "images": _extract_images(result.content),
            }

    def minecraft_call(name: str, arguments: dict | None = None) -> dict:
        return asyncio.run(_call(name, dict(arguments or {})))

    def minecraft_list_tools() -> list[str]:
        async def _list() -> list[str]:
            from fastmcp import Client
            async with Client(mcp_url) as client:
                return [tool.name for tool in await client.list_tools()]
        return asyncio.run(_list())

    def minecraft_tool_schemas() -> list[dict]:
        async def _list() -> list[dict]:
            from fastmcp import Client
            async with Client(mcp_url) as client:
                return [{"name": tool.name, "description": tool.description,
                         "parameters": tool.inputSchema} for tool in await client.list_tools()]
        return asyncio.run(_list())

    tools: dict[str, Callable[..., Any]] = {
        "minecraft_call": minecraft_call,
        "minecraft_list_tools": minecraft_list_tools,
        "minecraft_tool_schemas": minecraft_tool_schemas,
        "minecraft_plan": _build_plan_tool(plan_model),
    }
    return tools


def _build_plan_tool(plan_model) -> Callable[..., str]:
    """Let a model choose one allowed action, with a deterministic fallback."""

    def minecraft_plan(observation: str, allowed_tools: list[dict], goal: str) -> str:
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
