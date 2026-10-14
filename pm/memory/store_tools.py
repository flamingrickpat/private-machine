"""Scoped MemoryStore tools, exposed both as plain functions and as an MCP server.

Read tools resolve the invocation's visibility snapshot from the working
environment. When a run's ``WriteBatch`` is supplied, the same tools read
the batch's staged view and two proposal tools stage new records or
replacements. Proposals stay invisible to every other run until the batch is
committed; this module never commits.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping
from copy import deepcopy

from fastmcp import FastMCP

from pm.memory.yaml_codec import _deserialize_knoxel, _serialize_knoxel
from pm.memory.agent_tools import MemoryReadSession


def _read_knoxels(environment, batch) -> dict[int, dict]:
    if batch is not None:
        return batch.view().serialize_knoxels()
    return environment.visible_knoxels


def _read_gwt(environment, batch) -> dict[int, dict]:
    if batch is not None:
        return batch.view().serialize_gwt()
    return environment.visible_gwt


def build_tools(environment, batch=None, *, memory: MemoryReadSession | None = None) -> Mapping[str, Callable[..., Any]]:
    """Return the scoped tool functions for one invocation.

    ``batch`` is optional. Without it the tools are read-only. With it the
    read tools include the batch's staged records and two proposal tools add
    to that batch. The batch view is resolved on every call, so a proposal
    made earlier in the same run is visible to later reads.
    Optional ``memory`` adds accepted evidence tools with their own explicit
    workflow view. The caller owns that read session and its cancellation.
    Legacy tools retain the environment's existing grant.
    """

    def ghost_get(knoxel_id: int) -> dict:
        record = _read_knoxels(environment, batch).get(knoxel_id)
        if record is None:
            raise KeyError(knoxel_id)
        return deepcopy(record)

    def ghost_search(query: str) -> list[dict]:
        needle = query.casefold()
        matches: list[dict] = []
        records = _read_knoxels(environment, batch)
        for knoxel_id in sorted(records):
            record = records[knoxel_id]
            content = str(record.get("data", {}).get("content", ""))
            if needle in content.casefold():
                matches.append(deepcopy(record))
        return matches

    def ghost_gwt(tick_id: int) -> dict:
        record = _read_gwt(environment, batch).get(tick_id)
        if record is None:
            raise KeyError(tick_id)
        return deepcopy(record)

    tools: dict[str, Callable[..., Any]] = {
        "ghost_get": ghost_get,
        "ghost_search": ghost_search,
        "ghost_gwt": ghost_gwt,
    }

    if batch is not None:
        def ghost_propose_create(record: dict) -> dict:
            """Stage a new knoxel. Returns the record with its reserved ID."""
            if not isinstance(record, dict):
                raise ValueError("A proposed record must be a mapping.")
            created = batch.create(_deserialize_knoxel(record))
            return _serialize_knoxel(created)

        def ghost_propose_replace(record: dict) -> dict:
            """Stage a replacement for an existing knoxel by ID."""
            if not isinstance(record, dict):
                raise ValueError("A proposed record must be a mapping.")
            batch.replace(_deserialize_knoxel(record))
            return deepcopy(record)

        tools["ghost_propose_create"] = ghost_propose_create
        tools["ghost_propose_replace"] = ghost_propose_replace

    if memory is not None:
        tools.update(memory.tools())
    return tools


def build_mcp(environment, batch=None, *, memory: MemoryReadSession | None = None) -> FastMCP:
    """Register legacy tools and optional accepted evidence tools without network work.

    The caller owns the environment, batch, and optional memory read session.
    Each tool group retains its explicit grant. This factory commits no state.
    """
    tools = build_tools(environment, batch, memory=memory)
    server = FastMCP("ghost")
    for name, fn in tools.items():
        server.tool(name=name)(fn)
    return server


__all__ = ["build_tools", "build_mcp"]
