"""Scoped MemoryStore tools, exposed both as plain functions and as an MCP server.

Read tools resolve the invocation's visibility snapshot from the working
environment. When a run's ``WriteBatch`` is supplied, the same tools read
the batch's staged view and two proposal tools stage new records or
replacements. Proposals stay invisible to every other run until the batch is
committed; this module never commits.
"""
from __future__ import annotations


from collections.abc import Callable, Mapping
from typing import Protocol, runtime_checkable, TypeAlias
import json

from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from pm.context_render import ContextKind, render_context
from pm.context_render.records import RecordFile, record_file_payload, record_file_schema
from pm.memory.agent_tools import MemoryReadSession
from pm.memory.snapshot import Snapshot
from pm.memory.write_batch import WriteBatch
from pm.memory.yaml_codec import RecordPayload
from pm.records.base import RecordBase
from pm.records.global_workspace import GwtTurnRecord

RecordInput: TypeAlias = RecordBase | Mapping[str, object]
TurnInput: TypeAlias = GwtTurnRecord | Mapping[str, object]


class RecordTextReplacement(BaseModel):
    """A model may change content. The host owns every other record field."""

    model_config = ConfigDict(extra="forbid", strict=True)
    id: int = Field(description="The existing knoxel ID to supersede.")
    content: str = Field(description="The complete replacement text, without changing other fields.")


class MemoryToolEnvironment(Protocol):
    """The explicit read grant required by the legacy memory tool adapter."""

    @property
    def visible_knoxels(self) -> Mapping[int, RecordInput]: ...

    @property
    def visible_gwt(self) -> Mapping[int, TurnInput]: ...


@runtime_checkable
class HistoricalMemoryEnvironment(Protocol):
    """An environment that also exposes exact historical records in its frozen view."""

    @property
    def ghost_view(self) -> Snapshot | None: ...


def _read_knoxels(environment: MemoryToolEnvironment, batch: WriteBatch | None) -> Mapping[int, RecordInput]:
    return batch.view().all_knoxels if batch is not None else environment.visible_knoxels


def _as_record(value: RecordInput) -> RecordBase:
    return value if isinstance(value, RecordBase) else RecordPayload.model_validate(value).root


def build_tools(environment: MemoryToolEnvironment, batch: WriteBatch | None = None, *,
                memory: MemoryReadSession | None = None) -> Mapping[str, Callable[..., object]]:
    """Bind model-facing projections to an explicit invocation grant.

    Staged proposals remain local until their owner commits. Read responses use
    central rendering. Stored records and their complete metadata stay intact.
    """
    def ghost_get(knoxel_id: int) -> RecordFile:
        records = _read_knoxels(environment, batch)
        if knoxel_id in records:
            return record_file_payload(_as_record(records[knoxel_id]))
        view = batch.view() if batch is not None else environment.ghost_view if isinstance(environment, HistoricalMemoryEnvironment) else None
        if view is None:
            raise KeyError(knoxel_id)
        return record_file_payload(view.get(knoxel_id))

    def ghost_search(query: str) -> list[RecordFile]:
        needle = query.casefold()
        records = _read_knoxels(environment, batch)
        matches: list[RecordFile] = []
        for knoxel_id in sorted(records):
            record = _as_record(records[knoxel_id])
            if needle in record.content.casefold():
                matches.append(record_file_payload(record))
        return matches

    def ghost_gwt(tick_id: int) -> dict[str, JsonValue]:
        if batch is not None:
            turn = batch.view().gwt_turns[tick_id]
        else:
            value = environment.visible_gwt[tick_id]
            turn = value if isinstance(value, GwtTurnRecord) else GwtTurnRecord.model_validate(value)
        return json.loads(render_context(turn, ContextKind.Facts, schema_path=record_file_schema()).text)

    tools: dict[str, Callable[..., object]] = {"ghost_get": ghost_get, "ghost_search": ghost_search, "ghost_gwt": ghost_gwt}
    if batch is not None:
        def ghost_propose_create(record: RecordFile) -> RecordFile:
            """Validate an external proposal and return its reserved identity."""
            created = batch.create(RecordPayload.model_validate(record).root)
            return record_file_payload(created)

        def ghost_propose_replace(replacement: RecordTextReplacement) -> RecordFile:
            """Copy the source and change only its text. The owner commits or discards it.

            Validate plain workflow calls as well as structured MCP arguments.
            Missing sources and stale heads fail through the batch contract.
            """
            proposal = RecordTextReplacement.model_validate(replacement)
            source = batch.view().get(proposal.id)
            edited = source.copy_for_edit()
            edited.content = proposal.content
            successor = batch.replace(edited)
            return record_file_payload(successor)

        tools["ghost_propose_create"] = ghost_propose_create
        tools["ghost_propose_replace"] = ghost_propose_replace
    if memory is not None:
        tools.update(memory.tools())
    return tools


def build_mcp(environment: MemoryToolEnvironment, batch: WriteBatch | None = None, *,
              memory: MemoryReadSession | None = None) -> FastMCP:
    """Expose these same typed adapters through MCP without starting a listener."""
    server = FastMCP("ghost")
    for name, function in build_tools(environment, batch, memory=memory).items():
        server.tool(name=name)(function)
    return server


__all__ = ["build_tools", "build_mcp"]
