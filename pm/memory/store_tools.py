from __future__ import annotations

from fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field
from pm.context_render.memory_tool import render_memory_record, render_workspace_turn
from pm.memory.bash_view import MemoryBashEnvironment
from pm.memory.agent_tools import MemoryReadSession
from pm.memory.write_batch import WriteBatch
from pm.memory.yaml_codec import RecordPayload
from pm.records.base import RecordBase


class RecordTextReplacement(BaseModel):
    """A model may change content. The host owns every other record field."""

    model_config = ConfigDict(extra="forbid", strict=True)
    id: int = Field(description="The existing knoxel ID to supersede.")
    content: str = Field(description="The complete replacement text, without changing other fields.")


class MemoryTools:
    def __init__(self, environment: MemoryBashEnvironment, batch: WriteBatch | None = None) -> None:
        self.environment = environment
        self.batch = batch

    def source(self, record_id: int) -> RecordBase:
        if self.batch is not None:
            return self.batch.view().get(record_id)
        if record_id in self.environment.records:
            return self.environment.records[record_id]
        assert self.environment.ghost_view is not None
        return self.environment.ghost_view.get(record_id)

    def ghost_get(self, knoxel_id: int) -> str:
        return render_memory_record(self.source(knoxel_id))

    def ghost_search(self, query: str) -> list[str]:
        needle = query.casefold()
        records = self.batch.view().all_knoxels if self.batch is not None else self.environment.records
        matches: list[str] = []
        for record_id in sorted(records, reverse=True):
            record = records[record_id]
            if needle in record.content.casefold():
                text = record.content[:400]
                suffix = " [excerpt; fetch source for complete evidence]" if len(record.content) > 400 else ""
                matches.append(f"Source {record_id} ({record.type.value}): {text}{suffix}")
                if len(matches) == 8:
                    break
        return matches

    def ghost_gwt(self, tick_id: int) -> str:
        turn = self.batch.view().gwt_turns[tick_id] if self.batch is not None else self.environment.turns[tick_id]
        return render_workspace_turn(turn)

    def ghost_propose_create(self, record: RecordPayload) -> str:
        assert self.batch is not None
        return render_memory_record(self.batch.create(record.root))

    def ghost_propose_replace(self, replacement: RecordTextReplacement) -> str:
        assert self.batch is not None
        source = self.batch.view().get(replacement.id)
        edited = source.copy_for_edit()
        edited.content = replacement.content
        return render_memory_record(self.batch.replace(edited))


def build_mcp(environment: MemoryBashEnvironment, batch: WriteBatch | None = None, *,
              memory: MemoryReadSession | None = None) -> FastMCP:
    tools = MemoryTools(environment, batch)
    server = FastMCP("ghost")
    server.tool(name="ghost_get")(tools.ghost_get)
    server.tool(name="ghost_search")(tools.ghost_search)
    server.tool(name="ghost_gwt")(tools.ghost_gwt)
    if batch is not None:
        server.tool(name="ghost_propose_create")(tools.ghost_propose_create)
        server.tool(name="ghost_propose_replace")(tools.ghost_propose_replace)
    if memory is not None:
        server.tool(name="memory_view")(memory.view)
        server.tool(name="memory_search")(memory.search)
        server.tool(name="memory_get")(memory.get)
        server.tool(name="memory_sources")(memory.sources)
        if memory.config.graph:
            server.tool(name="memory_neighbors")(memory.neighbors)
    return server
