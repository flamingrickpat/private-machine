"""Tool descriptors the Assistant model is offered.

The names match the functions supplied by the module tools factory and
registered on the scoped MCP servers. Ghost read tools use the same scoped
view as every other run.
"""
from __future__ import annotations

from workflows.dialogue_assistant.schemas import ToolDescriptor

TOOL_DESCRIPTORS: list[ToolDescriptor] = [
    ToolDescriptor(
        name="ghost_get",
        description="Read one Ghost knoxel by id from the scoped view.",
        parameters={
            "type": "object",
            "properties": {"knoxel_id": {"type": "integer"}},
            "required": ["knoxel_id"],
        },
    ),
    ToolDescriptor(
        name="ghost_search",
        description="Search the scoped Ghost view for text.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    ),
    ToolDescriptor(
        name="ghost_gwt",
        description="Read one committed GWT turn by tick id.",
        parameters={
            "type": "object",
            "properties": {"tick_id": {"type": "integer"}},
            "required": ["tick_id"],
        },
    ),
    ToolDescriptor(
        name="managed_start",
        description="Start one managed workflow run. A repeated action id is idempotent.",
        parameters={
            "type": "object",
            "properties": {
                "action_id": {"type": "string"},
                "workflow_name": {"type": "string"},
                "instructions": {"type": "string"},
                "parent_correlation": {"type": "string"},
                "artifact_path": {"type": "string"},
                "arguments": {"type": "object"},
            },
            "required": ["action_id", "workflow_name"],
        },
    ),
    ToolDescriptor(
        name="managed_status",
        description="Read the durable receipt for one managed run.",
        parameters={
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
    ),
    ToolDescriptor(
        name="managed_list",
        description="List every receipt in the managed-run table.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDescriptor(
        name="managed_update",
        description="Queue one instruction revision for the next safe boundary.",
        parameters={
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "instructions": {"type": "string"},
                "request_id": {"type": "string"},
            },
            "required": ["run_id", "instructions"],
        },
    ),
    ToolDescriptor(
        name="managed_terminate",
        description="Set the code-owned cancellation signal for one managed run.",
        parameters={
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
    ),
    ToolDescriptor(
        name="managed_answer",
        description="Answer a waiting run's outstanding question and resume it.",
        parameters={
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "question_id": {"type": "string"},
                "answer": {"type": "string"},
            },
            "required": ["run_id", "answer"],
        },
    ),
]

TOOL_NAMES: list[str] = [tool.name for tool in TOOL_DESCRIPTORS]

__all__ = ["TOOL_DESCRIPTORS", "TOOL_NAMES"]
