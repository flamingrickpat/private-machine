"""Typed request and report for Assistant dialogue.

This is a separate message contract from Story dialogue. Normal user,
assistant, and tool messages preserve tool-call structure. Tool receipts
record what actually ran; a dialogue claim about a tool is supported by its
receipt, not by the model's memory.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AssistantMessage(BaseModel):
    """One message in the tool-calling transcript. Tool structure is kept."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[dict] = Field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""


class ToolDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    parameters: dict = Field(default_factory=dict)


class ToolReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    name: str
    arguments: dict = Field(default_factory=dict)
    executed: bool
    result: Any = None
    error: str = ""
    transport: str = "python"


class AssistantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pm.assistant-request.v1"] = "pm.assistant-request.v1"
    companion_name: str = Field(min_length=1)
    user_name: str = Field(min_length=1)
    character_card: str = ""
    private_ontology: str = ""
    temporal_context: str = ""
    prior_dialogue: list[AssistantMessage] = Field(default_factory=list)
    current_user_message: str = Field(min_length=1)
    memories: list[str] = Field(default_factory=list)
    workspace: str = ""
    tools: list[ToolDescriptor] = Field(default_factory=list)
    tool_grant: list[str] = Field(default_factory=list)
    max_tool_turns: int = 6


class AssistantReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pm.assistant-report.v1"] = "pm.assistant-report.v1"
    companion_name: str
    message_roles: list[str]
    messages: list[AssistantMessage]
    final_text: str
    tool_receipts: list[ToolReceipt]
    accepted: bool
    rejection_reason: str = ""


__all__ = [
    "AssistantMessage",
    "AssistantReport",
    "AssistantRequest",
    "ToolDescriptor",
    "ToolReceipt",
]
