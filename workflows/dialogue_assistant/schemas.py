"""Typed request and report for Assistant dialogue.

This is a separate message contract from Story dialogue. Normal user,
assistant, and tool messages preserve tool-call structure. Tool receipts
record what actually ran; a dialogue claim about a tool is supported by its
receipt, not by the model's memory.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from workflows.public_speech.public_context import PublicContext
from workflows.public_speech.speech_review import SpeechReview


class AssistantMessage(BaseModel):
    """One message in the tool-calling transcript. Tool structure is kept."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[dict] = Field(default_factory=list)
    tool_call_id: str = ""
    name: str = ""
    reasoning: str | None = None
    reasoning_content: str | None = None
    reasoning_details: list[dict] | None = None


class ToolDescriptor(BaseModel):
    """A granted tool. Read-only tools cannot change external or managed state."""
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    parameters: dict = Field(default_factory=dict)
    read_only: bool = False


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
    partner_name: str = Field(min_length=1)
    character_card: str = ""
    # Ability lines: configured, unavailable, and notes, one "- " line each.
    abilities: str = ""
    # The current time and the gap since the last exchange, in words.
    time: str = ""
    # Accepted feelings in words.
    feelings: str = ""
    prior_dialogue: list[AssistantMessage] = Field(default_factory=list)
    current_user_message: str
    memories: list[str] = Field(default_factory=list)
    workspace: str = ""
    tools: list[ToolDescriptor] = Field(default_factory=list)
    tool_grant: list[str] = Field(default_factory=list)
    max_tool_turns: int = 8
    selected_action: str = ""
    public_context: PublicContext | None = None
    minecraft_read: Literal["", "inventory", "state"] = ""


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
    speech_review: SpeechReview | None = None
    review_reason: str = ""


__all__ = [
    "AssistantMessage",
    "AssistantReport",
    "AssistantRequest",
    "ToolDescriptor",
    "ToolReceipt",
]
