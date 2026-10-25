"""Shared token estimator for planning and measurement.

Wraps tiktoken with image counting and per-record overhead. All token
budgets in the context planner, temporal solver, and context window
measurement use this module so counts stay consistent.

Image tokens use the constant from pm.constants.PROMPT_IMAGE_TOKENS.

Plain text counting belongs to ``pm.context_render.tokens``, because the
context budgets and the memory inspector count story text with it. This module
adds the chat framing on top of that one counter.
"""
from __future__ import annotations

from pm.constants import PROMPT_IMAGE_TOKENS
# One text counter for the whole runtime. pm.llm may import pm.context_render.
from pm.context_render.tokens import text_tokens

#: Per-record framing overhead in tokens (role tag, separators, name).
RECORD_OVERHEAD = 12

#: Per-turn framing overhead in tokens (role, separators, generation prompt).
TURN_OVERHEAD = 12

#: Base overhead for a full message list (system framing, generation prompt).
MESSAGES_BASE_OVERHEAD = 24


def image_tokens(count: int = 1) -> int:
    """Return the estimated token cost for ``count`` images."""
    return count * PROMPT_IMAGE_TOKENS


def record_tokens(text: str) -> int:
    """Return the planning token cost of one record (text + framing)."""
    return RECORD_OVERHEAD + text_tokens(text)


def turn_tokens(role: str, content: str, images: int = 0) -> int:
    """Return the estimated token cost of one chat turn."""
    return TURN_OVERHEAD + text_tokens(role) + text_tokens(content) + image_tokens(images)


def messages_tokens(messages: list[dict], images: int = 0) -> int:
    """Return the estimated token cost of a full message list.

    Each message dict needs "role" and "content" keys. ``images`` is the
    total image count across all messages.
    """
    count = MESSAGES_BASE_OVERHEAD + image_tokens(images)
    for msg in messages:
        count += turn_tokens(msg.get("role", ""), msg.get("content", ""))
    return count


__all__ = [
    "MESSAGES_BASE_OVERHEAD",
    "PROMPT_IMAGE_TOKENS",
    "RECORD_OVERHEAD",
    "TURN_OVERHEAD",
    "image_tokens",
    "messages_tokens",
    "record_tokens",
    "text_tokens",
    "turn_tokens",
]
