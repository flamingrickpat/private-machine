"""Typed completion tools for synchronous and asynchronous workflow roles."""
from __future__ import annotations

from collections.abc import Sequence
from threading import Event
from typing import Protocol
from dataclasses import dataclass, field

from pm.llm.prompt_encoding import TurnInput
from pm.context_render.turns import Turn

@dataclass(frozen=True)
class Reasoning:
    """Provider reasoning kept apart from final assistant content.

    Opaque details can carry provider signatures needed for later tool turns.
    They are not evidence, tool arguments, or a public reply.
    """

    text: str = ""
    details: tuple[dict, ...] = ()


@dataclass(frozen=True)
class Completion:
    """A neutral model result with final content, reasoning, and usage."""

    turn: Turn
    reasoning: Reasoning = field(default_factory=Reasoning)
    usage: dict = field(default_factory=dict)
    model: str = ""
    finish_reason: str | None = None

    @property
    def text(self) -> str:
        """Return final text; never substitute private reasoning."""
        return self.turn.content.text


class TextCompletion(Protocol):
    """Complete turns and return final text. Reasoning is not final text."""

    def __call__(self, turns: Sequence[TurnInput], /, *, cancel: Event | None = None, **generation: object) -> str: ...


class AsyncTextCompletion(Protocol):
    """Complete turns asynchronously and return final text."""

    async def __call__(self, turns: Sequence[TurnInput], /, *, cancel: Event | None = None, **generation: object) -> str: ...


class ToolCompletion(Protocol):
    """Return a typed provider result with tool calls and separate reasoning."""

    async def __call__(self, turns: Sequence[TurnInput], /, *, cancel: Event | None = None,
                       tools: list[dict] | None = None, **generation: object) -> Completion: ...


class CompletionBackend(Protocol):
    """A backend consumes neutral turns and returns a neutral completion."""

    async def complete(self, turns: Sequence[TurnInput], *, cancel: Event | None = None,
                       **generation: object) -> Completion: ...
