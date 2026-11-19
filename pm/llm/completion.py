"""Typed completion tools for synchronous and asynchronous workflow roles."""
from __future__ import annotations

from collections.abc import Sequence
from threading import Event
from typing import Protocol
from dataclasses import dataclass, field
from pydantic import BaseModel, ConfigDict

from pm.llm.prompt_encoding import TurnInput
from pm.llm.generation import GenerationSettings
from pm.context_render.turns import Turn

@dataclass(frozen=True)
class Reasoning:
    """Provider reasoning kept apart from final assistant content.

    Opaque details can carry provider signatures needed for later tool turns.
    They are not evidence, tool arguments, or a public reply.
    """

    text: str = ""
    details_json: str | None = None


class ProviderTokenUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


@dataclass(frozen=True)
class Completion:
    """A neutral model result with final content, reasoning, and usage."""

    turn: Turn
    reasoning: Reasoning = field(default_factory=Reasoning)
    usage: ProviderTokenUsage = field(default_factory=ProviderTokenUsage)
    model: str = ""
    finish_reason: str | None = None

    @property
    def text(self) -> str:
        """Return final text; never substitute private reasoning."""
        return self.turn.content.text


class CompletionBackend(Protocol):
    """A backend consumes neutral turns and returns a neutral completion."""

    async def complete(self, turns: Sequence[TurnInput], *, cancel: Event | None = None,
                       settings: GenerationSettings = GenerationSettings()) -> Completion: ...
