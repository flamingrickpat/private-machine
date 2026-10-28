"""Static contracts for direct model calls and local context measurement."""
from collections.abc import Sequence
from enum import StrEnum
from threading import Event
from typing import Protocol, TypedDict
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from pm.llm.prompt_encoding import TurnInput
from pm.llm.completion import TextCompletion as TextCompletion


class ModelTool(StrEnum):
    """Named tool slots supplied to trusted Python workflow roles."""

    Complete = "llm_complete"
    MeasureContext = "llm_measure_context"


class ModelProtocolContext(BaseModel):
    """Wire-level schemas that also occupy model input context."""

    model_config = ConfigDict(extra="ignore")
    tools: list[dict[str, JsonValue]] = Field(default_factory=list[dict[str, JsonValue]])
    response_format: dict[str, JsonValue] | None = None
    chat_template_kwargs: dict[str, JsonValue] = Field(default_factory=dict[str, JsonValue])


class ContextMeasurement(TypedDict):
    """Token planning receipt. Estimates are deliberately separate from provider usage."""

    context_window: int
    context_source: str
    input_tokens: int
    measurement: str
    input_fraction: float
    input_budget_tokens: int
    measurement_complete: bool


class ContextMeasurer(Protocol):
    """Measure the rendered turns using the selected model's configured capacity."""

    def __call__(self, messages: Sequence[TurnInput], /, *, cancel: Event | None = None,
                 **generation: object) -> ContextMeasurement: ...


class WorkflowCompletion(Protocol):
    """A host-bound tool whose closure already owns workflow cancellation."""

    def __call__(self, messages: Sequence[TurnInput], **generation: object) -> str: ...
