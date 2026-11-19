"""Static contracts for direct model calls and local context measurement."""
from typing import TypedDict
from pydantic import BaseModel, ConfigDict, Field, JsonValue



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
