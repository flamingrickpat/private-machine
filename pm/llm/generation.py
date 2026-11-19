from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from pm.llm.reasoning import ReasoningBudget, ReasoningOptions


class ToolFunctionSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    description: str = ""
    parameters: dict[str, JsonValue]
    strict: bool | None = None


class ToolSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["function"] = "function"
    function: ToolFunctionSchema


class ResponseSchema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)
    name: str
    strict: bool = True
    schema_body: dict[str, JsonValue] = Field(validation_alias="schema", serialization_alias="schema")


class ResponseFormat(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["json_object", "json_schema", "text"]
    json_schema: ResponseSchema | None = None


class ChatTemplateOptions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    enable_thinking: bool | None = None


class ProviderRouting(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    order: tuple[str, ...] = ()
    require_parameters: bool | None = None
    allow_fallbacks: bool | None = None


class GenerationSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    enable_thinking: bool | None = None
    reasoning: ReasoningOptions | None = None
    reasoning_budget: ReasoningBudget | None = None
    reasoning_tokens: int | None = None
    reasoning_budget_tokens: int | None = None
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None
    tools: tuple[ToolSchema, ...] = ()
    response_format: ResponseFormat | None = None
    chat_template_kwargs: ChatTemplateOptions | None = None
    provider: ProviderRouting | None = None
    stop: tuple[str, ...] | None = None
    tool_choice: Literal["auto", "none", "required"] | None = None

    def output_allowance(self, default_tokens: int | None) -> int:
        tokens = self.max_tokens or self.max_completion_tokens or default_tokens
        if tokens is None:
            raise ValueError("A model request requires an explicit output allowance.")
        return tokens

    def with_tools(self, tools: tuple[ToolSchema, ...]) -> GenerationSettings:
        return GenerationSettings(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
            max_completion_tokens=self.max_completion_tokens,
            enable_thinking=self.enable_thinking,
            reasoning=self.reasoning,
            reasoning_budget=self.reasoning_budget,
            reasoning_tokens=self.reasoning_tokens,
            reasoning_budget_tokens=self.reasoning_budget_tokens,
            reasoning_effort=self.reasoning_effort,
            tools=tools,
            response_format=self.response_format,
            chat_template_kwargs=self.chat_template_kwargs,
            provider=self.provider,
            stop=self.stop,
            tool_choice=self.tool_choice,
        )
