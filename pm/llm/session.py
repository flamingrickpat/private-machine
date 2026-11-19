from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from threading import Event
from collections.abc import Sequence
from uuid import uuid4

from pm.context_render.prompt import RenderSettings, RendererCaptureSettings
from pm.llm.completion import Completion
from pm.llm.generation import GenerationSettings, ToolSchema
from pm.llm.prompt_encoding import TurnInput
from pm.llm.stage_model import LazyStageModel


@dataclass(slots=True)
class ModelSession:
    model: LazyStageModel
    cancel: Event | None = None
    yield_requested: Event | None = None
    _settings: RenderSettings | None = field(default=None, init=False)

    def render_settings(self, *, max_tokens: int | None = None) -> RenderSettings:
        from pm.llm.workflow_sessions import serving_context
        connection = self.model.connection
        output_tokens = max_tokens if max_tokens is not None else connection.completion_token_limit
        if output_tokens is None:
            raise ValueError("Prompt rendering requires an explicit output token allowance.")
        if self._settings is None:
            context_window, _source = asyncio.run(serving_context(
                connection, self.cancel if self.cancel is not None else Event(), live_test=self.model.live_test))
            self._settings = RenderSettings(context_window=context_window,
                input_fraction=connection.context_input_fraction, reserved_output_tokens=output_tokens)
        return RenderSettings(context_window=self._settings.context_window, input_fraction=self._settings.input_fraction,
            reserved_output_tokens=output_tokens, protocol_tokens=self._settings.protocol_tokens,
            correction_tokens=self._settings.correction_tokens,
            estimate_factor=self._settings.estimate_factor)

    def renderer_settings_for(self, generation: GenerationSettings) -> RenderSettings:
        from pm.llm.protocol_context import reserve_protocol
        resolved = self.render_settings(max_tokens=generation.output_allowance(self.model.connection.completion_token_limit))
        return reserve_protocol(resolved, generation)

    def resolved_renderer_settings_for(self, generation: GenerationSettings) -> RenderSettings:
        from pm.llm.protocol_context import reserve_protocol
        resolved = self.settings
        output_tokens = generation.output_allowance(self.model.connection.completion_token_limit)
        settings = RenderSettings(context_window=resolved.context_window, input_fraction=resolved.input_fraction,
            reserved_output_tokens=output_tokens, protocol_tokens=resolved.protocol_tokens,
            correction_tokens=resolved.correction_tokens,
            estimate_factor=resolved.estimate_factor)
        return reserve_protocol(settings, generation)

    @property
    def settings(self) -> RenderSettings:
        if self._settings is None:
            raise RuntimeError("Resolve renderer settings on the owning thread before entering async execution.")
        return self._settings

    def complete(self, turns: Sequence[TurnInput], settings: GenerationSettings = GenerationSettings()) -> str:
        self.check_cancelled()
        result = self.model.complete(turns, settings=settings, cancel=self.cancel)
        self.check_cancelled()
        return result

    async def complete_async(self, turns: Sequence[TurnInput], settings: GenerationSettings = GenerationSettings()) -> str:
        self.check_cancelled()
        result = await self.model.complete_chat_async(turns, settings=settings, cancel=self.cancel)
        self.check_cancelled()
        return result.text

    async def complete_tools_async(self, turns: Sequence[TurnInput], *, tools: tuple[ToolSchema, ...] = (),
                                   settings: GenerationSettings = GenerationSettings()) -> Completion:
        self.check_cancelled()
        if tools != settings.tools:
            settings = GenerationSettings.with_tools(settings, tools)
        result = await self.model.complete_chat_async(turns, settings=settings, cancel=self.cancel)
        self.check_cancelled()
        return result.completion()

    def capture_settings(self, operation: str) -> RendererCaptureSettings:
        options = self.model.capture_options
        return RendererCaptureSettings(options.directory / operation / uuid4().hex,
                                       options.capture_input, options.capture_output)

    def check_cancelled(self) -> None:
        if self.yield_requested is not None and self.yield_requested.is_set():
            if self.cancel is not None:
                self.cancel.set()
            raise asyncio.CancelledError
        if self.cancel is not None and self.cancel.is_set():
            raise asyncio.CancelledError
