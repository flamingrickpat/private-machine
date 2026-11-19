from __future__ import annotations

import json
from pm.context_render.budget import TokenEstimator
from pm.context_render.prompt import RenderSettings
from pm.context_render.turns import Role, Turn
from pm.llm.generation import GenerationSettings


def protocol_turns(settings: GenerationSettings) -> tuple[Turn, ...]:
    sections: list[str] = []
    for tool in settings.tools:
        sections.append(json.dumps(tool.model_dump(mode="json", by_alias=True, exclude_none=True), ensure_ascii=False))
    if settings.response_format is not None:
        sections.append(json.dumps(settings.response_format.model_dump(mode="json", by_alias=True, exclude_none=True), ensure_ascii=False))
    if settings.chat_template_kwargs is not None:
        sections.append(json.dumps(settings.chat_template_kwargs.model_dump(mode="json", by_alias=True, exclude_none=True), ensure_ascii=False))
    return (Turn(Role.System, "\n".join(sections)),) if sections else ()


def protocol_token_allowance(settings: GenerationSettings, estimator: TokenEstimator) -> int:
    turns = protocol_turns(settings)
    if not turns:
        return 0
    # Separate rounded estimates can differ from the combined estimate by one token.
    return estimator.estimate(turns) - estimator.estimate(()) + 1


def reserve_protocol(settings: RenderSettings, generation: GenerationSettings) -> RenderSettings:
    allowance = protocol_token_allowance(generation, TokenEstimator(estimate_factor=settings.estimate_factor))
    return RenderSettings(context_window=settings.context_window, input_fraction=settings.input_fraction,
        reserved_output_tokens=settings.reserved_output_tokens, protocol_tokens=settings.protocol_tokens + allowance,
        correction_tokens=settings.correction_tokens,
        estimate_factor=settings.estimate_factor)
