"""Measure direct chat inputs against the selected serving context."""
from __future__ import annotations

import asyncio
from threading import Event
from collections.abc import Sequence


from pm.llm.config import Connection
from pm.llm.workflow_sessions import serving_context
from pm.llm.prompt_encoding import TurnInput
from pm.llm.contracts import ContextMeasurement, ModelProtocolContext
from pm.context_render import Role, Turn, render_context


def measured_turns(messages: Sequence[TurnInput]) -> list[Turn]:
    """Return the turns exactly as the provider will receive them.

    An image tag is text until it resolves to an image part, and an image part
    costs :data:`pm.constants.PROMPT_IMAGE_TOKENS` rather than its tag's few
    words. Measuring the unresolved text therefore under-counts every prompt
    that carries pictures, while still claiming an image allowance in its
    receipt. Resolving first keeps the estimate and the request the same shape.
    """
    from pm.llm.multimodal import resolve_turns
    return resolve_turns(messages)


async def measure_context(connection: Connection, messages: Sequence[TurnInput], cancel: Event, *,
                          live_test: bool = False, protocol_context: ModelProtocolContext | None = None,
                          **generation: object) -> ContextMeasurement:
    """Estimate rendered turns locally and reserve the configured output fraction.

    Capacity discovery uses the selected serving configuration. Token counting
    uses both local tiktoken encodings and a safety margin. It does not ask the
    provider to tokenize the prompt. Cancellation stops capacity discovery.
    """
    if cancel.is_set():
        raise asyncio.CancelledError
    from pm.context_render.budget import ContextBudget
    context, source = await serving_context(connection, cancel, live_test=live_test)
    budget = ContextBudget(context_window=context, input_fraction=min(0.8, connection.context_input_fraction))
    protocol = protocol_context if protocol_context is not None else ModelProtocolContext.model_validate(generation)
    turns = measured_turns(messages)
    if protocol.tools or protocol.response_format is not None or protocol.chat_template_kwargs:
        turns.append(Turn(Role.System, render_context(protocol.model_dump(exclude_defaults=True))))
    count = budget.estimate(turns)
    return {"context_window": context, "context_source": source, "input_tokens": count,
            "measurement": budget.measurement_label(turns),
            "input_fraction": budget.input_fraction, "input_budget_tokens": budget.input_ceiling,
            "measurement_complete": False}


def estimate_tokens(model: str, messages: Sequence[TurnInput]) -> tuple[int, str]:
    """Keep the public estimate API. Count locally with a deliberate safety margin."""
    from pm.context_render.budget import TokenEstimator
    estimator = TokenEstimator()
    turns = measured_turns(messages)
    return estimator.estimate(turns), estimator.measurement_label(turns)
