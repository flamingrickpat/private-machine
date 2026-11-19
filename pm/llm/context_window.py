"""Measure direct chat inputs against the selected serving context."""
from __future__ import annotations

import asyncio
from threading import Event
from collections.abc import Sequence


from pm.llm.config import Connection
from pm.llm.workflow_sessions import serving_context
from pm.llm.prompt_encoding import TurnInput
from pm.llm.contracts import ContextMeasurement
from pm.llm.generation import GenerationSettings
from pm.context_render import Turn
from pm.llm.protocol_context import protocol_turns


def measured_turns(messages: Sequence[TurnInput]) -> list[Turn]:
    """Return the turns exactly as the provider will receive them.

    An image tag is text until it resolves to an image part, and an image part
    costs :data:`pm.constants.PROMPT_IMAGE_TOKENS` rather than its tag's few
    words. Measuring the unresolved text therefore under-counts every prompt
    that carries pictures, while still claiming an image allowance in its
    receipt. Resolving first keeps the estimate and the request the same shape.
    """
    from pm.llm.prompt_encoding import normalize_turns
    return normalize_turns(messages)


async def measure_context(connection: Connection, messages: Sequence[TurnInput], cancel: Event, *,
                          live_test: bool = False, settings: GenerationSettings = GenerationSettings()) -> ContextMeasurement:
    """Estimate rendered turns locally and reserve the configured output fraction.

    Capacity discovery uses the selected serving configuration. Token counting
    uses both local tiktoken encodings and a safety margin. It does not ask the
    provider to tokenize the prompt. Cancellation stops capacity discovery.
    """
    if cancel.is_set():
        raise asyncio.CancelledError
    from pm.context_render.budget import ContextBudget
    context, source = await serving_context(connection, cancel, live_test=live_test)
    output_tokens = settings.output_allowance(connection.completion_token_limit)
    input_ceiling = min(int(context * connection.context_input_fraction), context - output_tokens)
    budget = ContextBudget(context_window=context, input_fraction=connection.context_input_fraction)
    turns = measured_turns(messages)
    turns.extend(protocol_turns(settings))
    count = budget.estimate(turns)
    return {"context_window": context, "context_source": source, "input_tokens": count,
            "measurement": budget.measurement_label(turns),
            "input_fraction": budget.input_fraction, "input_budget_tokens": input_ceiling,
            "measurement_complete": True}


def estimate_tokens(model: str, messages: Sequence[TurnInput]) -> tuple[int, str]:
    """Keep the public estimate API. Count locally with a deliberate safety margin."""
    from pm.context_render.budget import TokenEstimator
    estimator = TokenEstimator()
    turns = measured_turns(messages)
    return estimator.estimate(turns), estimator.measurement_label(turns)
