"""Measure direct chat inputs against the selected serving context."""
from __future__ import annotations

import asyncio
from threading import Event
from urllib.parse import urlsplit, urlunsplit
from collections.abc import Sequence

import httpx
import tiktoken

from pm.llm.http_client import ModelError, cancellable, wait_until_ready
from pm.llm.config import Connection
from pm.llm.workflow_sessions import serving_context
from pm.llm.prompt_encoding import TurnInput, normalize_turns
from pm.llm.multimodal import encode_messages, prompt_image_count
from pm.constants import PROMPT_IMAGE_TOKENS
from pm.context_render.turns import Text


async def measure_context(connection: Connection, messages: Sequence[TurnInput], cancel: Event, *,
                          live_test: bool = False, **generation) -> dict:
    """Count templated tokens and reserve a configured fraction for output.

    Compatible endpoints supply their actual chat template and tokenizer.
    Other endpoints use an explicitly labelled tiktoken estimate and require
    configured serving capacity. Cancellation closes every discovery request.
    Transport and invalid protocol errors raise ModelError without prompt text.
    A silent endpoint is waited out: the measurement request repeats every 30
    seconds until the endpoint answers. ``live_test=True`` keeps the fail-fast
    error for diagnostics and tests.

    ``measurement_complete`` says whether the count is exact. It is False when
    the prompt holds media, because no route counts image tiles before
    inference. Callers treat that count as a planning estimate, not as a
    reason to refuse the call. See
    :meth:`pm.llm.structured._Repair.check_budget`.
    """
    if cancel.is_set():
        raise asyncio.CancelledError
    context, source = await serving_context(connection, cancel, live_test=live_test)
    has_media = prompt_image_count(messages) > 0 or any(
        any(not isinstance(part, Text) for part in turn.content.parts) for turn in normalize_turns(messages))
    parts = urlsplit(connection.base_url)
    root = urlunsplit((parts.scheme, parts.netloc, parts.path.removesuffix("/v1"), "", ""))
    key = connection.api_key.get_secret_value()
    headers = {"Authorization": "Bearer " + key} if key else {}
    payload = {**encode_messages(messages), "add_generation_prompt": True}
    from pm.llm.activation import provider_generation
    generation = provider_generation(connection, generation)
    for field in ("chat_template_kwargs", "tools"):
        if field in generation:
            payload[field] = generation[field]
    async def count_with_server() -> tuple[int, str]:
        """Ask the endpoint for its own template and token count. Retried until it answers."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                template = await cancellable(client.post(root + "/apply-template", headers=headers, json=payload), cancel)
                if template.status_code in {404, 405, 501}:
                    return estimate_tokens(connection.model, messages)
                template.raise_for_status()
                prompt = template.json()["prompt"]
                if not isinstance(prompt, str):
                    raise ValueError("Invalid chat template.")
                response = await cancellable(client.post(root + "/tokenize", headers=headers,
                    json={"content": prompt, "add_special": False, "parse_special": True}), cancel)
                response.raise_for_status()
                tokens = response.json()["tokens"]
                if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
                    raise ValueError("Invalid token sequence.")
                return len(tokens), "server.chat_template_and_tokenizer"
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            raise ModelError("Context measurement failed. Check the selected model endpoint and tokenizer.") from None

    if parts.hostname == "openrouter.ai" or has_media:
        # OpenRouter has no llama.cpp template or tokenizer routes. Provider
        # usage after inference is authoritative. This count is an estimate.
        count, measurement = estimate_tokens(connection.model, messages)
    else:
        count, measurement = await wait_until_ready(count_with_server, cancel, live_test=live_test,
                                                    label=f"Token measurement at {connection.base_url}")
    return {"context_window": context, "context_source": source, "input_tokens": count,
            "measurement": measurement, "input_fraction": connection.context_input_fraction,
            "input_budget_tokens": int(context * connection.context_input_fraction),
            "measurement_complete": not has_media}


def estimate_tokens(model: str, messages: Sequence[TurnInput]) -> tuple[int, str]:
    """Label the fallback estimate when the provider has no tokenizer route.

    The provider remains authoritative. Chat framing adds an estimated token
    allowance; the configured response reserve does not cap generation.
    """
    from pm.llm.token_estimate import (
        MESSAGES_BASE_OVERHEAD, PROMPT_IMAGE_TOKENS, TURN_OVERHEAD, text_tokens,
    )
    turns = normalize_turns(messages)
    images = prompt_image_count(turns)
    media = any(any(not isinstance(part, Text) for part in turn.content.parts) for turn in turns)
    # Each image part (a resolved tag or a direct image) adds a fixed planning
    # estimate. Image tags outside the last-N rule are text and count as text.
    count = MESSAGES_BASE_OVERHEAD + images * PROMPT_IMAGE_TOKENS
    for turn in turns:
        values = [turn.role.value, *turn.content.notes, *(part.text for part in turn.content.parts if isinstance(part, Text))]
        values.extend(str(value) for value in (turn.tool_calls, turn.tool_call_id, turn.name) if value)
        count += TURN_OVERHEAD + sum(text_tokens(str(v)) for v in values)
    # Binary size cannot establish image or audio token usage. Images use the
    # constant estimate above; other media stay unmeasured and are labelled.
    label = ("estimated.text_only.media_unmeasured." if media and not images else
             "estimated.images_by_constant." if images else "estimated.tiktoken.")
    return count, label + "o200k_base"
