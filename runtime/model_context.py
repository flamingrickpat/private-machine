"""Measure direct chat inputs against the selected serving context."""
from __future__ import annotations

from threading import Event
from urllib.parse import urlsplit, urlunsplit

import httpx
import tiktoken

from runtime.model_io import ModelError, cancellable
from runtime.models import Connection
from runtime.workflow_models import serving_context


async def measure_context(connection: Connection, messages: list[dict], cancel: Event, **generation) -> dict:
    """Count templated tokens and reserve a configured fraction for output.

    Compatible endpoints supply their actual chat template and tokenizer.
    Other endpoints use an explicitly labelled tiktoken estimate and require
    configured serving capacity. Cancellation closes every discovery request.
    Transport and invalid protocol errors raise ModelError without prompt text.
    """
    context, source = await serving_context(connection, cancel)
    parts = urlsplit(connection.base_url)
    root = urlunsplit((parts.scheme, parts.netloc, parts.path.removesuffix("/v1"), "", ""))
    key = connection.api_key.get_secret_value()
    headers = {"Authorization": "Bearer " + key} if key else {}
    payload = {"messages": messages, "add_generation_prompt": True}
    for field in ("chat_template_kwargs", "tools"):
        if field in generation:
            payload[field] = generation[field]
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            template = await cancellable(client.post(root + "/apply-template", headers=headers, json=payload), cancel)
            if template.status_code in {404, 405, 501}:
                count, measurement = estimate_tokens(connection.model, messages)
            else:
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
                count, measurement = len(tokens), "server.chat_template_and_tokenizer"
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        raise ModelError("Context measurement failed. Check the selected model endpoint and tokenizer.") from None
    return {"context_window": context, "context_source": source, "input_tokens": count,
            "measurement": measurement, "input_fraction": connection.context_input_fraction,
            "input_budget_tokens": int(context * connection.context_input_fraction)}


def estimate_tokens(model: str, messages: list[dict]) -> tuple[int, str]:
    """Label the fallback estimate when the provider has no tokenizer route.

    The provider remains authoritative. Chat framing adds an estimated token
    allowance; the configured response reserve does not cap generation.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")
    count = 24 + sum(12 + sum(len(encoding.encode(str(value), disallowed_special=()))
                             for value in message.values()) for message in messages)
    return count, "estimated.tiktoken." + encoding.name
