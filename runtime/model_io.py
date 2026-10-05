"""One HTTP path for chat, readiness, and embeddings.

Production retries transient transport and server errors until cancellation.
Live tests report these errors immediately. Permanent HTTP errors and invalid
responses fail explicitly. Programming errors retain their tracebacks.
Neither response bodies nor credentials appear in transport error messages.
"""
from __future__ import annotations

import asyncio
import sys
from threading import Event

import httpx

from runtime.models import Connection


class ModelError(RuntimeError):
    pass


def rejected_message_protocol(response: httpx.Response) -> bool:
    """Identify llama.cpp's deterministic parser rejection without exposing text.

    A malformed tool transcript cannot recover through transport retries.
    Other server errors retain the existing transient retry behavior.
    """
    if response.status_code < 500:
        return False
    try:
        error = response.json().get("error", {})
    except (ValueError, AttributeError):
        return False
    return isinstance(error, dict) and str(error.get("message", "")).startswith("Failed to parse messages:")


async def cancellable(awaitable, cancel: Event | None):
    """Cancel in-flight I/O without a generation deadline or worker thread."""
    if cancel is None:
        return await awaitable
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            if cancel.is_set():
                raise asyncio.CancelledError
            if task.done():
                return task.result()
            await asyncio.wait({task}, timeout=0.1)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


class JsonClient:
    def __init__(self, connection: Connection, *, live_test: bool = False):
        self.connection = connection
        self.live_test = live_test
        self.request_count = 0

    async def request(self, method: str, route: str, *, payload: dict | None = None, cancel: Event | None = None) -> dict:
        headers = {}
        key = self.connection.api_key.get_secret_value()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        # Connecting can fail. Once connected, inference has no speed deadline.
        async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0)) as client:
            while True:
                if cancel is not None and cancel.is_set():
                    raise asyncio.CancelledError
                self.request_count += 1
                failure = None
                try:
                    response = await cancellable(client.request(
                        method, f"{self.connection.base_url}/{route}", headers=headers, json=payload,
                    ), cancel)
                except httpx.TransportError:
                    failure = "Model connection failed. Check the endpoint."
                else:
                    if rejected_message_protocol(response):
                        raise ModelError("The model rejected the message protocol. Inspect the tool-call transcript.")
                    if response.status_code == 429 or response.status_code >= 500:
                        failure = f"Model service returned HTTP {response.status_code}."
                    elif not response.is_success:
                        raise ModelError(f"Model request failed (HTTP {response.status_code}). Check credentials, credit, and request config.")
                    else:
                        try:
                            data = response.json()
                        except ValueError:
                            raise ModelError("The model service returned invalid JSON.") from None
                        if not isinstance(data, dict) or "error" in data:
                            raise ModelError("The model service returned an invalid result or an API error.")
                        return data
                if self.live_test:
                    raise ModelError(failure) from None
                print(f"{failure} Retrying until cancellation.", file=sys.stderr)
                await cancellable(asyncio.sleep(1), cancel)
