"""One HTTP path for chat, readiness, and embeddings.

Production waits through transient transport and server errors, with a
30-second pause until cancellation. Tests can select a bounded or fail-fast
policy. Permanent HTTP errors and invalid responses fail explicitly. Programming errors retain their tracebacks.
Neither response bodies nor credentials appear in transport error messages.

Chat responses use a per-prompt response sequence. Seeds are random and do
not affect cache keys. A cache hit does not establish endpoint readiness.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
import time
from pathlib import Path
from threading import Event

import httpx

from pm.llm.config import Connection
from pm.llm.cache import completion_path, publish
from pm.llm.concurrency import inference_admission


def replay_response(path: Path) -> httpx.Response | None:
    """Return the reserved wire response, or None when the file is absent."""
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return None
    return httpx.Response(200, content=content)


def store_response(path: Path, content: bytes) -> None:
    """Keep one exact wire response so a matching completion skips the endpoint.

    A write failure warns on stderr. It cannot change the completion result.
    """
    try:
        publish(path, content)
    except OSError as error:
        print(f"Completion cache write failed: {type(error).__name__}: {error}", file=sys.stderr)


class ModelError(RuntimeError):
    pass


class EmptyCompletionError(ModelError):
    """The provider returned neither usable text nor a tool call."""


class ProviderCompletionError(ModelError):
    """The provider failed or filtered generation. Partial text is not a result."""


def completion_error(data: object) -> ProviderCompletionError | None:
    """Recognize a failed completion inside a successful HTTP response.

    Some providers return partial content with finish_reason=error and HTTP
    200. Do not pass that fragment to a stage parser or retry a content filter.
    An ordinary completed refusal remains a valid model response.
    """
    if not isinstance(data, dict) or not isinstance(data.get("choices"), list):
        return None
    for choice in data["choices"]:
        if isinstance(choice, dict) and (choice.get("error") is not None
                or choice.get("finish_reason") in {"error", "content_filter"}):
            return ProviderCompletionError("The model provider failed or filtered generation. No partial completion was accepted.")
    return None


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
    """Share cancellable transient waits across compatible model endpoints.

    Retries repeat the same HTTP request after a connection failure, HTTP 429
    or HTTP 5xx. A lost response can therefore cause duplicate inference.
    This client does not retry tool execution or accept a partial response.
    """

    def __init__(self, connection: Connection, *, live_test: bool = False,
                 transient_retries: int | None = None, retry_delay_seconds: float = 30):
        self.connection = connection
        self.live_test = live_test
        self.transient_retries = 0 if live_test and transient_retries is None else transient_retries
        if self.transient_retries is not None and self.transient_retries < 0:
            raise ValueError("transient_retries must be nonnegative or None for unlimited waits.")
        self.retry_delay_seconds = retry_delay_seconds
        self.request_count = 0
        self.cache_hits = 0

    async def _send(self, client, method: str, route: str, headers: dict,
                    payload: dict | None, cancel: Event | None, cache_path: Path | None = None):
        """Log each inference attempt before dispatch, including retries.

        The caller reserves the cache position before transport retries.
        A replay writes its receipt but reaches no inference endpoint.
        Cancellation leaves the prompt and an error receipt. Logging failures
        cannot change the HTTP result. Readiness requests have no prompt log.
        """
        from pm.llm.prompt_log import start_prompt_log, finish_prompt_log
        started = time.monotonic()
        replayed = replay_response(cache_path) if cache_path is not None else None
        path = start_prompt_log(connection=self.connection, payload=payload, route=route,
                                request_count=self.request_count) if method == "POST" else None
        if replayed is not None:
            self.cache_hits += 1
            response = replayed
        else:
            try:
                self.request_count += 1
                response = await cancellable(client.request(
                    method, f"{self.connection.base_url}/{route}", headers=headers, json=payload,
                ), cancel)
            except BaseException as error:
                finish_prompt_log(path, error=error, elapsed=time.monotonic() - started)
                raise
            self._store(cache_path, response)
        try:
            data = response.json()
        except ValueError:
            data = None
        error = completion_error(data) if response.is_success else ModelError(f"HTTP {response.status_code}")
        finish_prompt_log(path, response=data, error=error, elapsed=time.monotonic() - started)
        return response

    @staticmethod
    def _store(path: Path | None, response: httpx.Response) -> None:
        """Store a successful chat response. Errors and partial results are not replayable."""
        if path is None or not response.is_success:
            return
        try:
            data = response.json()
        except ValueError:
            return
        if isinstance(data, dict) and "error" not in data and completion_error(data) is None:
            store_response(path, response.content)

    async def request(self, method: str, route: str, *, payload: dict | None = None, cancel: Event | None = None) -> dict:
        """Wait through temporary failures until recovery or cancellation.

        Production waits have no retry quota. An explicit transient_retries
        bound permits finite diagnostics; exhaustion raises ModelError.
        Invalid JSON, provider-filtered output and permanent HTTP errors fail
        directly. Cancellation interrupts in-flight I/O and retry waits.
        """
        headers = {}
        key = self.connection.api_key.get_secret_value()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        cache_path = None
        chat = method == "POST" and route == "chat/completions"
        cached_chat = chat and os.environ.get("PM_COMPLETION_CACHE", "1") != "0"
        if chat:
            payload = {**(payload or {}), "seed": secrets.randbelow(2**31)}
        # Connecting can fail. Once connected, inference has no speed deadline.
        async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0)) as client:
            retries = 0
            while True:
                if cancel is not None and cancel.is_set():
                    raise asyncio.CancelledError
                failure = None
                try:
                    if chat:
                        async with inference_admission(self.connection.base_url).slot(cancel):
                            if cached_chat and cache_path is None:
                                # Queue cancellation must not advance the response counter.
                                cache_path = completion_path(self.connection.base_url, payload or {})
                            response = await self._send(client, method, route, headers, payload, cancel, cache_path)
                    else:
                        response = await self._send(client, method, route, headers, payload, cancel, cache_path)
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
                        provider_error = completion_error(data)
                        if provider_error is not None:
                            raise provider_error
                        return data
                if self.transient_retries is not None and retries >= self.transient_retries:
                    raise ModelError(failure) from None
                retries += 1
                limit = str(self.transient_retries) if self.transient_retries is not None else "until cancellation"
                print(f"{failure} Retrying in {self.retry_delay_seconds:g} seconds "
                      f"({retries}/{limit}).", file=sys.stderr)
                await cancellable(asyncio.sleep(self.retry_delay_seconds), cancel)
