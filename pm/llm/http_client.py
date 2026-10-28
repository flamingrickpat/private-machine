"""One HTTP path for chat, readiness, and embeddings.

Production waits through transient transport and server errors, with a
30-second pause until cancellation. Tests can select a bounded or fail-fast
policy. Permanent HTTP errors and invalid responses fail explicitly. Programming errors retain their tracebacks.
Neither response bodies nor credentials appear in transport error messages.

Chat responses use a per-prompt response sequence. Seeds are random and do
not affect cache keys. A cache hit does not establish endpoint readiness.

:func:`wait_until_ready` extends the same endless wait to the readiness and
tokenizer probes that do not go through :class:`JsonClient`.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
import time
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from threading import Event
from typing import TypeVar

import httpx

from pm.llm.config import Connection
from pm.llm.cache import completion_path, publish
from pm.llm.concurrency import inference_admission

#: A value returned by one awaited endpoint operation.
T = TypeVar("T")


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


async def cancellable(awaitable: Awaitable[T], cancel: Event | None) -> T:
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


async def wait_until_ready(step: Callable[[], Awaitable[T]], cancel: Event | None = None, *,
                           live_test: bool = False, delay_seconds: float = 30,
                           label: str = "Model endpoint") -> T:
    """Call ``step`` again and again until it succeeds. This wait has no limit.

    ``step`` is an async callable that talks to a model endpoint. A closed
    port, a model that is still loading, and a broken endpoint answer all mean
    "not ready yet". Print one line, sleep, and call ``step`` again. The
    program never fails here. The operator can restart the server and the
    paused call continues on its own, so no work is lost.

    ``cancel`` ends the wait with ``asyncio.CancelledError``.
    ``live_test=True`` raises the first error instead of waiting. Tests,
    startup probes, and one-shot diagnostics use that mode. Any other
    exception type is a code fault and propagates unchanged.
    """
    attempts = 0
    while True:
        try:
            return await step()
        except (ModelError, httpx.HTTPError) as error:
            if live_test:
                raise
            # Never put response bodies or credentials in this reason.
            reason = str(error) if isinstance(error, ModelError) else "The connection failed."
        # The endpoint is not ready. Wait, then repeat the same call.
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        attempts += 1
        logging.getLogger(__name__).warning("Endpoint wait label=%s attempt=%s reason=%s", label, attempts, reason)
        print(f"{label} is not ready: {reason} Retrying in {delay_seconds:g} seconds "
              f"(attempt {attempts}; this wait has no limit).", file=sys.stderr)
        await cancellable(asyncio.sleep(delay_seconds), cancel)


class JsonClient:
    """Share cancellable transient waits across compatible model endpoints.

    Retries repeat the same HTTP request after a connection failure, HTTP 429
    or HTTP 5xx. A lost response can therefore cause duplicate inference.
    This client does not retry tool execution or accept a partial response.
    """

    def __init__(self, connection: Connection, *, live_test: bool = False,
                 transient_retries: int | None = None, retry_delay_seconds: float = 30) -> None:
        self.connection = connection
        self.live_test = live_test
        self.transient_retries = 0 if live_test and transient_retries is None else transient_retries
        if self.transient_retries is not None and self.transient_retries < 0:
            raise ValueError("transient_retries must be nonnegative or None for unlimited waits.")
        self.retry_delay_seconds = retry_delay_seconds
        self.request_count = 0
        self.cache_hits = 0

    async def _send(self, client: httpx.AsyncClient, method: str, route: str, headers: dict[str, str],
                    payload: dict[str, object] | None, cancel: Event | None, cache_path: Path | None = None,
                    transcript_path: Path | None = None) -> httpx.Response:
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
        logger = logging.getLogger(__name__)
        logger.info("Model request start route=%s model=%s cache_hit=%s",
                    route, self.connection.model, replayed is not None)
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
                logger.info("Model request stopped route=%s error=%s", route, type(error).__name__)
                raise
            self._store(cache_path, response)
        try:
            data = response.json()
        except ValueError:
            data = None
        error = completion_error(data) if response.is_success else ModelError(f"HTTP {response.status_code}")
        finish_prompt_log(path, response=data, error=error, elapsed=time.monotonic() - started,
                          replayed=replayed is not None)
        logger.info("Model request end route=%s status=%s seconds=%.3f", route, response.status_code,
                    time.monotonic() - started)
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

    async def request(self, method: str, route: str, *, payload: dict[str, object] | None = None,
                      cancel: Event | None = None, transcript_path: Path | None = None) -> dict:
        """Wait through temporary failures until recovery or cancellation.

        Production waits have no retry quota. An explicit transient_retries
        bound permits finite diagnostics; exhaustion raises ModelError.
        Invalid JSON, provider-filtered output and permanent HTTP errors fail
        directly. Cancellation interrupts in-flight I/O and retry waits.

        The returned mapping is the provider's JSON body. It stays open here
        because each route has its own shape; the caller validates it (for chat
        through :class:`~pm.llm.activation.ChatResult`) or reads named fields.
        """
        headers = {}
        key = self.connection.api_key.get_secret_value()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        cache_path: Path | None = None
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
                            response = await self._send(client, method, route, headers, payload, cancel, cache_path, transcript_path)
                    else:
                        response = await self._send(client, method, route, headers, payload, cancel, cache_path, transcript_path)
                except httpx.TransportError:
                    # A transport failure is retryable: the request may never
                    # have reached the model, and the wait has no quota.
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
                # Exhaustion is only reachable through an explicit diagnostic
                # bound. Production keeps the same request until cancellation.
                if self.transient_retries is not None and retries >= self.transient_retries:
                    raise ModelError(failure) from None
                retries += 1
                logging.getLogger(__name__).warning("Model request retry route=%s attempt=%s delay_seconds=%s",
                                                    route, retries, self.retry_delay_seconds)
                limit = str(self.transient_retries) if self.transient_retries is not None else "until cancellation"
                print(f"{failure} Retrying in {self.retry_delay_seconds:g} seconds "
                      f"({retries}/{limit}).", file=sys.stderr)
                await cancellable(asyncio.sleep(self.retry_delay_seconds), cancel)
