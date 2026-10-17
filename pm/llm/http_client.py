"""One HTTP path for chat, readiness, and embeddings.

Production waits through transient transport and server errors, with a
30-second pause until cancellation. Tests can select a bounded or fail-fast
policy. Permanent HTTP errors and invalid responses fail explicitly. Programming errors retain their tracebacks.
Neither response bodies nor credentials appear in transport error messages.

Every chat completion carries a process seed and is replayed from a local
completion cache. The cache is a development shortcut: it makes a repeated
run cheap, and it is not evidence that the endpoint still answers.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path
from threading import Event, Lock

import httpx

from pm.llm.config import Connection
from pm.llm.prompt_log import log_directory, DEFAULT_DIR

_SEED_LOCK = Lock()
_SEED = 0


def next_seed() -> int:
    """Return the seed for the next chat completion.

    The first completion in a process uses 1, the next uses 2, and so on.
    The sequence repeats in a later process, so a repeated run of the same
    completion order reads the same cache files.
    """
    global _SEED
    with _SEED_LOCK:
        _SEED += 1
        return _SEED


def replay_path(payload: dict) -> Path:
    """Return the cache file for this exact payload.

    The key covers the model, the prompt, the generation settings, and the
    seed, so a matching file belongs to the same completion. The cache shares
    the prompt-log directory, which tests redirect to a temporary folder.
    """
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return DEFAULT_DIR / f"comp_hash_{digest}.json"


def replay_response(payload: dict) -> httpx.Response | None:
    """Return the stored wire response for this payload, or None when absent."""
    try:
        content = replay_path(payload).read_bytes()
    except FileNotFoundError:
        return None
    return httpx.Response(200, content=content)


def store_response(payload: dict, content: bytes) -> None:
    """Keep one exact wire response so a matching completion skips the endpoint.

    A write failure warns on stderr. It cannot change the completion result.
    """
    path = replay_path(payload)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    except OSError as error:
        print(f"Completion cache write failed: {type(error).__name__}: {error}", file=sys.stderr)


class ModelError(RuntimeError):
    pass


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

    async def _send(self, client, method: str, route: str, headers: dict,
                    payload: dict | None, cancel: Event | None, seed: int | None = None):
        """Log each inference attempt before dispatch, including retries.

        A chat payload receives the caller's seed, or the next process seed.
        A stored response for the same payload replaces the HTTP request, so a
        replayed completion still writes its receipt but reaches no endpoint.
        Cancellation leaves the prompt and an error receipt. Logging failures
        cannot change the HTTP result. Readiness requests have no prompt log.
        """
        from pm.llm.prompt_log import start_prompt_log, finish_prompt_log
        started = time.monotonic()
        replayed = None
        if method == "POST":
            payload = {**(payload or {}), "seed": next_seed() if seed is None else seed}
            replayed = replay_response(payload)
        path = start_prompt_log(connection=self.connection, payload=payload, route=route,
                                request_count=self.request_count) if method == "POST" else None
        if replayed is not None:
            response = replayed
        else:
            try:
                response = await cancellable(client.request(
                    method, f"{self.connection.base_url}/{route}", headers=headers, json=payload,
                ), cancel)
            except BaseException as error:
                finish_prompt_log(path, error=error, elapsed=time.monotonic() - started)
                raise
            self._store(payload, response)
        try:
            data = response.json()
        except ValueError:
            data = None
        error = completion_error(data) if response.is_success else ModelError(f"HTTP {response.status_code}")
        finish_prompt_log(path, response=data, error=error, elapsed=time.monotonic() - started)
        return response

    @staticmethod
    def _store(payload: dict | None, response: httpx.Response) -> None:
        """Store a successful chat response. Errors and partial results are not replayable."""
        if payload is None or not response.is_success:
            return
        try:
            data = response.json()
        except ValueError:
            return
        if isinstance(data, dict) and completion_error(data) is None:
            store_response(payload, response.content)

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
        # One completion keeps one seed across retries, so a retried request
        # cannot shift the seeds of later completions.
        seed = next_seed() if method == "POST" else None
        # Connecting can fail. Once connected, inference has no speed deadline.
        async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0)) as client:
            retries = 0
            while True:
                if cancel is not None and cancel.is_set():
                    raise asyncio.CancelledError
                self.request_count += 1
                failure = None
                try:
                    response = await self._send(client, method, route, headers, payload, cancel, seed)
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
