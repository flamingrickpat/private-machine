"""Lazy chat connections for PM classification and isolated agent recovery.

Construction resolves configuration without network calls. Each model session
owns activation and serving context discovery. The caller retains redacted
service events beside its kernel journal. This module never selects Laya.
An endpoint that is down pauses the caller until it answers again; it does not
end the session.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sys
from threading import Event, Lock
import time
from typing import Iterator
from urllib.parse import urlsplit, urlunsplit

import httpx
from pm_workflows.inputs import InputError
from pm_workflows.runtime import AgentModel, WorkflowRuntime, WorkflowTerminated

from pm.llm.activation import ModelActivation
from pm.llm.http_client import JsonClient, ModelError, cancellable, wait_until_ready
from pm.llm.config import Connection, LlmPreset, ModelConfig, resolve_connection


def wait_before_retry(cancel: Event, delay_seconds: float = 30) -> None:
    """Sleep between retries on this thread. Cancellation ends the sleep early."""
    deadline = time.monotonic() + delay_seconds
    while time.monotonic() < deadline:
        if cancel.is_set():
            raise WorkflowTerminated()
        time.sleep(0.1)


async def serving_context(connection: Connection, cancel: Event, *, live_test: bool = False) -> tuple[int, str]:
    """Read a configured bound or llama.cpp's actual serving context.

    OpenRouter supplies its catalog bound. Other remote endpoints need
    ``context_window`` when they have no compatible ``/props`` route.
    ``override_context`` can reduce the resolved window.
    Local training context metadata cannot establish serving limits.
    Cancellation closes the discovery request. No source context is clipped.

    An endpoint that is down, or that has not finished loading, is waited
    out. The discovery call repeats every 30 seconds until the endpoint
    answers, so the caller pauses instead of failing. ``live_test=True``
    returns the first error instead, for startup probes, tests, and gates.
    """
    if connection.context_window is not None:
        return min(connection.context_window, connection.override_context or connection.context_window), "configuration"
    return await wait_until_ready(lambda: _discover_serving_context(connection, cancel), cancel,
                                  live_test=live_test, label=f"Serving context at {connection.base_url}")


async def _discover_serving_context(connection: Connection, cancel: Event) -> tuple[int, str]:
    """Ask the endpoint for its serving context once. See :func:`serving_context`."""
    parts = urlsplit(connection.base_url)
    if parts.hostname == "openrouter.ai":
        # OpenRouter owns provider routing. Its catalog supplies the context
        # bound when the operator does not configure a smaller bound.
        result = await JsonClient(connection, live_test=True).request("GET", "models", cancel=cancel)
        entries = result.get("data", [])
        selected = next((item for item in entries if item.get("id") == connection.model), None)
        value = selected.get("context_length") if selected else None
        if type(value) is not int or value <= 0:
            raise ModelError("OpenRouter has no context bound for the configured model.")
        return min(value, connection.override_context or value), "openrouter.models.context_length"
    props_url = urlunsplit((parts.scheme, parts.netloc, parts.path.removesuffix("/v1") + "/props", "", ""))
    key = connection.api_key.get_secret_value()
    headers = {"Authorization": "Bearer " + key} if key else {}
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await cancellable(client.get(props_url, headers=headers), cancel)
        except httpx.TransportError:
            raise ModelError("Serving context discovery failed. Check chat or configure context_window.") from None
    try:
        response.raise_for_status()
        value = response.json()["default_generation_settings"]["n_ctx"]
    except (httpx.HTTPStatusError, ValueError, KeyError, TypeError):
        raise ModelError("The endpoint has no serving context. Configure chat.context_window.") from None
    if type(value) is not int or value <= 0:
        raise ModelError("The serving context must be a positive integer.")
    return min(value, connection.override_context or value), "server.props.default_generation_settings.n_ctx"


class WorkflowModels:
    """Supply named chat presets only when a classifier needs inference.

    The application can share its cognitive activation owner. Explicit runtime
    connections take precedence. Recovery uses Default unless the caller selects
    another preset. Agent diagnostics retain automatic compaction in live gates.
    """

    def __init__(
        self, config: ModelConfig, owner: ModelActivation, *,
        recovery_preset: LlmPreset = LlmPreset.Default, live_test: bool = False,
    ) -> None:
        self.connections = {str(name): resolve_connection(config, preset=name) for name in config.chat}
        self.owner = owner
        self.recovery_preset = recovery_preset
        self.live_test = live_test
        self._lock = Lock()
        self._receipt_lock = Lock()

    def bind(self, runtime: WorkflowRuntime, receipt_path: Path) -> WorkflowRuntime:
        """Return an invocation copy with a lazy session and local service log.

        Existing explicit sessions remain in control. Binding creates no file,
        starts no service, and sends no request. Children inherit this callback
        unless their factory supplies another session.
        """
        if runtime.classification_session is not None:
            return runtime

        def session(name: str, recovery: bool, cancel: Event):
            return self.session(name, recovery, cancel, receipt_path=receipt_path)

        return replace(runtime, classification_session=session)

    def _record(self, path: Path, event: dict) -> None:
        with self._receipt_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    @contextmanager
    def session(
        self, name: str, recovery: bool, cancel: Event, *, receipt_path: Path,
    ) -> Iterator[AgentModel]:
        """Activate, lease, and release one normal LLM connection.

        A dead endpoint holds this call open. Readiness, the lease, and
        context discovery are repeated every 30 seconds until the endpoint
        serves the configured model, so the caller pauses instead of failing
        and the session opens when the server is back. Only ``live_test``
        mode turns a readiness failure into a scoped INVALID cause.
        Cancellation raises ``WorkflowTerminated`` and releases active ownership.
        Unexpected code errors retain their traceback. Receipts omit credentials.
        The lease covers discovery, inference, and the full recovery session.
        """
        if cancel.is_set():
            raise WorkflowTerminated()
        selected = str(self.recovery_preset) if recovery else name
        event = {"schema": "pm.workflow-model-session.v1", "connection": selected,
                 "requested_connection": name, "recovery": recovery, "time": time.time()}
        connection = self.connections.get(selected)
        if connection is None:
            self._record(receipt_path, {**event, "event": "failed", "code": "missing_connection"})
            raise InputError("missing_connection", f"chat connection {selected!r} is not configured")
        event.update({"base_url": connection.base_url, "model": connection.model})
        started = time.monotonic()
        leased = False
        lease = None
        try:
            # Hold this lock only through readiness and lease acquisition.
            # Calls can then overlap, but activation cannot switch an active
            # model. A dead endpoint holds the lock through its retries, which
            # is correct: no other session can use that endpoint either.
            while not self._lock.acquire(timeout=0.1):
                if cancel.is_set():
                    raise WorkflowTerminated()
            try:
                attempts = 0
                while True:
                    if cancel.is_set():
                        raise WorkflowTerminated()
                    reused = self.owner.is_ready(connection)
                    # Build a fresh lease each pass. A rejected lease cannot be re-entered.
                    lease = self.owner.use(connection)
                    try:
                        if not reused:
                            asyncio.run(self.owner.activate(connection, cancel=cancel, live_test=self.live_test))
                        lease.__enter__()
                    except ModelError as error:
                        # Another thread can switch the active model between
                        # readiness and the lease. Wait and start over, forever.
                        if self.live_test:
                            raise
                        attempts += 1
                        print(f"Cannot use model {connection.model}: {error} Retrying in 30 seconds "
                              f"(attempt {attempts}; this wait has no limit).", file=sys.stderr)
                        wait_before_retry(cancel)
                        continue
                    leased = True
                    break
            finally:
                self._lock.release()
            context, source = asyncio.run(serving_context(connection, cancel, live_test=self.live_test))
            self._record(receipt_path, {**event, "event": "ready", "reused_activation": reused,
                                       "startup_script_ran": not reused and connection.startup_script is not None,
                                       "context_window": context, "context_source": source,
                                       "readiness_ms": (time.monotonic() - started) * 1000})
            yield AgentModel(connection.base_url, connection.model, connection.api_key.get_secret_value(),
                             context, enable_thinking=False, live_test=False)
        except (asyncio.CancelledError, WorkflowTerminated):
            self._record(receipt_path, {**event, "event": "cancelled"})
            raise WorkflowTerminated() from None
        except ModelError as error:
            self._record(receipt_path, {**event, "event": "failed", "code": "endpoint_unavailable", "reason": str(error)})
            raise InputError("endpoint_unavailable", str(error)) from None
        finally:
            if leased:
                lease.__exit__(None, None, None)
                self._record(receipt_path, {**event, "event": "released", "elapsed_ms": (time.monotonic() - started) * 1000})
