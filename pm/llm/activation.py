"""Explicit model activation and active-call ownership.

Finite startup scripts run only during activate(). A failed script or wrong
served alias raises ModelError. Production readiness waits are cancellable.
Live tests fail on unavailable endpoints. Logic errors retain tracebacks.
The owner covers clients in this process, not unrelated external model users.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pm.llm.http_client import JsonClient, ModelError, cancellable
from pm.llm.config import Connection


def script_command(script: Path) -> list[str]:
    if not script.is_file():
        raise ModelError("The startup script does not exist. Correct startup_script in the config.")
    if script.suffix.lower() == ".ps1":
        return [shutil.which("pwsh") or "powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    if script.suffix.lower() == ".py":
        return [sys.executable, str(script)]
    if script.suffix.lower() == ".sh":
        return ["bash", str(script)]
    raise ModelError("Use a finite .ps1, .sh, or .py startup script.")


async def run_startup(script: Path, cancel: Event | None = None) -> None:
    if cancel is not None and cancel.is_set():
        raise asyncio.CancelledError
    process = await asyncio.create_subprocess_exec(
        *script_command(script), cwd=script.parent,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        code = await cancellable(process.wait(), cancel)
        if code != 0:
            raise ModelError(f"The model startup script failed (exit {code}). Check the script's local log.")
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()


class ModelActivation:
    def __init__(self):
        self._lock = Lock()
        self._active_calls = 0
        self._switching = False
        self._ready: dict[str, str] = {}

    def is_ready(self, connection: Connection) -> bool:
        """Return whether this owner admits calls for the resolved alias."""
        with self._lock:
            return not self._switching and self._ready.get(connection.base_url) == connection.model

    async def activate(self, connection: Connection, *, cancel: Event | None = None, live_test: bool = False):
        with self._lock:
            if self._active_calls or self._switching:
                raise ModelError("A model is in use. Finish active calls before activation.")
            self._switching = True
            # A script can switch GPU ownership across ports. Recheck each later group.
            self._ready.clear()
        try:
            if connection.startup_script is not None:
                await run_startup(connection.startup_script, cancel)
            result = await JsonClient(connection, live_test=live_test).request("GET", "models", cancel=cancel)
            try:
                aliases = {entry["id"] for entry in result["data"]}
            except (KeyError, TypeError):
                raise ModelError("Model discovery returned an invalid model list.") from None
            if connection.model not in aliases:
                raise ModelError("The endpoint does not advertise the configured model alias.")
            with self._lock:
                self._ready[connection.base_url] = connection.model
        finally:
            with self._lock:
                self._switching = False

    @contextmanager
    def use(self, connection: Connection):
        with self._lock:
            if self._switching or self._ready.get(connection.base_url) != connection.model:
                raise ModelError("Activate the configured model before a generation group.")
            self._active_calls += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_calls -= 1


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)
    content: str | None = None
    tool_calls: list[dict] = Field(default_factory=list)


class ChatChoice(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)
    message: ChatMessage
    finish_reason: str | None = None


class ChatResult(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)
    model: str
    choices: list[ChatChoice] = Field(min_length=1)


class ChatClient:
    def __init__(self, connection: Connection, activation: ModelActivation, *, live_test: bool = False):
        self.connection = connection
        self.activation = activation
        self.client = JsonClient(connection, live_test=live_test)

    async def complete(self, messages: list[dict], *, cancel: Event | None = None, **generation) -> ChatResult:
        if "model" in generation or "stream" in generation:
            raise ValueError("Use the resolved model and non-streaming response path.")
        with self.activation.use(self.connection):
            response = await self.client.request("POST", "chat/completions", payload={
                **generation, "model": self.connection.model, "messages": messages, "stream": False,
            }, cancel=cancel)
        try:
            result = ChatResult.model_validate(response)
        except ValidationError:
            raise ModelError("The model service returned an invalid chat completion.") from None
        if result.model != self.connection.model:
            raise ModelError("The completion model does not match the activated model alias.")
        if result.choices[0].finish_reason in {"length", "content_filter"}:
            raise ModelError("The model completion ended before a usable response: "
                             f"finish_reason={result.choices[0].finish_reason}; usage={response.get('usage', {})}.")
        message = result.choices[0].message
        if not message.content and not message.tool_calls:
            raise ModelError("The model returned no content or tool call. Inspect its generation config.")
        return result
