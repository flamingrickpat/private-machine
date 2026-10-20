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
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pm.llm.http_client import EmptyCompletionError, JsonClient, ModelError, cancellable
from pm.llm.config import Connection
from collections.abc import Sequence
from pm.llm.multimodal import encode_messages
from pm.llm.prompt_encoding import TurnInput, tool_calls_from_wire
from pm.llm.completion import Completion, Reasoning
from pm.llm.reasoning import ReasoningOptions
from pm.context_render.turns import Role, Turn


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
        self._metadata: dict[tuple[str, str], dict] = {}

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
                self._metadata[(connection.base_url, connection.model)] = next(
                    entry for entry in result["data"] if entry["id"] == connection.model)
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

    def metadata(self, connection: Connection) -> dict:
        """Return discovery metadata for the active model. Activation owns updates."""
        with self._lock:
            return dict(self._metadata[(connection.base_url, connection.model)])


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)
    content: str | None = None
    tool_calls: list[dict] = Field(default_factory=list)
    reasoning: str | None = None
    reasoning_content: str | None = None
    reasoning_details: list[dict] | None = None


class ChatChoice(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)
    message: ChatMessage
    finish_reason: str | None = None


class ChatResult(BaseModel):
    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)
    model: str
    choices: list[ChatChoice] = Field(min_length=1)

    @property
    def text(self) -> str:
        """Return final text without substituting reasoning or tool arguments."""
        return self.choices[0].message.content or ""

    def completion(self) -> Completion:
        """Normalize the provider result while retaining separate reasoning.

        Signed reasoning details stay available for replay on tool turns.
        Provider usage and final content remain distinct from that reasoning.
        """
        choice = self.choices[0]
        message = choice.message
        state = {key: value for key, value in message.model_dump(mode="json").items()
                 if key not in {"content", "tool_calls", "role"} and value is not None}
        turn = Turn(Role.Assistant, message.content or "", tool_calls_from_wire(message.tool_calls), provider_state=state or None)
        reasoning = Reasoning(message.reasoning or message.reasoning_content or "", tuple(message.reasoning_details or []))
        return Completion(turn, reasoning, (self.model_extra or {}).get("usage") or {}, self.model, choice.finish_reason)


OPENROUTER_HOST = "openrouter.ai"


def provider_generation(connection: Connection, generation: dict) -> dict:
    """Translate reasoning controls without changing the caller's dictionaries.

    ``reasoning`` accepts ReasoningOptions or its dictionary fields. OpenRouter
    receives that object and requires parameter support during routing. Local
    serving receives template controls and Strata's reasoning_budget_tokens.
    Explicit reasoning takes precedence over the legacy thinking switch.
    Duplicate budget or effort sources raise ValueError. Other kwargs pass through.
    """
    host = (urlsplit(connection.base_url).hostname or "").lower()
    result = dict(generation)
    tokens = result.pop("reasoning_tokens", None)
    local_tokens = result.pop("reasoning_budget_tokens", None)
    if local_tokens is not None:
        if tokens is not None:
            raise ValueError("Specify one reasoning token allowance.")
        tokens = local_tokens
    effort = result.pop("reasoning_effort", None)
    thinking = result.pop("enable_thinking", (result.get("chat_template_kwargs") or {}).get("enable_thinking"))
    if "reasoning" in result:
        if tokens is not None or effort is not None:
            raise ValueError("Use reasoning fields or legacy reasoning aliases, not both.")
        supplied = result.pop("reasoning")
        options = supplied if isinstance(supplied, ReasoningOptions) else ReasoningOptions.model_validate(supplied)
    else:
        options = ReasoningOptions(enabled=thinking, max_tokens=tokens, effort=effort)
    controls = options.model_dump(exclude_none=True)
    if host == OPENROUTER_HOST:
        result.pop("chat_template_kwargs", None)
        if controls:
            result["reasoning"] = controls
            result["provider"] = {"require_parameters": True, **result.get("provider", {})}
        return result
    template = dict(result.get("chat_template_kwargs") or {})
    if options.enabled is not None:
        template["enable_thinking"] = options.enabled
    elif options.max_tokens is not None:
        template["enable_thinking"] = True
    if options.effort is not None:
        result["reasoning_effort"] = options.effort
        if options.effort == "none":
            template["enable_thinking"] = False
        elif "enable_thinking" in template:
            template["enable_thinking"] = True
    if template:
        result["chat_template_kwargs"] = template
    if options.max_tokens is not None:
        result["reasoning_budget_tokens"] = options.max_tokens
    if options.exclude is not None:
        result["reasoning"] = {"exclude": options.exclude}
    return result


class ChatClient:
    def __init__(self, connection: Connection, activation: ModelActivation, *, live_test: bool = False):
        self.connection = connection
        self.activation = activation
        self.client = JsonClient(connection, live_test=live_test)

    async def complete(self, messages: Sequence[TurnInput], *, cancel: Event | None = None, **generation: object) -> ChatResult:
        if "model" in generation or "stream" in generation:
            raise ValueError("Use the resolved model and non-streaming response path.")
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        budget = generation.pop("reasoning_budget", None)
        if budget is not None:
            if generation.keys() & {"reasoning", "reasoning_tokens", "reasoning_effort", "reasoning_budget_tokens"}:
                raise ValueError("Choose a reasoning policy or explicit reasoning controls, not both.")
            if budget.fraction == 0:
                generation.update(enable_thinking=False)
            else:
                from pm.llm.context_window import estimate_tokens
                metadata = self.activation.metadata(self.connection)
                if ((urlsplit(self.connection.base_url).hostname or "").lower() == OPENROUTER_HOST
                        and not (metadata.get("reasoning") or {}).get("supports_max_tokens")):
                    raise ModelError("The model does not advertise numeric reasoning-budget support.")
                context = self.connection.context_window or metadata.get("context_length") or metadata["meta"]["n_ctx"]
                if type(context) is not int or context <= 0:
                    raise ModelError("A reasoning budget needs a configured or advertised context window.")
                maximum = (metadata.get("top_provider") or {}).get("max_completion_tokens") or context
                maximum = min(maximum, generation.get("max_tokens", maximum), generation.get("max_completion_tokens", maximum))
                if "tools" in generation:
                    import json
                    measured = [*messages, Turn(Role.User, json.dumps(generation["tools"], ensure_ascii=False))]
                else:
                    measured = messages
                input_tokens, _ = estimate_tokens(self.connection.model, measured)
                generation.update(budget.generation(context_window=context, input_tokens=input_tokens, max_output_tokens=maximum))
        # Image tags become image parts or alt text here, and only here.
        encoded = encode_messages(messages)
        outbound = provider_generation(self.connection, generation)
        with self.activation.use(self.connection):
            response = await self.client.request("POST", "chat/completions", payload={
                **outbound, "model": self.connection.model,
                **encoded, "stream": False,
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
        if not (message.content or "").strip() and not message.tool_calls:
            raise EmptyCompletionError("The model returned no content or tool call. Inspect its generation config.")
        if (outbound.get("reasoning") or {}).get("exclude"):
            # Local servers can retain reasoning in their wire response. Hide
            # it here too, while preserving final content, usage and raw logs.
            for choice in result.choices:
                choice.message.reasoning = None
                choice.message.reasoning_content = None
                choice.message.reasoning_details = None
        return result


class OpenAICompletionBackend:
    """Normalize an activated chat client into the shared completion contract."""

    def __init__(self, client: ChatClient) -> None:
        self.client = client

    async def complete(self, turns: Sequence[TurnInput], *, cancel: Event | None = None,
                       **generation: object) -> Completion:
        """Return content and reasoning separately; transport errors propagate."""
        result = await self.client.complete(turns, cancel=cancel, **generation)
        return result.completion()
