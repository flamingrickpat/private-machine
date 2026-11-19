"""Explicit model activation and active-call ownership.

Finite startup scripts run only during activate(). Production activation keeps
retrying until the endpoint serves the configured alias, so a stopped server
pauses the caller instead of raising ModelError. A failed script and a wrong
served alias follow the same wait. Cancellation ends every wait, and live
tests still fail on the first error. Logic errors retain their tracebacks.
The owner covers clients in this process, not unrelated external model users.
"""
from __future__ import annotations

from pm.utils.operating_system import OperatingSystem
import asyncio
import json
import httpx
import logging
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Lock
from typing import TypedDict, Required, Literal
from collections.abc import Generator
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, JsonValue, TypeAdapter

from pm.llm.http_client import EmptyCompletionError, JsonClient, ModelError, cancellable, EndpointRetry
from pm.llm.config import Connection
from pm.llm.multimodal import encode_messages
from pm.llm.prompt_encoding import TurnInput
from pm.llm.completion import Completion, Reasoning, ProviderTokenUsage
from pm.llm.reasoning import ReasoningOptions
from pm.llm.generation import GenerationSettings
from pm.context_render.turns import Role, Turn, ProviderState, ToolCall


class ModelDiscoveryLimits(TypedDict, total=False):
    """The serving limits that a local endpoint reports for one model.

    A local server states its loaded context window here. The overlay reads it
    only as a fallback, because a training context is not a serving limit.
    """

    n_ctx: int


class ModelDiscoveryReasoning(TypedDict, total=False):
    """The reasoning support that a hosted catalog advertises for one model."""

    supports_max_tokens: bool


class ModelDiscoveryProvider(TypedDict, total=False):
    """The provider-side caps that a hosted catalog publishes for one model."""

    max_completion_tokens: int | None


class ModelDiscoveryEntry(TypedDict, total=False):
    """One entry of a provider model catalog (``GET /models``).

    Only the fields this runtime reads are named. A provider can publish more
    optional fields, and the entry keeps them untouched.
    """

    id: Required[str]
    context_length: int
    reasoning: ModelDiscoveryReasoning
    top_provider: ModelDiscoveryProvider
    meta: ModelDiscoveryLimits


def script_command(script: Path) -> list[str]:
    """Return the interpreter and arguments for one finite startup script.

    Raises ModelError for a missing script or an unsupported suffix, so a
    wrong config entry fails before any process starts.
    """
    if not script.is_file():
        raise ModelError("The startup script does not exist. Correct startup_script in the config.")
    if script.suffix.lower() not in {".ps1", ".py", ".sh"}:
        raise ModelError("Use a finite .ps1, .sh, or .py startup script.")
    return OperatingSystem.script_command(script)


async def run_startup(script: Path, cancel: Event | None = None) -> None:
    """Run one finite startup script and wait for its exit code.

    A nonzero exit raises ModelError with the code only; the script's output
    stays in its own local log. ``cancel`` stops the process and re-raises
    ``asyncio.CancelledError`` so the owner's cancellation is never swallowed.
    """
    if cancel is not None and cancel.is_set():
        raise asyncio.CancelledError
    process = await OperatingSystem.start_async_process(script_command(script), cwd=script.parent)
    try:
        code = await cancellable(process.wait(), cancel)
        if code != 0:
            raise ModelError(f"The model startup script failed (exit {code}). Check the script's local log.")
    finally:
        await OperatingSystem.terminate_async_process(process)


class ModelActivation:
    """Own which model this process admits calls for, and who may switch it.

    The owner is process-wide. One activation holds the switch, clears the
    readiness of every endpoint, and records the served alias and its
    discovery metadata. A generation group holds one active call, so a switch
    cannot start while another completion is in flight.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._active_calls = 0
        self._switching = False
        self._ready: dict[str, str] = {}
        self._metadata: dict[tuple[str, str], ModelDiscoveryEntry] = {}

    def is_ready(self, connection: Connection) -> bool:
        """Return whether this owner admits calls for the resolved alias."""
        with self._lock:
            return not self._switching and self._ready.get(connection.base_url) == connection.model

    async def activate(self, connection: Connection, *, cancel: Event | None = None,
                       live_test: bool = False) -> None:
        """Activate the connection. Wait as long as the endpoint is not ready.

        A closed port, a model that is still loading, and a failed startup
        script are repeated until the endpoint serves the configured alias.
        The caller waits instead of failing, so an endpoint restart resumes
        the paused work. A wait for a busy model also clears, because the
        active calls finish. ``cancel`` ends the wait with
        ``asyncio.CancelledError``. ``live_test=True`` fails on the first
        error, which is what tests and diagnostic gates need.
        """
        retry = EndpointRetry(f"Model {connection.model} at {connection.base_url}", cancel, live_test=live_test)
        while True:
            try:
                return await self._activate(connection, cancel=cancel, live_test=live_test)
            except (ModelError, httpx.HTTPError) as error:
                await retry.wait(error)


    async def _activate(self, connection: Connection, *, cancel: Event | None = None,
                        live_test: bool = False) -> None:
        """Try activation once. Readiness and errors follow the old contract."""
        logger = logging.getLogger(__name__)
        logger.info("Model activation start model=%s endpoint=%s", connection.model, connection.base_url)
        with self._lock:
            if self._active_calls or self._switching:
                raise ModelError("A model is in use. Finish active calls before activation.")
            self._switching = True
            # A script can switch GPU ownership across ports. Recheck each later group.
            self._ready.clear()
        try:
            if connection.startup_script is not None:
                await run_startup(connection.startup_script, cancel)
            result = ModelCatalog.model_validate(await JsonClient(connection, live_test=live_test).request("GET", "models", cancel=cancel))
            try:
                aliases = {entry["id"] for entry in result.data}
            except (KeyError, TypeError):
                raise ModelError("Model discovery returned an invalid model list.") from None
            if connection.model not in aliases:
                raise ModelError("The endpoint does not advertise the configured model alias.")
            with self._lock:
                self._ready[connection.base_url] = connection.model
                # Keep the catalog entry so a later call can read the served
                # context window and the advertised reasoning support.
                self._metadata[(connection.base_url, connection.model)] = next(
                    entry for entry in result.data if entry["id"] == connection.model)
        finally:
            with self._lock:
                self._switching = False
            logger.info("Model activation exit model=%s ready=%s", connection.model, self.is_ready(connection))

    @contextmanager
    def use(self, connection: Connection) -> Generator[None, None, None]:
        """Hold one active call so no activation can switch the model meanwhile."""
        with self._lock:
            if self._switching or self._ready.get(connection.base_url) != connection.model:
                raise ModelError("Activate the configured model before a generation group.")
            self._active_calls += 1
        try:
            yield
        finally:
            with self._lock:
                self._active_calls -= 1

    def metadata(self, connection: Connection) -> ModelDiscoveryEntry:
        """Return a copy of the discovery metadata of the active model.

        Activation owns the stored entry; a caller gets its own mapping so no
        reader can change what a later reasoning budget reads.
        """
        with self._lock:
            entry = self._metadata[(connection.base_url, connection.model)]
        return ModelDiscoveryEntry(**entry)


class ChatFunctionCall(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    arguments: str


class ChatToolCall(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    type: Literal["function"] = "function"
    function: ChatFunctionCall


class ChatMessage(BaseModel):
    """One provider assistant message, keeping provider extras for replay.

    ``content`` stays separate from ``reasoning``, ``reasoning_content``, and
    the signed ``reasoning_details`` that a later tool turn can replay.
    Unexpected provider fields are kept because ``extra="allow"``.
    """

    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)
    content: str | None = None
    tool_calls: list[ChatToolCall] = Field(default_factory=list[ChatToolCall])
    reasoning: str | None = None
    reasoning_content: str | None = None
    reasoning_details: list[dict[str, JsonValue]] | None = None


class ChatChoice(BaseModel):
    """One provider choice. ``finish_reason`` decides whether it is usable."""

    model_config = ConfigDict(extra="allow", hide_input_in_errors=True)
    message: ChatMessage
    finish_reason: str | None = None


class ChatResult(BaseModel):
    """One non-streaming chat completion with its usage and provider extras."""

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
        details = json.dumps(message.reasoning_details, ensure_ascii=False) if message.reasoning_details is not None else None
        state = ProviderState(message.reasoning, message.reasoning_content, details)
        turn = Turn(Role.Assistant, message.content or "", tuple(ToolCall(call.id, call.function.name, call.function.arguments) for call in message.tool_calls), provider_state=state)
        reasoning = Reasoning(message.reasoning or message.reasoning_content or "", details)
        usage = ProviderTokenUsage.model_validate((self.model_extra or {}).get("usage") or {})
        return Completion(turn, reasoning, usage, self.model, choice.finish_reason)



OPENROUTER_HOST = "openrouter.ai"




class ModelCatalog(BaseModel):
    data: list[ModelDiscoveryEntry]


class ChatClient:
    """Send neutral turns to one activated connection and check the result.

    Reasoning budgets are resolved against the activated model metadata, image
    tags become provider parts, and the response is validated into
    :class:`ChatResult`. A truncated, filtered, or empty completion raises
    instead of returning a partial answer.
    """

    def __init__(self, connection: Connection, activation: ModelActivation, *, live_test: bool = False) -> None:
        self.connection = connection
        self.activation = activation
        self.client = JsonClient(connection, live_test=live_test)

    async def complete(self, messages: Sequence[TurnInput], *, cancel: Event | None = None,
                       settings: GenerationSettings = GenerationSettings()) -> ChatResult:
        """Return one validated completion for these turns."""
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        maximum = settings.output_allowance(self.connection.completion_token_limit)
        reasoning = settings.reasoning
        tokens = settings.reasoning_tokens or settings.reasoning_budget_tokens
        if settings.reasoning_tokens is not None and settings.reasoning_budget_tokens is not None:
            raise ValueError("Specify one reasoning token allowance.")
        if reasoning is not None and (tokens is not None or settings.reasoning_effort is not None):
            raise ValueError("Use reasoning options or legacy aliases, not both.")
        if reasoning is None:
            thinking = settings.enable_thinking
            if thinking is None and settings.chat_template_kwargs is not None:
                thinking = settings.chat_template_kwargs.enable_thinking
            reasoning = ReasoningOptions(enabled=thinking, max_tokens=tokens, effort=settings.reasoning_effort)
        budget = settings.reasoning_budget
        if budget is not None:
            if settings.reasoning is not None or tokens is not None or settings.reasoning_effort is not None:
                raise ValueError("Choose a reasoning budget or explicit reasoning controls.")
            from pm.llm.context_window import estimate_tokens
            from pm.llm.workflow_sessions import serving_context
            context, _source = await serving_context(self.connection, cancel if cancel is not None else Event(), live_test=self.client.live_test)
            input_tokens, _label = estimate_tokens(self.connection.model, messages)
            reasoning = budget.generation(context_window=context, input_tokens=input_tokens, max_output_tokens=maximum)
        encoded = encode_messages(messages)
        outbound = settings.model_dump(mode="json", by_alias=True, exclude_none=True,
            exclude={"reasoning_budget", "reasoning", "enable_thinking", "reasoning_tokens", "reasoning_budget_tokens",
                     "reasoning_effort", "chat_template_kwargs"})
        if not settings.tools:
            outbound.pop("tools")
        if settings.max_tokens is None and settings.max_completion_tokens is None:
            outbound["max_tokens"] = maximum
        controls = reasoning.model_dump(exclude_none=True)
        host = (urlsplit(self.connection.base_url).hostname or "").lower()
        if host == OPENROUTER_HOST:
            if controls:
                outbound["reasoning"] = controls
        else:
            template = settings.chat_template_kwargs.model_dump(exclude_none=True) if settings.chat_template_kwargs is not None else {}
            if reasoning.enabled is not None:
                template["enable_thinking"] = reasoning.enabled
            elif reasoning.max_tokens is not None:
                template["enable_thinking"] = True
            if reasoning.effort is not None:
                outbound["reasoning_effort"] = reasoning.effort
                template["enable_thinking"] = reasoning.effort != "none"
            if template:
                outbound["chat_template_kwargs"] = template
            if reasoning.max_tokens is not None:
                outbound["reasoning_budget_tokens"] = reasoning.max_tokens
            if reasoning.exclude is not None:
                outbound["reasoning"] = {"exclude": reasoning.exclude}
        payload = TypeAdapter(dict[str, JsonValue]).validate_python({
            **outbound, "model": self.connection.model, **encoded, "stream": False,
        })
        with self.activation.use(self.connection):
            response = await self.client.request("POST", "chat/completions", payload=payload, cancel=cancel)
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
        if reasoning.exclude:
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
                       settings: GenerationSettings = GenerationSettings()) -> Completion:
        """Return content and reasoning separately; transport errors propagate."""
        result = await self.client.complete(turns, cancel=cancel, settings=settings)
        return result.completion()
