"""Lazy, application-owned model activation for the real cognitive path.

``build_app`` never contacts a model. The application constructs one
``LazyStageModel`` and hands it to every real workflow. The first completion
activates the configured connection on the calling module thread; later
completions reuse it. Activation is therefore explicit and observable, and no
model switch can happen while another completion is in flight.

Visible completions have no automatic ``max_tokens`` cap. Workflows can budget
hidden reasoning separately. The application shares admission slots across
module threads. Cancellation releases a slot after the request finishes
cleanup. An empty response is retried at most once. An endpoint that is down
pauses activation and requests until it returns; it never ends the session.
"""
from __future__ import annotations

import asyncio
from threading import BoundedSemaphore, Event, Lock
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pm.llm.contracts import ContextMeasurement, ModelProtocolContext

from pm.llm.activation import ChatClient, ModelActivation, OpenAICompletionBackend
from pm.llm.http_client import EmptyCompletionError
from pm.llm.config import Connection
from pm.llm.completion import AsyncTextCompletion, Completion, ToolCompletion
from pm.llm.prompt_encoding import TurnInput
from pm.llm.concurrency import inference_admission


async def complete_with_retry(
    client: ChatClient,
    messages: Sequence[TurnInput],
    generation: dict[str, object],
) -> str:
    """One completion with a bounded empty-content retry.

    An empty completion is stochastic, so one fresh sample is worth asking
    for. A second empty answer fails loudly: the prompt or the generation
    configuration needs a human look, and no fallback text replaces it.
    """
    last: EmptyCompletionError | None = None
    for attempt in range(2):
        try:
            result = await client.complete(messages, **generation)
        except EmptyCompletionError as error:
            if attempt == 1:
                raise
            last = error
            continue
        return result.text
    raise last if last is not None else EmptyCompletionError("The model returned no content.")


class LazyStageModel:
    """Activate a configured connection on first use, then serve completions."""

    def __init__(
        self,
        connection: Connection,
        owner: ModelActivation,
        *,
        live_test: bool = False,
        max_concurrency: int = 4,
    ) -> None:
        self.connection = connection
        self.owner = owner
        self.live_test = live_test
        self.admission = inference_admission(connection.base_url)
        self._lock = Lock()
        self._client: ChatClient | None = None
        self._slots = BoundedSemaphore(max_concurrency)

    @asynccontextmanager
    async def slot(self, cancel: Event | None = None) -> AsyncIterator[None]:
        """Admit one request across module threads. Waiting consumes no worker thread."""
        while not self._slots.acquire(blocking=False):
            if cancel is not None and cancel.is_set():
                raise asyncio.CancelledError
            await asyncio.sleep(0.05)
        try:
            if cancel is not None and cancel.is_set():
                raise asyncio.CancelledError
            yield
        finally:
            self._slots.release()

    async def _preflight(self, messages: Sequence[TurnInput], cancel: Event | None, generation: dict[str, object]) -> None:
        """Reject impossible rendered context before a text, reasoning, or tool call."""
        from pm.llm.context_window import measure_context
        from pm.context_render.budget import RequiredContextTooLarge
        measurement = await measure_context(self.connection, messages, cancel if cancel is not None else Event(),
            live_test=self.live_test, protocol_context=ModelProtocolContext.model_validate(generation))
        if measurement["input_tokens"] > measurement["input_budget_tokens"]:
            raise RequiredContextTooLarge(f"Rendered prompt estimate {measurement['input_tokens']} exceeds input ceiling {measurement['input_budget_tokens']}.")

    async def _complete(self, messages: Sequence[TurnInput], generation: dict[str, object]) -> str:
        cancel = generation.get("cancel")
        assert cancel is None or isinstance(cancel, Event)
        await self._preflight(messages, cancel, generation)
        async with self.slot(cancel):
            return await complete_with_retry(self.ensure_client(cancel=cancel), messages, generation)

    @property
    def activated(self) -> bool:
        with self._lock:
            return self._client is not None

    @property
    def client(self) -> ChatClient:
        return self.ensure_client()

    @property
    def request_count(self) -> int:
        with self._lock:
            client = self._client
        return client.client.request_count if client is not None else 0

    def ensure_client(self, *, cancel: Event | None = None) -> ChatClient:
        """Activate and cache the client. Cancellation also stops readiness waits.

        Activation waits as long as the endpoint is down, so the first
        completion pauses rather than failing. See
        :meth:`pm.llm.activation.ModelActivation.activate`.
        """
        with self._lock:
            # Classification can already own this same ready connection.
            # Reuse it so the first cognitive call does not request a switch.
            if not self.owner.is_ready(self.connection):
                asyncio.run(self.owner.activate(self.connection, cancel=cancel, live_test=self.live_test))
            if self._client is None:
                self._client = ChatClient(self.connection, self.owner, live_test=self.live_test)
            return self._client

    def complete(self, messages: Sequence[TurnInput], **generation: object) -> str:
        """Synchronous completion for Python-role stages."""
        self.ensure_client(cancel=generation.get("cancel"))
        return asyncio.run(self._complete(messages, generation))

    def complete_result(self, messages: Sequence[TurnInput], **generation: object) -> Completion:
        """Return final content and reasoning separately for an explicit caller.

        Activation and provider errors retain their existing boundaries.
        This method does not substitute reasoning for a missing final answer.
        """
        client = self.ensure_client(cancel=generation.get("cancel"))

        async def complete() -> Completion:
            cancel = generation.get("cancel")
            assert cancel is None or isinstance(cancel, Event)
            await self._preflight(messages, cancel, generation)
            async with self.slot(cancel):
                return await OpenAICompletionBackend(client).complete(messages, **generation)

        return asyncio.run(complete())

    def measure_context(self, messages: Sequence[TurnInput], *, cancel: Event | None = None, **generation: object) -> ContextMeasurement:
        """Measure tokens using the same activated connection as inference.

        A dead endpoint pauses this call until the endpoint returns. It does
        not raise for an unavailable endpoint.
        """
        from threading import Event
        from pm.llm.context_window import measure_context
        self.ensure_client(cancel=cancel)
        with self.owner.use(self.connection):
            return asyncio.run(measure_context(self.connection, messages, cancel if cancel is not None else Event(),
                                               live_test=self.live_test, **generation))

    def build_async_tool(self, *, cancel: Event | None = None) -> AsyncTextCompletion:
        """Return an async completion callable for roles that run their own loop.

        Activation happens now, when the tool is built on the module thread,
        so the tool body only awaits an already-activated client. ``cancel``
        also stops readiness waits before the tool exists.
        """
        self.ensure_client(cancel=cancel)

        return AsyncModelTool(self, cancel)

    def build_tool_completion(self, *, cancel: Event | None = None) -> ToolCompletion:
        """Return an async tool-calling completion that preserves tool_calls.

        Assistant dialogue needs the full assistant message, including any
        requested tool calls. Activation happens when the tool is built.
        ``cancel`` stops that activation when the owning workflow stops.
        """
        client = self.ensure_client(cancel=cancel)

        async def complete_tool(messages: Sequence[TurnInput], tools=None, **generation: object) -> Completion:
            selected_cancel = generation.get("cancel", cancel)
            assert selected_cancel is None or isinstance(selected_cancel, Event)
            if tools:
                generation["tools"] = tools
            await self._preflight(messages, selected_cancel, generation)
            last: EmptyCompletionError | None = None
            for attempt in range(3):
                try:
                    async with self.slot(generation.get("cancel")):
                        result = await client.complete(messages, **generation)
                    return result.completion()
                except EmptyCompletionError as error:
                    last = error
                    if attempt == 2:
                        raise
                    await asyncio.sleep(0.3 * (attempt + 1))
            raise last if last is not None else EmptyCompletionError("The model returned no content.")

        return complete_tool


__all__ = ["LazyStageModel", "complete_with_retry"]


@dataclass(frozen=True)
class AsyncModelTool:
    """A measured asynchronous completion with an explicit owner cancellation event."""

    model: LazyStageModel
    owner_cancel: Event | None

    async def __call__(self, turns: Sequence[TurnInput], /, *, cancel: Event | None = None,
                       **generation: object) -> str:
        generation["cancel"] = cancel if cancel is not None else self.owner_cancel
        return await self.model._complete(turns, generation)

    def measure_context(self, messages: Sequence[TurnInput], *, cancel: Event | None = None,
                        **generation: object) -> ContextMeasurement:
        return self.model.measure_context(messages, cancel=cancel if cancel is not None else self.owner_cancel, **generation)
