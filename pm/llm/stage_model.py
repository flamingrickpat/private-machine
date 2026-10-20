"""Lazy, application-owned model activation for the real cognitive path.

``build_app`` never contacts a model. The application constructs one
``LazyStageModel`` and hands it to every real workflow. The first completion
activates the configured connection on the calling module thread; later
completions reuse it. Activation is therefore explicit and observable, and no
model switch can happen while another completion is in flight.

Visible completions have no automatic ``max_tokens`` cap. Workflows can budget
hidden reasoning separately. The application shares admission slots across
module threads. Cancellation releases a slot after the request finishes
cleanup. An empty response is retried at most once.
"""
from __future__ import annotations

import asyncio
from threading import BoundedSemaphore, Lock
from contextlib import asynccontextmanager
from collections.abc import Sequence

from pm.llm.activation import ChatClient, ModelActivation, OpenAICompletionBackend
from pm.llm.http_client import ModelError
from pm.llm.config import Connection
from pm.llm.completion import AsyncTextCompletion, Completion, ToolCompletion
from pm.llm.prompt_encoding import TurnInput
from pm.llm.concurrency import inference_admission


async def complete_with_retry(
    client: ChatClient,
    messages: Sequence[TurnInput],
    generation: dict,
) -> str:
    """One completion with a bounded empty-content retry."""
    last: ModelError | None = None
    for attempt in range(2):
        try:
            result = await client.complete(messages, **generation)
        except ModelError as error:
            if "no content" not in str(error) or attempt == 1:
                raise
            last = error
            continue
        return result.text
    raise last if last is not None else ModelError("The model returned no content.")


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
    async def slot(self, cancel=None):
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

    async def _complete(self, messages, generation):
        async with self.slot(generation.get("cancel")):
            return await complete_with_retry(self.ensure_client(cancel=generation.get("cancel")), messages, generation)

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

    def ensure_client(self, *, cancel=None) -> ChatClient:
        """Activate and cache the client. Cancellation also stops readiness waits."""
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

        async def complete():
            async with self.slot(generation.get("cancel")):
                return await OpenAICompletionBackend(client).complete(messages, **generation)

        return asyncio.run(complete())

    def measure_context(self, messages: Sequence[TurnInput], *, cancel=None, **generation: object) -> dict:
        """Measure tokens using the same activated connection as inference."""
        from threading import Event
        from pm.llm.context_window import measure_context
        self.ensure_client(cancel=cancel)
        with self.owner.use(self.connection):
            return asyncio.run(measure_context(self.connection, messages, cancel if cancel is not None else Event(), **generation))

    def build_async_tool(self) -> AsyncTextCompletion:
        """Return an async completion callable for roles that run their own loop.

        Activation happens now, when the tool is built on the module thread,
        so the tool body only awaits an already-activated client.
        """
        self.ensure_client()

        async def complete_async(messages: Sequence[TurnInput], **generation: object) -> str:
            return await self._complete(messages, generation)

        complete_async.measure_context = self.measure_context
        return complete_async

    def build_tool_completion(self) -> ToolCompletion:
        """Return an async tool-calling completion that preserves tool_calls.

        Assistant dialogue needs the full assistant message, including any
        requested tool calls. Activation happens when the tool is built.
        """
        client = self.ensure_client()

        async def complete_tool(messages: Sequence[TurnInput], tools=None, **generation: object) -> Completion:
            if tools:
                generation["tools"] = tools
            last: ModelError | None = None
            for attempt in range(3):
                try:
                    async with self.slot(generation.get("cancel")):
                        result = await client.complete(messages, **generation)
                    return result.completion()
                except ModelError as error:
                    last = error
                    if attempt == 2:
                        raise
                    await asyncio.sleep(0.3 * (attempt + 1))
            raise last if last is not None else ModelError("The model returned no content.")

        return complete_tool


__all__ = ["LazyStageModel", "complete_with_retry"]
