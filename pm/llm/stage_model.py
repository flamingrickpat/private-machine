"""Lazy, application-owned model activation for the real cognitive path.

``build_app`` never contacts a model. The application constructs one
``LazyStageModel`` and hands it to every real workflow. The first completion
activates the configured connection on the calling module thread; later
completions reuse it. Activation is therefore explicit and observable, and no
model switch can happen while another completion is in flight.

Visible completions are bounded by ``Connection.completion_token_limit`` (8192
by default) so a degenerate sample cannot run to the provider's whole output
ceiling. Workflows can budget hidden reasoning separately, and an explicit
``max_tokens`` from the caller still wins. The application shares admission
slots across module threads. Cancellation releases a slot after the request finishes
cleanup. An empty response is retried at most once. An endpoint that is down
pauses activation and requests until it returns; it never ends the session.
"""
from __future__ import annotations

import asyncio
from threading import Event, Lock
from collections.abc import Sequence
from pm.llm.contracts import ContextMeasurement
from pm.llm.generation import GenerationSettings

from pm.llm.activation import ChatClient, ChatResult, ModelActivation
from pm.llm.http_client import EmptyCompletionError
from pm.llm.config import Connection
from pm.llm.completion import Completion
from pm.context_render.prompt import RendererCaptureOptions
from pm.llm.prompt_encoding import TurnInput
from pm.llm.concurrency import inference_admission


async def complete_chat_with_retry(
    client: ChatClient,
    messages: Sequence[TurnInput],
    settings: GenerationSettings,
    cancel: Event | None = None,
) -> ChatResult:
    """One completion with a bounded empty-content retry, as the provider returned it.

    An empty completion is stochastic, so one fresh sample is worth asking
    for. A second empty answer fails loudly: the prompt or the generation
    configuration needs a human look, and no fallback text replaces it. The
    result keeps the provider usage block for a caller that retains receipts.
    """
    last: EmptyCompletionError | None = None
    for attempt in range(2):
        try:
            return await client.complete(messages, settings=settings, cancel=cancel)
        except EmptyCompletionError as error:
            if attempt == 1:
                raise
            last = error
            continue
    raise last if last is not None else EmptyCompletionError("The model returned no content.")


async def complete_with_retry(
    client: ChatClient,
    messages: Sequence[TurnInput],
    settings: GenerationSettings,
    cancel: Event | None = None,
) -> str:
    """Return the final text of :func:`complete_chat_with_retry`."""
    result = await complete_chat_with_retry(client, messages, settings, cancel)
    return result.text


class LazyStageModel:
    """Activate a configured connection on first use, then serve completions."""

    def __init__(
        self,
        connection: Connection,
        owner: ModelActivation,
        *,
        live_test: bool = False,
        capture_options: RendererCaptureOptions | None = None,
    ) -> None:
        self.connection = connection
        self.owner = owner
        self.live_test = live_test
        self.capture_options = capture_options if capture_options is not None else RendererCaptureOptions()
        self.admission = inference_admission(connection.base_url, max_concurrent_requests=connection.max_concurrent_requests)
        self._lock = Lock()
        self._client: ChatClient | None = None

    async def _preflight(self, messages: Sequence[TurnInput], cancel: Event | None, settings: GenerationSettings) -> None:
        from pm.llm.context_window import measure_context
        from pm.context_render.budget import RequiredContextTooLarge
        measurement = await measure_context(self.connection, messages, cancel if cancel is not None else Event(),
            live_test=self.live_test, settings=settings)
        if measurement["input_tokens"] > measurement["input_budget_tokens"]:
            raise RequiredContextTooLarge(f"Rendered prompt estimate {measurement['input_tokens']} exceeds input ceiling {measurement['input_budget_tokens']}.")

    async def complete_chat_async(self, messages: Sequence[TurnInput], settings: GenerationSettings = GenerationSettings(),
                                  cancel: Event | None = None) -> ChatResult:
        await self._preflight(messages, cancel, settings)
        client = await self.ensure_client_async(cancel=cancel)
        return await complete_chat_with_retry(client, messages, settings, cancel)

    async def ensure_client_async(self, *, cancel: Event | None = None) -> ChatClient:
        if not self.owner.is_ready(self.connection):
            await self.owner.activate(self.connection, cancel=cancel, live_test=self.live_test)
        with self._lock:
            if self._client is None:
                self._client = ChatClient(self.connection, self.owner, live_test=self.live_test)
            return self._client

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

    def complete(self, messages: Sequence[TurnInput], settings: GenerationSettings = GenerationSettings(),
                 cancel: Event | None = None) -> str:
        return self.complete_chat(messages, settings, cancel).text

    def complete_chat(self, messages: Sequence[TurnInput], settings: GenerationSettings = GenerationSettings(),
                      cancel: Event | None = None) -> ChatResult:
        return asyncio.run(self.complete_chat_async(messages, settings, cancel))

    def complete_result(self, messages: Sequence[TurnInput], settings: GenerationSettings = GenerationSettings(),
                        cancel: Event | None = None) -> Completion:
        return self.complete_chat(messages, settings, cancel).completion()

    def measure_context(self, messages: Sequence[TurnInput], *, cancel: Event | None = None,
                        settings: GenerationSettings = GenerationSettings()) -> ContextMeasurement:
        from pm.llm.context_window import measure_context
        return asyncio.run(measure_context(self.connection, messages, cancel if cancel is not None else Event(),
                                           live_test=self.live_test, settings=settings))



__all__ = ["LazyStageModel", "complete_chat_with_retry", "complete_with_retry"]
