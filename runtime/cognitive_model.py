"""Lazy, application-owned model activation for the real cognitive path.

``build_app`` never contacts a model. The application constructs one
``LazyStageModel`` and hands it to every real workflow. The first completion
activates the configured connection on the calling module thread; later
completions reuse it. Activation is therefore explicit and observable, and no
model switch can happen while another completion is in flight.

The retry rule matches checkpoint one: a reasoning backend can return empty
content for a small completion budget. The budget defaults high, and an empty
response is retried at most once. There is no generation deadline and no
endless reprompt loop.
"""
from __future__ import annotations

import asyncio
from threading import Lock

from runtime.model_activation import ChatClient, ModelActivation
from runtime.model_io import ModelError
from runtime.models import Connection

DEFAULT_MAX_TOKENS = 8192


async def complete_with_retry(
    client: ChatClient,
    messages: list[dict],
    generation: dict,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """One completion with a bounded empty-content retry."""
    generation.setdefault("max_tokens", max_tokens)
    last: ModelError | None = None
    for attempt in range(2):
        try:
            result = await client.complete(messages, **generation)
        except ModelError as error:
            if "no content" not in str(error) or attempt == 1:
                raise
            last = error
            continue
        return result.choices[0].message.content or ""
    raise last if last is not None else ModelError("The model returned no content.")


class LazyStageModel:
    """Activate a configured connection on first use, then serve completions."""

    def __init__(
        self,
        connection: Connection,
        owner: ModelActivation,
        *,
        live_test: bool = False,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.connection = connection
        self.owner = owner
        self.live_test = live_test
        self.max_tokens = max_tokens
        self._lock = Lock()
        self._client: ChatClient | None = None

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

    def ensure_client(self) -> ChatClient:
        """Activate and cache the client. Safe to call from a module thread."""
        with self._lock:
            if self._client is None:
                asyncio.run(self.owner.activate(self.connection, live_test=self.live_test))
                self._client = ChatClient(self.connection, self.owner, live_test=self.live_test)
            return self._client

    def complete(self, messages: list[dict], **generation) -> str:
        """Synchronous completion for Python-role stages."""
        client = self.ensure_client()
        return asyncio.run(complete_with_retry(client, messages, generation, max_tokens=self.max_tokens))

    def build_async_tool(self):
        """Return an async completion callable for roles that run their own loop.

        Activation happens now, when the tool is built on the module thread,
        so the tool body only awaits an already-activated client.
        """
        client = self.ensure_client()

        async def complete_async(messages: list[dict], **generation) -> str:
            return await complete_with_retry(client, messages, generation, max_tokens=self.max_tokens)

        return complete_async

    def build_tool_completion(self):
        """Return an async tool-calling completion that preserves tool_calls.

        Assistant dialogue needs the full assistant message, including any
        requested tool calls. Activation happens when the tool is built.
        """
        client = self.ensure_client()

        async def complete_tool(messages: list[dict], tools=None, **generation) -> dict:
            generation.setdefault("max_tokens", self.max_tokens)
            if tools is not None:
                generation["tools"] = tools
            last: ModelError | None = None
            for attempt in range(2):
                try:
                    result = await client.complete(messages, **generation)
                    return result.model_dump(mode="json")
                except ModelError as error:
                    if "no content" not in str(error) or attempt == 1:
                        raise
                    last = error
            raise last if last is not None else ModelError("The model returned no content.")

        return complete_tool


__all__ = ["DEFAULT_MAX_TOKENS", "LazyStageModel", "complete_with_retry"]
