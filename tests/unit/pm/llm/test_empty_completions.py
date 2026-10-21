"""Keep external empty output separate from transport and programming errors."""
import asyncio
import unittest
from unittest.mock import AsyncMock

from pm.llm.activation import ChatClient, ModelActivation
from pm.llm.config import Connection
from pm.llm.http_client import EmptyCompletionError, ModelError, ProviderCompletionError
from pm.llm.stage_model import complete_with_retry


class EmptyCompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_declared_empty_output_receives_a_retry(self):
        complete = AsyncMock(side_effect=[EmptyCompletionError("empty"), type("Result", (), {"text": "hello"})()])
        result = await complete_with_retry(type("Client", (), {"complete": complete})(), [], {})
        self.assertEqual(result, "hello")
        self.assertEqual(complete.await_count, 2)

    async def test_other_errors_keep_their_identity_and_fail_once(self):
        for error in (ModelError("no content in an unrelated error"), ProviderCompletionError("filtered"),
                      TypeError("bug"), KeyError("bug"), asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                complete = AsyncMock(side_effect=error)
                with self.assertRaises(type(error)) as caught:
                    await complete_with_retry(type("Client", (), {"complete": complete})(), [], {})
                self.assertIs(caught.exception, error)
                self.assertEqual(complete.await_count, 1)

    async def test_whitespace_is_empty_but_tool_calls_are_valid_output(self):
        connection = Connection(base_url="http://127.0.0.1:1/v1", model="test")
        owner = ModelActivation()
        owner._ready[connection.base_url] = connection.model
        client = ChatClient(connection, owner)
        client.client.request = AsyncMock(return_value={"model": "test", "choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": " \n\t"}}]})
        with self.assertRaises(EmptyCompletionError):
            await client.complete([{"role": "user", "content": "hello"}])
        client.client.request.return_value["choices"][0]["message"]["tool_calls"] = [
            {"id": "read", "type": "function", "function": {"name": "read", "arguments": "{}"}}]
        result = await client.complete([{"role": "user", "content": "hello"}])
        self.assertTrue(result.choices[0].message.tool_calls)
