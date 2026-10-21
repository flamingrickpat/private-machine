"""Real subprocess and cancellation tests, separate from live generation."""
import asyncio
import os
import socket
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from threading import Event

from pm.llm.activation import ModelActivation, run_startup
from pm.llm.http_client import JsonClient, ModelError
from pm.llm.config import resolve_connection


class ActivationTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_chat_attempt_retains_prompt_without_credentials(self):
        import json
        with tempfile.TemporaryDirectory() as directory, socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            connection = resolve_connection(base_url=f"http://127.0.0.1:{reserved.getsockname()[1]}/v1",
                                            model="qwen", api_key="private-credential-marker")
            client = JsonClient(connection, live_test=True)
            payload = {"model": "qwen", "messages": [{"role": "user", "content": "hello"}]}
            with mock.patch.dict(os.environ, {"PM_PROMPT_LOG_DIR": directory, "PM_PROMPT_LOG": "1"}):
                with self.assertRaises(ModelError):
                    await client.request("POST", "chat/completions", payload=payload)
            prompt = next(Path(directory).glob("*.md"))
            text = prompt.read_text(encoding="utf-8")
            self.assertIn("- model: qwen", text)
            self.assertIn("### 1. user\n\n  hello", text)
            events = [json.loads(line) for line in next(Path(directory).glob("usage-*.jsonl")).read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual([event["event"] for event in events], ["start", "error"])
            self.assertEqual(events[-1]["call_id"], prompt.stem)
            self.assertFalse(list(Path(directory).glob("*.json")))
            self.assertNotIn("private-credential-marker", "".join(
                path.read_text() for path in Path(directory).iterdir()))

    async def test_real_finite_script_runs_and_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "start.py"
            script.write_text("from pathlib import Path\nPath('started.txt').write_text('started')\n", encoding="utf-8")
            await run_startup(script)
            self.assertEqual((script.parent / "started.txt").read_text(), "started")

    async def test_native_shell_startup_script(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / ("start.ps1" if os.name == "nt" else "start.sh")
            body = "Set-Content -LiteralPath 'started.txt' -Value 'started'\nexit 0\n" if os.name == "nt" else "printf started > started.txt\nexit 0\n"
            script.write_text(body, encoding="utf-8")
            await run_startup(script)
            self.assertEqual((script.parent / "started.txt").read_text().strip(), "started")

    async def test_nonzero_script_exit_is_not_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "start.py"
            script.write_text("print('private-output-marker')\nraise SystemExit(7)\n", encoding="utf-8")
            with self.assertRaises(ModelError) as error:
                await run_startup(script)
            self.assertIn("exit 7", str(error.exception))
            self.assertNotIn("private-output-marker", str(error.exception))

    async def test_cancellation_stops_owned_script(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "wait.py"
            script.write_text("import time\ntime.sleep(3600)\n", encoding="utf-8")
            cancel = Event()
            asyncio.get_running_loop().call_later(0.1, cancel.set)
            with self.assertRaises(asyncio.CancelledError):
                await run_startup(script, cancel)

    async def test_readiness_wait_is_cancellable(self):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            connection = resolve_connection(base_url=f"http://127.0.0.1:{reserved.getsockname()[1]}/v1", model="qwen", api_key="")
            owner = ModelActivation()
            cancel = Event()
            asyncio.get_running_loop().call_later(0.1, cancel.set)
            with self.assertRaises(asyncio.CancelledError):
                await owner.activate(connection, cancel=cancel)
            self.assertFalse(owner._switching)
            with self.assertRaises(ModelError):
                with owner.use(connection):
                    self.fail("A cancelled activation cannot admit generation.")

    async def test_live_test_mode_fails_without_retry_or_credential_echo(self):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            connection = resolve_connection(base_url=f"http://127.0.0.1:{reserved.getsockname()[1]}/v1", model="qwen", api_key="private-credential-marker")
            client = JsonClient(connection, live_test=True)
            with self.assertRaises(ModelError) as error:
                await client.request("GET", "models")
            self.assertNotIn("private-credential-marker", str(error.exception))
            self.assertEqual(client.request_count, 1)

    async def test_cancelled_request_never_sends(self):
        connection = resolve_connection(base_url="http://127.0.0.1:1/v1", model="qwen", api_key="")
        client = JsonClient(connection, live_test=True)
        cancel = Event()
        cancel.set()
        with self.assertRaises(asyncio.CancelledError):
            await client.request("GET", "models", cancel=cancel)
        self.assertEqual(client.request_count, 0)


class ProviderGenerationTests(unittest.TestCase):
    """The thinking switch reaches OpenRouter in its own field."""

    def _connection(self, url):
        from pm.llm.config import Connection
        return Connection(base_url=url, model="m")

    def test_openrouter_receives_reasoning_enabled(self):
        from pm.llm.activation import provider_generation
        generation = {"temperature": 0.1, "chat_template_kwargs": {"enable_thinking": False}}
        result = provider_generation(self._connection("https://openrouter.ai/api/v1"), generation)
        self.assertEqual(result["reasoning"], {"enabled": False})
        self.assertNotIn("reasoning", generation)

    def test_workflow_thinking_option_is_neutral(self):
        from pm.llm.activation import provider_generation
        generation = {"enable_thinking": False, "temperature": 0.2}
        hosted = provider_generation(self._connection("https://openrouter.ai/api/v1"), generation)
        local = provider_generation(self._connection("http://localhost/v1"), generation)
        self.assertEqual(hosted["reasoning"], {"enabled": False})
        self.assertEqual(local["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("enable_thinking", hosted)
        self.assertNotIn("chat_template_kwargs", hosted)
        self.assertNotIn("enable_thinking", local)
        self.assertEqual(generation, {"enable_thinking": False, "temperature": 0.2})

    def test_local_host_and_explicit_reasoning_stay_unchanged(self):
        from pm.llm.activation import provider_generation
        generation = {"chat_template_kwargs": {"enable_thinking": False}}
        local = provider_generation(self._connection("http://127.0.0.1:8080/v1"), generation)
        self.assertEqual(local, generation)
        explicit = {**generation, "reasoning": {"effort": "low"}}
        hosted = provider_generation(self._connection("https://openrouter.ai/api/v1"), explicit)
        self.assertEqual(hosted["reasoning"], {"effort": "low"})


if __name__ == "__main__":
    unittest.main()
