"""Real subprocess and cancellation tests, separate from live generation."""
import asyncio
import os
import socket
import tempfile
import unittest
from pathlib import Path
from threading import Event

from runtime.model_activation import ModelActivation, run_startup
from runtime.model_io import JsonClient, ModelError
from runtime.models import resolve_connection


class ActivationTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
