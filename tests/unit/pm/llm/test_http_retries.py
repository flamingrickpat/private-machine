"""Exercise retry delivery through real loopback HTTP, without model inference.

The server returns status fixtures and a captured provider error. These tests
check waits, explicit retry bounds and cancellation, not companion quality.
"""
import asyncio
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
import unittest

from pm.llm.config import resolve_connection
from pm.llm.http_client import JsonClient, ModelError, ProviderCompletionError
from pm.llm.activation import ModelActivation
from pm.llm.stage_model import LazyStageModel


class HttpRetryTests(unittest.IsolatedAsyncioTestCase):
    def serve(self, statuses: list[int], body: dict | None = None, *, response_gate: Event | None = None):
        """Own a loopback status/body fixture and count requests until cleanup."""
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                status = statuses[min(len(requests), len(statuses) - 1)]
                requests.append(self.path)
                if response_gate is not None:
                    response_gate.wait(timeout=2)
                encoded = json.dumps(body if body is not None else {"received": True}).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                try:
                    self.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError):
                    # The readiness test cancels before its delayed response.
                    pass

            do_POST = do_GET

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        connection = resolve_connection(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="status-fixture", api_key="")
        return connection, requests

    async def test_rate_limit_and_server_error_recover_with_same_route(self):
        connection, requests = self.serve([429, 503, 200])
        client = JsonClient(connection, retry_delay_seconds=0)
        self.assertEqual(await client.request("GET", "health"), {"received": True})
        self.assertEqual(requests, ["/v1/health"] * 3)

    async def test_five_retries_stop_after_six_attempts(self):
        connection, requests = self.serve([503])
        client = JsonClient(connection, transient_retries=5, retry_delay_seconds=0)
        with self.assertRaises(ModelError):
            await client.request("GET", "health")
        self.assertEqual(len(requests), 6)

    async def test_bad_request_is_not_retried(self):
        connection, requests = self.serve([400, 200])
        client = JsonClient(connection, retry_delay_seconds=0)
        with self.assertRaises(ModelError):
            await client.request("GET", "health")
        self.assertEqual(len(requests), 1)

    async def test_production_waits_beyond_the_old_five_retry_limit(self):
        connection, requests = self.serve([429] * 7 + [200])
        client = JsonClient(connection, retry_delay_seconds=0)
        self.assertIsNone(client.transient_retries)
        self.assertEqual(await client.request("GET", "health"), {"received": True})
        self.assertEqual(len(requests), 8)

    async def test_captured_http_success_with_partial_provider_error_is_not_a_completion(self):
        path = Path("test_data/captures/alpha/provider-partial-completion-error.json")
        capture = json.loads(path.read_text(encoding="utf-8"))
        connection, requests = self.serve([200], capture["response"])
        client = JsonClient(connection, transient_retries=5, retry_delay_seconds=0)
        with self.assertRaisesRegex(ProviderCompletionError, "No partial completion"):
            await client.request("POST", "chat/completions", payload={})
        self.assertEqual(len(requests), 1)

    async def test_completed_refusal_remains_a_model_response(self):
        body = {"choices": [{"finish_reason": "stop", "message": {
            "content": "I cannot do that.", "refusal": "I cannot do that."}}]}
        connection, requests = self.serve([200], body)
        self.assertEqual(await JsonClient(connection, transient_retries=0).request("POST", "chat/completions", payload={}), body)
        self.assertEqual(len(requests), 1)

    async def test_lazy_activation_wait_is_cancellable(self):
        gate = Event()
        connection, requests = self.serve([429], response_gate=gate)
        # The delayed response fails directly if cancellation forwarding breaks.
        # A regression must not leave a thread in an unlimited retry loop.
        model = LazyStageModel(connection, ModelActivation(), live_test=True)
        cancel = Event()
        task = asyncio.create_task(asyncio.to_thread(model.complete, [], cancel=cancel))
        try:
            while not requests and not task.done():
                await asyncio.sleep(0.01)
            cancel.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(requests, ["/v1/models"])
            self.assertFalse(model.activated)
        finally:
            gate.set()

    async def test_cancel_interrupts_the_thirty_second_pause(self):
        connection, requests = self.serve([429])
        client = JsonClient(connection)
        cancel = Event()
        started = asyncio.get_running_loop().time()
        task = asyncio.create_task(client.request("GET", "health", cancel=cancel))
        while not requests:
            await asyncio.sleep(0.01)
        # Wait until the response reaches the client, then cancel its pause.
        await asyncio.sleep(0.1)
        cancel.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        self.assertEqual(len(requests), 1)
        self.assertLess(asyncio.get_running_loop().time() - started, 2)
