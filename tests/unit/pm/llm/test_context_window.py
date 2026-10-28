"""Check token-budget configuration and real transport boundaries."""
import asyncio
import json
from pathlib import Path
import socket
from threading import Event
import unittest

from pydantic import ValidationError

from pm.llm.context_window import measure_context
from pm.llm.http_client import ModelError
from pm.llm.config import ConnectionConfig


class ModelContextTests(unittest.TestCase):
    def test_token_fraction_survives_connection_resolution(self):
        fixture = json.loads((Path(__file__).resolve().parents[4] / "test_data/captures/alpha/a13-context-token-budget.json").read_text(encoding="utf-8"))
        self.assertGreater(fixture["original_input_characters"], 120000)
        self.assertEqual(fixture["measurement"]["input_tokens"], 50160)
        self.assertEqual(fixture["measurement"]["context_window"], 131072)
        self.assertTrue(fixture["fits"])
        for capacity, expected in ((98304, 78643), (131072, 104857), (186624, 149299), (1000000, 800000)):
            config = ConnectionConfig(base_url="http://localhost/v1", model="qwen", context_window=capacity)
            connection = config.resolve()
            self.assertEqual(connection.context_window, capacity)
            self.assertEqual(int(connection.context_window * connection.context_input_fraction), expected)
            self.assertLess(fixture["measurement"]["input_tokens"], expected)
        tuned = ConnectionConfig(base_url="http://localhost/v1", model="qwen", context_input_fraction=0.7)
        self.assertEqual(tuned.resolve().context_input_fraction, 0.7)

    def test_fraction_rejects_strings_booleans_and_missing_response_space(self):
        for fraction in (True, False, "0.8", 0, 1, -0.1, 1.1):
            with self.subTest(fraction=fraction), self.assertRaises(ValidationError):
                ConnectionConfig(base_url="http://localhost/v1", model="qwen", context_input_fraction=fraction)

    def test_configured_capacity_estimates_without_a_running_endpoint(self):
        connection = ConnectionConfig(base_url="http://127.0.0.1:1/v1", model="qwen", context_window=98304).resolve()
        receipt = asyncio.run(measure_context(connection, [], Event(), live_test=True))
        self.assertEqual(receipt["context_window"], 98304)
        self.assertEqual(receipt["input_budget_tokens"], 78643)
        self.assertFalse(receipt["measurement_complete"])
        self.assertIn("tiktoken.conservative", receipt["measurement"])

    def test_missing_capacity_still_reports_a_discovery_failure(self):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            connection = ConnectionConfig(base_url=f"http://127.0.0.1:{reserved.getsockname()[1]}/v1", model="qwen").resolve()
            with self.assertRaisesRegex(ModelError, "Serving context discovery failed"):
                asyncio.run(measure_context(connection, [], Event(), live_test=True))

    def test_tool_schemas_occupy_context_without_mutating_the_original_turns(self):
        from pm.context_render import Role, Turn
        connection = ConnectionConfig(base_url="http://127.0.0.1:1/v1", model="qwen", context_window=98304).resolve()
        turns = [Turn(Role.User, "Read the referenced record.")]
        plain = asyncio.run(measure_context(connection, turns, Event(), live_test=True))
        tools = [{"type": "function", "function": {"name": "read_record", "description": "Read an exact knoxel ID.",
                 "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}}}}]
        with_tools = asyncio.run(measure_context(connection, turns, Event(), tools=tools, live_test=True))
        self.assertGreater(with_tools["input_tokens"], plain["input_tokens"])
        self.assertEqual(len(turns), 1)

    def test_cancelled_measurement_does_not_wait_for_the_endpoint(self):
        cancel = Event()
        cancel.set()
        connection = ConnectionConfig(base_url="http://127.0.0.1:1/v1", model="qwen", context_window=98304).resolve()
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(measure_context(connection, [], cancel))
