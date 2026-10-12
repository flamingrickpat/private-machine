"""Check token-budget configuration and real transport boundaries."""
import asyncio
import json
from pathlib import Path
import socket
from threading import Event
import unittest

from pydantic import ValidationError

from runtime.model_context import measure_context
from runtime.model_io import ModelError
from runtime.models import ConnectionConfig


class ModelContextTests(unittest.TestCase):
    def test_token_fraction_survives_connection_resolution(self):
        fixture = json.loads((Path(__file__).resolve().parents[1] / "test_data/captures/alpha/a13-context-token-budget.json").read_text(encoding="utf-8"))
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

    def test_closed_measurement_endpoint_reports_transport_failure(self):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            connection = ConnectionConfig(base_url=f"http://127.0.0.1:{reserved.getsockname()[1]}/v1",
                                          model="qwen", context_window=98304).resolve()
            with self.assertRaisesRegex(ModelError, "Context measurement failed"):
                asyncio.run(measure_context(connection, [], Event()))

    def test_cancelled_measurement_does_not_wait_for_the_endpoint(self):
        cancel = Event()
        cancel.set()
        connection = ConnectionConfig(base_url="http://127.0.0.1:1/v1", model="qwen", context_window=98304).resolve()
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(measure_context(connection, [], cancel))
