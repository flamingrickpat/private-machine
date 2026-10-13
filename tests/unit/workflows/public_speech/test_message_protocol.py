"""Classify the actual A10 parser rejection captured from local chat.

These checks establish HTTP error classification. The A09 citation checks of
the old claim review were removed with that review. The hallucination review
(``speech_review``) has its own tests.
"""
import json
from pathlib import Path
import unittest

import httpx

from pm.llm.http_client import rejected_message_protocol
from workflows.dialogue_assistant.steps import validate_public_speech


class MessageProtocolTests(unittest.TestCase):
    def test_captured_parser_rejection_is_permanent(self):
        captured = json.loads((Path(__file__).resolve().parents[4] / "test_data/captures/alpha/a10-message-protocol-error.json").read_text())
        response = httpx.Response(captured["status"], json=captured["body"])
        self.assertTrue(rejected_message_protocol(response))

    def test_other_server_failures_remain_transient(self):
        self.assertFalse(rejected_message_protocol(httpx.Response(500, content=b"service unavailable")))
        self.assertFalse(rejected_message_protocol(httpx.Response(503, json={"error": {"message": "busy"}})))
        self.assertFalse(rejected_message_protocol(httpx.Response(200, json={"choices": []})))

    def test_captured_tool_markup_cannot_be_published_as_speech(self):
        captured = (Path(__file__).resolve().parents[4] / "test_data/captures/alpha/a10-assistant-tool-markup.txt").read_text()
        with self.assertRaisesRegex(ValueError, "tool protocol"):
            validate_public_speech(captured)
