"""Prompt log files. No model endpoint and no real prompt directory.

Each check writes into a temporary directory, so the operator's
``~/.pm/private-machine`` is never touched by the foundation suite.
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pm.llm.config import resolve_connection
from pm.llm.prompt_log import (
    MAX_LINE, finish_prompt_log, log_directory, start_prompt_log, wrap_text, write_prompt_log,
)
import json


def example_connection():
    return resolve_connection(base_url="http://127.0.0.1:8080/v1", model="qwen", api_key="")


class WrapTests(unittest.TestCase):
    def test_long_lines_wrap_at_the_limit(self):
        lines = wrap_text("word " * 60)
        self.assertTrue(lines)
        self.assertTrue(all(len(line) <= MAX_LINE for line in lines), [len(line) for line in lines])

    def test_blank_lines_are_kept(self):
        self.assertEqual(wrap_text("a\n\nb"), ["a", "", "b"])


class WriteTests(unittest.TestCase):
    def test_prompt_exists_before_response_and_exact_payload_survives_wrapping(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = {"model": "qwen", "messages": [{"role": "user", "content": "a  b\n" + "x" * 150}],
                       "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}]}
            path = start_prompt_log(connection=example_connection(), payload=payload,
                                    route="chat/completions", request_count=1, directory=Path(directory))
            self.assertIn("status: pending", path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(path.with_suffix(".json").read_text())["payload"], payload)
            self.assertFalse(path.with_suffix(".response.json").exists())
            finish_prompt_log(path, response={"usage": {"prompt_tokens": 5, "completion_tokens": 2,
                                                       "completion_tokens_details": {"reasoning_tokens": 1},
                                                       "cost": 0.0001}}, elapsed=1)
            receipt = json.loads(path.with_suffix(".response.json").read_text())
            self.assertEqual(receipt["usage"]["completion_tokens_details"]["reasoning_tokens"], 1)
            events = [json.loads(line) for line in next(Path(directory).glob("usage-*.jsonl")).read_text().splitlines()]
            self.assertEqual([item["event"] for item in events], ["start", "end"])
            self.assertEqual(events[0]["call_id"], events[1]["call_id"])

    def test_cancelled_call_keeps_prompt_and_error_receipt(self):
        import asyncio
        with tempfile.TemporaryDirectory() as directory:
            path = start_prompt_log(connection=example_connection(), payload={"messages": []},
                                    route="chat/completions", request_count=1, directory=Path(directory))
            finish_prompt_log(path, error=asyncio.CancelledError())
            receipt = json.loads(path.with_suffix(".response.json").read_text())
            self.assertEqual(receipt["event"], "error")
            self.assertIn("CancelledError", receipt["error"])

    def test_embeddings_keep_usage_without_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = start_prompt_log(connection=example_connection(), payload={"input": ["hello"]},
                                    route="embeddings", request_count=1, directory=Path(directory))
            finish_prompt_log(path, response={"data": [{"embedding": [0.5]}], "usage": {"prompt_tokens": 1}})
            receipt = json.loads(path.with_suffix(".response.json").read_text())
            self.assertEqual(receipt["usage"]["prompt_tokens"], 1)
            self.assertNotIn("embedding", receipt["response"])

    def test_one_file_holds_metadata_and_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_prompt_log(
                connection=example_connection(),
                messages=[{"role": "system", "content": "You are Spamuel. " * 20},
                          {"role": "user", "content": "hello"}],
                generation={"temperature": 0.8},
                response={"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 30, "completion_tokens": 2, "total_tokens": 32}},
                elapsed=1.0, request_count=3, directory=Path(directory),
            )
            self.assertIsNotNone(path)
            text = path.read_text(encoding="utf-8")
            self.assertEqual(path.parent, Path(directory))
            self.assertTrue(path.name.endswith(".md"))
            self.assertIn("- model: qwen", text)
            self.assertIn("- tokens_per_second: 2.0", text)
            self.assertIn("- prompt_tokens: 30", text)
            self.assertIn("- status: ok", text)
            self.assertIn("### 1. system", text)
            self.assertIn('### 2. user', text)
            self.assertIn("hi", text)
            self.assertTrue(all(len(line) <= MAX_LINE for line in text.splitlines()))

    def test_an_error_file_records_the_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_prompt_log(connection=example_connection(),
                                    messages=[{"role": "user", "content": "hello"}],
                                    error=RuntimeError("transport down"),
                                    elapsed=0.5, directory=Path(directory))
            text = path.read_text(encoding="utf-8")
            self.assertIn("- status: error (RuntimeError)", text)
            self.assertIn("transport down", text)

    def test_logging_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {"PM_PROMPT_LOG": "0"}):
            path = write_prompt_log(connection=example_connection(),
                                    messages=[{"role": "user", "content": "hello"}],
                                    directory=Path(directory))
            self.assertIsNone(path)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_a_write_failure_does_not_raise(self):
        with tempfile.TemporaryDirectory() as directory:
            blocked = Path(directory) / "blocked"
            blocked.write_text("not a directory", encoding="utf-8")
            path = write_prompt_log(connection=example_connection(),
                                    messages=[{"role": "user", "content": "hello"}],
                                    directory=blocked)
            self.assertIsNone(path)

    def test_default_directory_is_the_private_machine_home(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PM_PROMPT_LOG_DIR", None)
            self.assertEqual(log_directory(), Path.home() / ".pm" / "private-machine")


if __name__ == "__main__":
    unittest.main()
