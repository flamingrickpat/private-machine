"""Deterministic checks for the replay harness's own bookkeeping.

The harness is test tooling, so its utilities get small in-process checks: they
are pure functions over recorded text. Nothing here substitutes for a recorded
scenario; it only proves that the recording format, the request identity, and
the divergence report behave as their contracts say.
"""
from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from pm.llm.prompt_encoding import TurnInput
from tests.replay.determinism import IdentitySequence, deterministic_identities
from tests.replay.script import (
    RecordingReader, RecordingWriter, ReplayDivergence, normalize_timing, request_text,
)


def turns(text: str) -> list[TurnInput]:
    """Return one system turn holding ``text``, as the stages would send it."""
    return [{"role": "system", "content": text}]


class TimingNormalizationTests(unittest.TestCase):
    """Measured durations must not change a request's identity."""

    def test_elapsed_fields_are_placeholders_at_every_escaping_depth(self) -> None:
        for depth in range(4):
            quote = "\\" * (2 * depth - 1) if depth else ""
            text = f'{{{quote}"active_elapsed_seconds{quote}":0.0929999999}}'
            normalized = normalize_timing(text)
            self.assertIn("<elapsed>", normalized)
            self.assertNotIn("0.0929999999", normalized)

    def test_normalization_is_idempotent_and_keeps_json_valid(self) -> None:
        payload = {"episode": {"active_elapsed_seconds": 1.25, "stalled_steps": 0}}
        text = json.dumps({"turn": json.dumps(payload)})
        once = normalize_timing(text)
        self.assertEqual(normalize_timing(once), once)
        json.loads(once)

    def test_a_name_without_a_number_is_left_alone(self) -> None:
        text = '{"active_elapsed_seconds": "unknown"}'
        self.assertEqual(normalize_timing(text), text)


class RequestIdentityTests(unittest.TestCase):
    """The request identity must describe content, not scheduling."""

    def test_cancellation_does_not_change_the_request(self) -> None:
        from threading import Event
        running = Event()
        cancelled = Event()
        cancelled.set()
        first = request_text("completion", turns("hello"), {"temperature": 0.7, "cancel": running})
        second = request_text("completion", turns("hello"), {"temperature": 0.7, "cancel": cancelled})
        self.assertEqual(first, second)

    def test_generation_options_are_part_of_the_request(self) -> None:
        first = request_text("completion", turns("hello"), {"temperature": 0.7})
        second = request_text("completion", turns("hello"), {"temperature": 1.0})
        self.assertNotEqual(first, second)

    def test_kind_separates_a_completion_from_a_measurement(self) -> None:
        first = request_text("completion", turns("hello"), {})
        second = request_text("measurement", turns("hello"), {})
        self.assertNotEqual(first, second)


class RecordingRoundTripTests(unittest.TestCase):
    """A recording serves its answers back, and refuses a different question."""

    def test_a_recorded_call_replays_exactly(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            request = request_text("completion", turns("hello"), {"temperature": 0.7})
            with RecordingWriter(path / "calls.jsonl") as writer:
                writer.append("completion", request, "recorded answer", 0.5)
            reader = RecordingReader(path / "calls.jsonl")
            self.assertEqual(reader.next_call("completion", request).response, "recorded answer")
            self.assertEqual(reader.remaining, 0)

    def test_a_changed_request_is_a_loud_divergence(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            original = request_text("completion", turns("hello"), {"temperature": 0.7})
            with RecordingWriter(path / "calls.jsonl") as writer:
                writer.append("completion", original, "recorded answer", 0.5)
            reader = RecordingReader(path / "calls.jsonl")
            changed = request_text("completion", turns("goodbye"), {"temperature": 0.7})
            with self.assertRaises(ReplayDivergence) as raised:
                reader.next_call("completion", changed)
            self.assertIn("changed since the recording", str(raised.exception))
            self.assertIn("hello", raised.exception.expected)

    def test_a_different_call_kind_is_a_loud_divergence(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            request = request_text("completion", turns("hello"), {})
            with RecordingWriter(path / "calls.jsonl") as writer:
                writer.append("completion", request, "recorded answer", 0.5)
            reader = RecordingReader(path / "calls.jsonl")
            with self.assertRaisesRegex(ReplayDivergence, "holds a completion"):
                reader.next_call("measurement", request)

    def test_exhausted_recording_is_a_loud_divergence(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            request = request_text("completion", turns("hello"), {})
            with RecordingWriter(path / "calls.jsonl") as writer:
                writer.append("completion", request, "recorded answer", 0.5)
            reader = RecordingReader(path / "calls.jsonl")
            reader.next_call("completion", request)
            with self.assertRaisesRegex(ReplayDivergence, "more calls than the recording"):
                reader.next_call("completion", request)


class DeterministicIdentityTests(unittest.TestCase):
    """Identity allocation repeats and is restored afterwards."""

    def test_the_same_seed_draws_the_same_identities(self) -> None:
        first = IdentitySequence("scenario:1")
        second = IdentitySequence("scenario:1")
        self.assertEqual([first.next_uuid() for _ in range(3)], [second.next_uuid() for _ in range(3)])

    def test_different_seeds_draw_different_identities(self) -> None:
        self.assertNotEqual(IdentitySequence("a").next_uuid(), IdentitySequence("b").next_uuid())

    def test_the_real_identity_source_is_restored(self) -> None:
        original = uuid.uuid4
        expected = IdentitySequence("scenario:1").next_uuid()
        with deterministic_identities("scenario:1"):
            self.assertEqual(uuid.uuid4(), expected)
        self.assertIs(uuid.uuid4, original)


if __name__ == "__main__":
    unittest.main()
