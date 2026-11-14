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
    ProviderUsage, RecordedCall, RecordingReader, RecordingWriter, ReplayDivergence, normalize_timing, request_text,
    total_usage,
)

#: One provider receipt as a recorded completion carries it.
USAGE = ProviderUsage(prompt_tokens=10, completion_tokens=2, total_tokens=12, cost=0.001)


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
                writer.append("completion", request, "recorded answer", 0.5, usage=USAGE)
            reader = RecordingReader(path / "calls.jsonl")
            self.assertEqual(reader.next_call("completion", request).response, "recorded answer")
            self.assertEqual(reader.remaining, 0)

    def test_a_changed_request_is_a_loud_divergence(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            original = request_text("completion", turns("hello"), {"temperature": 0.7})
            with RecordingWriter(path / "calls.jsonl") as writer:
                writer.append("completion", original, "recorded answer", 0.5, usage=USAGE)
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
                writer.append("completion", request, "recorded answer", 0.5, usage=USAGE)
            reader = RecordingReader(path / "calls.jsonl")
            with self.assertRaisesRegex(ReplayDivergence, "holds a completion"):
                reader.next_call("measurement", request)

    def test_exhausted_recording_is_a_loud_divergence(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            request = request_text("completion", turns("hello"), {})
            with RecordingWriter(path / "calls.jsonl") as writer:
                writer.append("completion", request, "recorded answer", 0.5, usage=USAGE)
            reader = RecordingReader(path / "calls.jsonl")
            reader.next_call("completion", request)
            with self.assertRaisesRegex(ReplayDivergence, "more calls than the recording"):
                reader.next_call("completion", request)


class ProviderUsageTests(unittest.TestCase):
    """A recording states what it cost, and never invents an absent receipt."""

    def test_completions_keep_their_receipt_and_the_sample_total(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory)
            request = request_text("completion", turns("hello"), {})
            measured = request_text("measurement", turns("hello"), {})
            with RecordingWriter(path / "calls.jsonl") as writer:
                writer.append("measurement", measured, "{}", 0.1)
                writer.append("completion", request, "first", 0.5, usage=USAGE)
                writer.append("completion", request, "second", 0.5,
                              usage=ProviderUsage.model_validate({"prompt_tokens": 30, "completion_tokens": 5,
                                                                  "cost": 0.002, "cost_details": {"upstream": 0.002}}))
                total = writer.usage_total()
            self.assertEqual(total.completions, 2)
            self.assertEqual(total.prompt_tokens, 40)
            self.assertEqual(total.completion_tokens, 7)
            self.assertAlmostEqual(total.reported_cost_usd or 0.0, 0.003)
            rows = list(RecordingReader(path / "calls.jsonl"))
            self.assertIsNone(rows[0].usage)
            self.assertEqual(rows[1].usage, USAGE)
            # A field this harness does not name stays as the provider sent it.
            assert rows[2].usage is not None
            self.assertEqual(rows[2].usage.model_extra, {"cost_details": {"upstream": 0.002}})

    def test_an_unreported_cost_is_not_summed_as_zero(self) -> None:
        total = total_usage([USAGE, ProviderUsage(prompt_tokens=30, completion_tokens=5)])
        self.assertEqual(total.prompt_tokens, 40)
        self.assertIsNone(total.reported_cost_usd)

    def test_only_completions_carry_a_receipt(self) -> None:
        with TemporaryDirectory() as directory:
            request = request_text("completion", turns("hello"), {})
            with RecordingWriter(Path(directory) / "calls.jsonl") as writer:
                with self.assertRaisesRegex(ValueError, "Exactly the recorded completions"):
                    writer.append("completion", request, "answer", 0.5)
                with self.assertRaisesRegex(ValueError, "Exactly the recorded completions"):
                    writer.append("measurement", request, "{}", 0.5, usage=USAGE)

    def test_a_recording_from_before_usage_was_retained_still_loads(self) -> None:
        request = request_text("completion", turns("hello"), {})
        earlier_row = {"index": 0, "kind": "completion", "digest": "0" * 32, "request": request,
                       "response": "answer", "elapsed_seconds": 0.5}
        self.assertIsNone(RecordedCall.model_validate(earlier_row).usage)


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
