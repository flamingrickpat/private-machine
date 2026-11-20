"""Deterministic receipt regressions for overlapping inputs and missing provider costs."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from simulation.baseline_metrics import measure_run


class BaselineMetricsTests(unittest.TestCase):
    def test_waits_start_at_the_first_unanswered_input_for_each_participant(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state.json").write_text(
                '{"timeline":['
                '{"kind":"actor","target":"one","wall_at":"2026-06-01T00:00:00Z"},'
                '{"kind":"actor","target":"two","wall_at":"2026-06-01T00:00:02Z"},'
                '{"kind":"actor","target":"one","wall_at":"2026-06-01T00:00:05Z"},'
                '{"kind":"speech","source":"one","wall_at":"2026-06-01T00:00:12Z"}]}',
                encoding="utf-8",
            )
            attempt = root / "attempts/1"
            attempt.mkdir(parents=True)
            (attempt / "one-endpoint.json").write_text('{"pid":123}', encoding="utf-8")
            logs = root / "participants/one/logs/prompts"
            logs.mkdir(parents=True)
            (logs / "usage-123.jsonl").write_text(
                '{"call_id":"a","event":"start"}\n'
                '{"call_id":"a","event":"end","usage":{"cost":0.25}}\n'
                '{"call_id":"b","event":"start"}\n',
                encoding="utf-8",
            )
            metrics = measure_run(root)
            self.assertEqual(metrics.visible_exchanges, 1)
            self.assertEqual(metrics.pending_participants, 1)
            self.assertEqual(metrics.worst_wait_seconds, 12)
            self.assertEqual(metrics.participant_model_calls, 2)
            self.assertEqual(metrics.participant_calls_without_cost, 1)
            self.assertEqual(metrics.reported_participant_cost_usd, 0.25)
            self.assertIsNone(metrics.p95_minecraft_tool_seconds)
