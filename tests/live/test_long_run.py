"""Long-run checks: overnight messages, memory growth, repeated lines.

The helper tests and the stub run need no model. The live run needs a model
endpoint and real time, so it only runs with ``PM_LONGRUN_LIVE=1``; set
``PM_LONGRUN_HOURS`` (default 24) and ``PM_LONGRUN_CARD`` to change it.
"""
from __future__ import annotations

import os
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from tests.live import long_run as lr


class HelperTests(unittest.TestCase):
    def test_sleep_window_wraps_midnight(self):
        self.assertTrue(lr.is_asleep(datetime(2030, 1, 7, 23, 30), 23, 7))
        self.assertTrue(lr.is_asleep(datetime(2030, 1, 7, 3, 0), 23, 7))
        self.assertFalse(lr.is_asleep(datetime(2030, 1, 7, 7, 0), 23, 7))
        self.assertFalse(lr.is_asleep(datetime(2030, 1, 7, 12, 0), 23, 7))

    def test_repeat_stats_counts_exact_and_near(self):
        stats = lr.repeat_stats(["Good morning!", "good morning", "Good mornin'", "what a day", "ok then"])
        self.assertEqual(stats["lines"], 5)
        self.assertEqual(stats["exact"], 1)
        self.assertEqual(stats["near"], 1)

    def test_rss_slope(self):
        self.assertAlmostEqual(lr.rss_slope([(0, 100), (1, 110), (2, 120), (3, 130)], 0), 10.0)
        self.assertAlmostEqual(lr.rss_slope([(0, 500), (1, 100), (2, 100), (3, 100)], 1), 0.0)


class StubRunTests(unittest.TestCase):
    def test_stub_hour_writes_transcript_and_report(self):
        with TemporaryDirectory() as tmp:
            result = lr.run(lr.Params(hours=1, stub=True, out=Path(tmp), max_rss_mb_per_hour=1e9,
                                      max_repeat_fraction=1.0))
            self.assertTrue((Path(tmp) / "transcript.jsonl").is_file())
            self.assertTrue((Path(tmp) / "report.json").is_file())
            self.assertEqual(result.failures, [])


@unittest.skipUnless(os.environ.get("PM_LONGRUN_LIVE") == "1",
                     "Set PM_LONGRUN_LIVE=1 to run the live long-run test (model endpoint, long).")
class LiveLongRunTests(unittest.TestCase):
    def test_long_run(self):
        card = Path(os.environ.get("PM_LONGRUN_CARD", lr.DEFAULT_CARD))
        out = Path(".runtime/longrun") / datetime.now().strftime("%Y%m%dT%H%M%S")
        result = lr.run(lr.Params(hours=float(os.environ.get("PM_LONGRUN_HOURS", "24")),
                                  card=card, out=out))
        print(result.report)
        self.assertEqual(result.failures, [], result.failures)


if __name__ == "__main__":
    unittest.main()
