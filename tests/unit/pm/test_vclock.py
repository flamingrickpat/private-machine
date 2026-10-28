"""UTC conversion and semantic time stay separate from real diagnostics."""
from datetime import datetime, timezone
import unittest

from pm import vclock
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature


class VirtualClockTests(unittest.TestCase):
    def tearDown(self):
        vclock.install(None)

    def test_offset_conversion_and_forward_only_elapsed_time(self):
        clock = vclock.VirtualClock(datetime(2026, 1, 1, 1), utc_offset_hours=1)
        vclock.install(clock)
        self.assertEqual(vclock.now_aware(), datetime(2026, 1, 1, tzinfo=timezone.utc))
        clock.set(datetime(2026, 1, 1, 2))
        self.assertEqual(vclock.monotonic(), 3600)
        with self.assertRaises(ValueError):
            clock.set(datetime(2026, 1, 1, 1))

    def test_record_world_time_is_virtual_and_creation_time_is_real(self):
        vclock.install(vclock.VirtualClock(datetime(2000, 1, 1)))
        record = Feature(content="virtual event", feature_type=FeatureType.WorldEvent)
        self.assertEqual(record.timestamp_world_begin, datetime(2000, 1, 1))
        self.assertGreater(record.timestamp_creation.year, 2000)
        self.assertLessEqual(vclock.wait_cap(None), .02)
