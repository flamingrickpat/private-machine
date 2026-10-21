"""Closed time ranges of the temporal hierarchy."""
import unittest
from datetime import datetime

from pm.memory.reconsolidation.temporal import (LEVEL_DAY, LEVEL_MONTH, LEVEL_TIME_OF_DAY, LEVEL_WEEK, LEVEL_YEAR,
                                                closed_ranges, logical_day_start, ranges_of)


class TemporalTests(unittest.TestCase):
    def test_keys_match_the_old_format(self):
        keys = {item.level: item.key for item in ranges_of(datetime(2026, 5, 1, 20, 10))}
        self.assertEqual(keys, {LEVEL_TIME_OF_DAY: "2026-05-01-EVENING", LEVEL_DAY: "2026-05-01",
                                LEVEL_WEEK: "2026-W17", LEVEL_MONTH: "2026-05", LEVEL_YEAR: "2026"})

    def test_a_late_night_belongs_to_the_day_before(self):
        moment = datetime(2026, 5, 2, 3, 0)
        self.assertEqual(logical_day_start(moment), datetime(2026, 5, 1, 5, 0))
        keys = {item.level: item for item in ranges_of(moment)}
        self.assertEqual(keys[LEVEL_TIME_OF_DAY].key, "2026-05-01-NIGHT")
        self.assertEqual(keys[LEVEL_TIME_OF_DAY].start, datetime(2026, 5, 1, 22, 0))
        self.assertEqual(keys[LEVEL_TIME_OF_DAY].end, datetime(2026, 5, 2, 5, 0))
        self.assertEqual(keys[LEVEL_DAY].key, "2026-05-01")

    def test_one_night_key_per_logical_day(self):
        early = {item.level: item.key for item in ranges_of(datetime(2026, 5, 2, 1, 0))}
        late = {item.level: item.key for item in ranges_of(datetime(2026, 5, 2, 23, 0))}
        self.assertNotEqual(early[LEVEL_TIME_OF_DAY], late[LEVEL_TIME_OF_DAY])

    def test_month_and_year_bounds_start_at_five(self):
        keys = {item.level: item for item in ranges_of(datetime(2026, 5, 31, 23, 0))}
        self.assertEqual(keys[LEVEL_MONTH].start, datetime(2026, 5, 1, 5, 0))
        self.assertEqual(keys[LEVEL_MONTH].end, datetime(2026, 6, 1, 5, 0))
        self.assertEqual(keys[LEVEL_YEAR].start, datetime(2026, 1, 1, 5, 0))
        self.assertEqual(keys[LEVEL_YEAR].end, datetime(2027, 1, 1, 5, 0))

    def test_only_closed_ranges_finest_first(self):
        moments = [datetime(2026, 5, 1, 20, 10), datetime(2026, 5, 2, 9, 0)]
        ranges = closed_ranges(moments, now=datetime(2026, 5, 2, 13, 0))
        self.assertEqual([item.key for item in ranges], ["2026-05-01-EVENING", "2026-05-02-MORNING", "2026-05-01"])
        self.assertTrue(all(item.end <= datetime(2026, 5, 2, 13, 0) for item in ranges))


if __name__ == "__main__":
    unittest.main()
