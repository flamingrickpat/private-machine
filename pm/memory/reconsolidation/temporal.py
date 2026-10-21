"""Closed time ranges of the temporal memory hierarchy, in pure Python.

Port of ``pm/utils/time_hierarchy_utils.py`` and ``get_temporal_key`` of
``private-machine-head``. A logical day runs from 05:00 to 05:00 of the next
calendar day, so a late night belongs to the evening before it. Times are
naive UTC, like the record timestamps.

Levels and keys (the old values, which the test persona canon also uses):

| Level | Range | Key |
| --- | --- | --- |
| 6 | time of day | ``2026-05-01-EVENING`` |
| 5 | logical day | ``2026-05-01`` |
| 4 | logical week, Monday 05:00 | ``2026-W17`` |
| 3 | logical month | ``2026-05`` |
| 1 | logical year | ``2026`` |

Changes from the old code:

- The old code walked every day since 2020. :func:`closed_ranges` derives
  the ranges from the event times, so it only visits ranges with events.
- The old night had two segments (22-24 and 00-05) with keys from their
  middle times. The second segment of one day and the first segment of the
  next day got the same key, so one of them was never summarized. Here one
  NIGHT segment runs from 22:00 to 05:00, keyed by its logical day.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta

from pydantic import BaseModel

from pm.constants import RECONSOLIDATION_DAY_START_HOUR

LEVEL_YEAR = 1
LEVEL_MONTH = 3
LEVEL_WEEK = 4
LEVEL_DAY = 5
LEVEL_TIME_OF_DAY = 6
LEVEL_TOPICAL = 100

#: Time-of-day segments as hour offsets from the logical day start (05:00).
TIME_OF_DAY_SEGMENTS: tuple[tuple[str, int, int], ...] = (
    ("MORNING", 0, 7),     # 05:00 - 12:00
    ("NOON", 7, 9),        # 12:00 - 14:00
    ("AFTERNOON", 9, 13),  # 14:00 - 18:00
    ("EVENING", 13, 17),   # 18:00 - 22:00
    ("NIGHT", 17, 24),     # 22:00 - 05:00
)

#: The levels in build order. Finer levels come first, because a coarse
#: summary can read the summaries of a finer level.
BUILD_ORDER: tuple[int, ...] = (LEVEL_TIME_OF_DAY, LEVEL_DAY, LEVEL_WEEK, LEVEL_MONTH, LEVEL_YEAR)

#: The child level that a long range summarizes from, by level. The children
#: lie fully inside the parent range. A week does not nest in a month, so a
#: month reads days.
CHILD_LEVEL: dict[int, int] = {
    LEVEL_TIME_OF_DAY: LEVEL_TOPICAL,
    LEVEL_DAY: LEVEL_TIME_OF_DAY,
    LEVEL_WEEK: LEVEL_DAY,
    LEVEL_MONTH: LEVEL_DAY,
    LEVEL_YEAR: LEVEL_MONTH,
}


class TimeRange(BaseModel):
    """One closed range ``[start, end)`` of one level, with its unique key."""

    level: int = 0
    start: datetime = datetime.min
    end: datetime = datetime.min
    key: str = ""

    def contains(self, moment: datetime) -> bool:
        return self.start <= moment < self.end


def logical_day_start(moment: datetime, day_start_hour: int = RECONSOLIDATION_DAY_START_HOUR) -> datetime:
    """Return the start (05:00) of the logical day that holds ``moment``."""
    shifted = moment - timedelta(hours=day_start_hour)
    return shifted.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=day_start_hour)


def _add_months(start: datetime, months: int, day_start_hour: int) -> datetime:
    shifted = start - timedelta(hours=day_start_hour)
    index = shifted.month - 1 + months
    result = shifted.replace(year=shifted.year + index // 12, month=index % 12 + 1, day=1)
    return result + timedelta(hours=day_start_hour)


def ranges_of(moment: datetime, day_start_hour: int = RECONSOLIDATION_DAY_START_HOUR) -> list[TimeRange]:
    """Return the ranges of every level that contain ``moment``, finest first."""
    day = logical_day_start(moment, day_start_hour)
    logical_date = (day - timedelta(hours=day_start_hour)).date()
    result = []
    for label, begin, end in TIME_OF_DAY_SEGMENTS:
        segment = TimeRange(level=LEVEL_TIME_OF_DAY, start=day + timedelta(hours=begin),
                            end=day + timedelta(hours=end), key=f"{logical_date.isoformat()}-{label}")
        if segment.contains(moment):
            result.append(segment)
    result.append(TimeRange(level=LEVEL_DAY, start=day, end=day + timedelta(days=1), key=logical_date.isoformat()))
    week = day - timedelta(days=logical_date.weekday())
    result.append(TimeRange(level=LEVEL_WEEK, start=week, end=week + timedelta(days=7),
                            key=(week - timedelta(hours=day_start_hour)).strftime("%Y-W%W")))
    month = day - timedelta(days=logical_date.day - 1)
    result.append(TimeRange(level=LEVEL_MONTH, start=month, end=_add_months(month, 1, day_start_hour),
                            key=logical_date.strftime("%Y-%m")))
    year = day - timedelta(days=logical_date.timetuple().tm_yday - 1)
    result.append(TimeRange(level=LEVEL_YEAR, start=year, end=_add_months(year, 12, day_start_hour),
                            key=str(logical_date.year)))
    return result


def closed_ranges(moments: Iterable[datetime], now: datetime,
                  day_start_hour: int = RECONSOLIDATION_DAY_START_HOUR) -> list[TimeRange]:
    """Return every closed range that holds at least one of ``moments``.

    A range is closed when its end is not after ``now``. The result is in
    build order (:data:`BUILD_ORDER`), then by start time.
    """
    found: dict[str, TimeRange] = {}
    for moment in moments:
        for item in ranges_of(moment, day_start_hour):
            if item.end <= now:
                found.setdefault(item.key, item)
    order = {level: index for index, level in enumerate(BUILD_ORDER)}
    return sorted(found.values(), key=lambda item: (order[item.level], item.start, item.key))


__all__ = ["BUILD_ORDER", "CHILD_LEVEL", "LEVEL_DAY", "LEVEL_MONTH", "LEVEL_TIME_OF_DAY", "LEVEL_TOPICAL",
           "LEVEL_WEEK", "LEVEL_YEAR", "TIME_OF_DAY_SEGMENTS", "TimeRange", "closed_ranges", "logical_day_start",
           "ranges_of"]
