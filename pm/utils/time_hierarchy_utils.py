from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Tuple, Optional

_DEFAULT_TZ = timezone.utc
_SHIFT_HOURS = 5  # day starts at 05:00 local and ends at next-day 04:59:59

TimeRange = Tuple[int, datetime, datetime, str, str]  # (level, start, end, key, label)

# -------- Shift helpers (core trick) --------
def _shift(dt: datetime) -> datetime:
    return dt - timedelta(hours=_SHIFT_HOURS)

def _unshift(dt: datetime) -> datetime:
    return dt + timedelta(hours=_SHIFT_HOURS)

def _floor_to_shifted_day_start(dt: datetime) -> datetime:
    """
    Returns the logical day start (at 05:00) that contains dt.
    """
    s = _shift(dt)
    s = s.replace(hour=0, minute=0, second=0, microsecond=0)
    return _unshift(s)  # = 05:00 of the logical day

def _start_of_week_monday_shifted(dt: datetime) -> datetime:
    """
    Monday-based week start in shifted domain, then unshift back.
    """
    s = _shift(dt)
    s0 = s.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = s0 - timedelta(days=s0.weekday())  # Monday 00:00 (shifted)
    return _unshift(week_start)  # Monday 05:00 (logical)

def _start_of_month_shifted(dt: datetime) -> datetime:
    s = _shift(dt)
    m0 = s.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return _unshift(m0)  # 1st @ 05:00 (logical)

def _start_of_year_shifted(dt: datetime) -> datetime:
    s = _shift(dt)
    y0 = s.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return _unshift(y0)  # Jan 1 @ 05:00 (logical)

def _month_add_unshifted(start_logical: datetime, months: int) -> datetime:
    """
    Add months in the shifted calendar. Input/output are unshifted logical starts.
    """
    s = _shift(start_logical)
    y = s.year + ((s.month - 1 + months) // 12)
    m = ((s.month - 1 + months) % 12) + 1
    s2 = s.replace(year=y, month=m, day=1, hour=0, minute=0, second=0, microsecond=0)
    return _unshift(s2)

# -------- Iterators for CLOSED ranges with 05:00 day boundary --------

def _iter_closed_tod_ranges(self, day_start_0500: datetime, now: datetime) -> Iterable[TimeRange]:
    """
    Time-of-day slices inside a single logical day [05:00 .. next 05:00).
    Segments:
      MORNING   05–12
      NOON      12–14
      AFTERNOON 14–18
      EVENING   18–22
      NIGHT     22–24 (same calendar day)
      NIGHT     00–05 (next calendar day, but same logical day)
    """
    segs = [
        ("MORNING",    0,   7),   # 05:00 → 12:00
        ("NOON",       7,   9),   # 12:00 → 14:00
        ("AFTERNOON",  9,  13),   # 14:00 → 18:00
        ("EVENING",   13,  17),   # 18:00 → 22:00
        ("NIGHT",     17,  19),   # 22:00 → 24:00
        ("NIGHT",     19,  24),   # 00:00 → 05:00 (next calendar day)
    ]
    for label, h0, h1 in segs:
        s = day_start_0500 + timedelta(hours=h0)
        e = day_start_0500 + timedelta(hours=h1)
        if e <= now:
            mid = s + (e - s) / 2
            key = self.get_temporal_key(mid, level=6)
            # label the logical day by its date-at-05:00 for clarity
            logical_label = f"{(day_start_0500.date())} {label}"
            yield (6, s, e, key, logical_label)

def _iter_closed_day_ranges_0500(self, min_dt: datetime, now: datetime) -> Iterable[TimeRange]:
    """
    Full logical days [05:00 .. next 05:00) that are finished.
    """
    cur = _floor_to_shifted_day_start(min_dt)
    # Day end is start + 24h
    while cur + timedelta(days=1) <= now:
        s = cur
        e = cur + timedelta(days=1)
        mid = s + (e - s) / 2
        key = self.get_temporal_key(mid, level=5)
        # Label by the day’s 05:00 date
        yield (5, s, e, key, s.strftime("%Y-%m-%d(05:00)"))
        cur = e

def _iter_closed_week_ranges_0500(self, min_dt: datetime, now: datetime) -> Iterable[TimeRange]:
    """
    Monday-based logical weeks, each starting at Monday 05:00 and ending next Monday 05:00.
    """
    cur = _start_of_week_monday_shifted(min_dt)
    this_week = _start_of_week_monday_shifted(now)
    while cur < this_week:
        s = cur
        e = cur + timedelta(days=7)
        if e <= now:
            mid = s + (e - s) / 2
            key = self.get_temporal_key(mid, level=4)
            yield (4, s, e, key, f"Week {s.strftime('%Y-W%W')} @05:00")
        cur = e

def _iter_closed_month_ranges_0500(self, min_dt: datetime, now: datetime) -> Iterable[TimeRange]:
    """
    Logical months, each starting on the 1st @ 05:00 (shifted calendar) and ending at the next 1st @ 05:00.
    """
    cur = _start_of_month_shifted(min_dt)
    this_month = _start_of_month_shifted(now)
    while cur < this_month:
        s = cur
        e = _month_add_unshifted(s, 1)
        if e <= now:
            mid = s + (e - s) / 2
            key = self.get_temporal_key(mid, level=3)
            yield (3, s, e, key, s.strftime("%Y-%m @05:00"))
        cur = e

def _iter_closed_year_ranges_0500(self, min_dt: datetime, now: datetime) -> Iterable[TimeRange]:
    """
    Logical years, each starting Jan 1 @ 05:00 and ending next Jan 1 @ 05:00 (shifted calendar).
    """
    cur = _start_of_year_shifted(min_dt)
    this_year = _start_of_year_shifted(now)
    while cur < this_year:
        s = cur
        e = s.replace(year=s.year + 1)
        if e <= now:
            mid = s + (e - s) / 2
            key = self.get_temporal_key(mid, level=1)
            yield (1, s, e, key, f"{s.year} @05:00")
        cur = e
