"""Process clock for ordinary operation and controlled simulation time.

World time uses UTC. Model transport deadlines and diagnostics keep real time.
Each PM process owns one clock. Only its controller advances virtual time.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import threading
import time


class VirtualClock:
    """Advance UTC time without sleeping. Backward movement raises ValueError."""

    def __init__(self, start: datetime, utc_offset_hours: float | None = None):
        self._lock = threading.RLock()
        self.utc_offset = (start.utcoffset() or timedelta()) if utc_offset_hours is None else timedelta(hours=utc_offset_hours)
        self._start = self._utc(start)
        self._now = self._start

    def _utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return (value - self.utc_offset).replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def now(self) -> datetime:
        """Return a naive local value for contributor control compatibility."""
        with self._lock:
            return (self._now + self.utc_offset).replace(tzinfo=None)

    def set(self, value: datetime) -> None:
        """Move forward to an absolute time. Restore uses a new clock instance."""
        target = self._utc(value)
        with self._lock:
            if target < self._now:
                raise ValueError("Virtual time cannot move backward. Restore a coordinated checkpoint.")
            self._now = target

    def monotonic(self) -> float:
        """Return elapsed virtual seconds since this clock started."""
        with self._lock:
            return (self._now - self._start).total_seconds()

    def utc(self) -> datetime:
        """Return timezone-aware UTC world time."""
        with self._lock:
            return self._now

    def state(self) -> dict:
        """Retain the origin, current time, and display offset for exact resume."""
        with self._lock:
            return {"start": self._start.isoformat(), "now": self._now.isoformat(),
                    "utc_offset_hours": self.utc_offset.total_seconds() / 3600}

    @classmethod
    def from_state(cls, state: dict):
        """Restore elapsed time without moving an existing clock backward."""
        clock = cls(datetime.fromisoformat(state["start"]), state["utc_offset_hours"])
        clock.set(datetime.fromisoformat(state["now"]))
        return clock


_active: VirtualClock | None = None
_busy = 0
_busy_lock = threading.Lock()


def install(clock: VirtualClock | None) -> None:
    """Set the process clock before application construction. None restores real time."""
    global _active
    _active = clock


def active() -> VirtualClock | None:
    """Return the installed virtual clock, if present."""
    return _active


def utcnow() -> datetime:
    """Return naive UTC for existing world-time record fields."""
    return now_aware().replace(tzinfo=None)


def now_aware() -> datetime:
    """Return timezone-aware UTC world time."""
    return _active.utc() if _active is not None else datetime.now(timezone.utc)


def monotonic() -> float:
    """Return virtual elapsed seconds, or the ordinary monotonic clock."""
    return _active.monotonic() if _active is not None else time.monotonic()


def wait_cap(wait: float | None) -> float | None:
    """Poll virtual timers at most every 20 real milliseconds."""
    if _active is None:
        return wait
    return .02 if wait is None else min(wait, .02)


@contextmanager
def busy():
    """Count work in flight. Exceptions still reach the application."""
    global _busy
    with _busy_lock:
        _busy += 1
    try:
        yield
    finally:
        with _busy_lock:
            _busy -= 1


def busy_count() -> int:
    """Return the number of module batches in flight."""
    with _busy_lock:
        return _busy
