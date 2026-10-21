"""Separate a child's local stop from the enclosing task's cancellation."""
from __future__ import annotations

from threading import Event
import time


class LinkedStopEvent(Event):
    """Stop for either signal; setting this event stops only the child.

    The parent can cancel all later child work. A tool uncertainty can stop one
    actor without cancelling its planner. Clearing this event cannot clear the
    parent's signal. Waits poll the parent at intervals of at most 0.1 seconds.
    """

    def __init__(self, parent: Event) -> None:
        super().__init__()
        self.parent = parent

    def is_set(self) -> bool:
        """Return whether local work or its enclosing task must stop."""
        return self.parent.is_set() or super().is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for either stop signal; return false when the timeout expires."""
        deadline = time.monotonic() + timeout if timeout is not None else None
        while not self.is_set():
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            super().wait(0.1 if remaining is None else min(0.1, remaining))
        return True
