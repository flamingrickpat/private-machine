import threading
import time


class PerThreadBroadcastEvent:
    """
    A broadcast-style event with per-thread consumption.

    - set_all():     flip flag True, bump generation, wake everyone
    - clear_all():   flip flag False, bump generation, wake everyone
    - wait_set():    block until there's a *new* True generation for THIS thread, then consume it
    - poll_set():    nonblocking version; consume if a new True generation exists
    - ack():         consume current generation (whether True or False) for THIS thread

    Threads do not need to register. Each thread keeps an internal 'seen_generation'.
    """
    def __init__(self):
        self._cv = threading.Condition()
        self._flag = False               # desired global flag (set/clear for all)
        self._generation = 0             # monotonic; bumps on every global change
        self._tls = threading.local()    # per-thread 'seen_generation'

    # --------- helpers ---------
    def _get_seen(self):
        return getattr(self._tls, "seen_generation", -1)

    def _set_seen(self, gen):
        self._tls.seen_generation = gen

    # --------- broadcaster (thread A) API ---------
    def set_all(self):
        with self._cv:
            self._flag = True
            self._generation += 1
            self._cv.notify_all()

    def clear_all(self):
        with self._cv:
            self._flag = False
            self._generation += 1
            self._cv.notify_all()

    # --------- consumers (worker threads) API ---------
    def wait_set(self, timeout=None):
        """
        Block until there is a *new* set-event for THIS thread.
        Returns True if consumed, False on timeout.
        """
        end = None if timeout is None else (time.monotonic() + timeout)
        with self._cv:
            while True:
                if self._flag and self._get_seen() != self._generation:
                    # consume for this thread
                    self._set_seen(self._generation)
                    return True
                if timeout is not None:
                    remaining = end - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._cv.wait(remaining)
                else:
                    self._cv.wait()

    def poll_set(self):
        """
        Nonblocking check. If there is a *new* set-event for THIS thread, consume it and return True.
        Otherwise return False.
        """
        with self._cv:
            if self._flag and self._get_seen() != self._generation:
                self._set_seen(self._generation)
                return True
            return False

    def is_globally_set(self):
        """Snapshot of the global flag (not per-thread)."""
        with self._cv:
            return self._flag

    def ack(self):
        """
        Mark the current generation as seen for THIS thread (useful if you also want
        to 'consume' clear_all() transitions or to skip one broadcast).
        """
        with self._cv:
            self._set_seen(self._generation)

    def current_generation(self):
        with self._cv:
            return self._generation

    def my_seen_generation(self):
        return self._get_seen()