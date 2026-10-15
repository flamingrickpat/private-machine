"""Run independent calls with bounded workers and stable result order.

Callbacks must use the supplied cancellation signal for external I/O. The
first failure cancels siblings. All owned workers join before it propagates.
"""
from __future__ import annotations

from asyncio import CancelledError
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Event, Lock
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def ordered_parallel_map(operation: Callable[[T], R], items: Iterable[T], *, workers: int, cancel: Event) -> list[R]:
    """Return results in input order. Failure signals cancellation and joins workers."""
    errors: list[BaseException] = []
    lock = Lock()

    def call(item: T) -> R:
        try:
            if cancel.is_set():
                raise CancelledError
            return operation(item)
        except BaseException as error:
            with lock:
                if not errors:
                    errors.append(error)
            cancel.set()
            raise

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="inference") as pool:
        futures = [pool.submit(call, item) for item in items]
        wait(futures)
    if errors:
        raise errors[0]
    return [future.result() for future in futures]
