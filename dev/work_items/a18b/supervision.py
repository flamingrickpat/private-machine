"""Bound a case worker's lifetime without changing model transport policy.

STOP and deadlines first request cooperative cleanup. After the grace period,
the controller stops only the process tree it created. Forced termination is
incomplete evidence, even if a worker wrote an earlier successful receipt.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import time
from threading import Event, Thread
from typing import TextIO

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pm.utils.operating_system import OperatingSystem


@dataclass(frozen=True)
class WorkerResult:
    """Record process exit separately from the runtime's own completion claim."""

    returncode: int
    elapsed_seconds: float
    termination_reason: str | None = None
    forced: bool = False

    def to_dict(self) -> dict:
        """Return the serializable process evidence."""
        return asdict(self)


@contextmanager
def worker_cancellation(case: Path, timeout_seconds: float):
    """Signal model I/O to stop while the controller retains a process deadline."""
    cancel, done = Event(), Event()
    deadline = time.monotonic() + timeout_seconds

    def watch():
        while not done.wait(0.1):
            if (case / "STOP").exists() or (case.parent / "STOP").exists() or time.monotonic() >= deadline:
                cancel.set()
                return

    thread = Thread(target=watch, name="a18b_review_cancel", daemon=True)
    thread.start()
    try:
        yield cancel
    finally:
        done.set()
        thread.join()


def run_worker(command: list[str], *, cwd: Path, case: Path, log: TextIO,
               timeout_seconds: float, cleanup_seconds: float = 60) -> WorkerResult:
    """Wait for an owned worker, then retain its actual exit and stop reason.

    The case deadline includes activation, inference, restart, and cleanup.
    A STOP file in the suite or case directory requests cancellation. The
    caller must retain this result even when no terminal case receipt exists.
    """
    started = time.monotonic()
    process = OperatingSystem.start_binary_process(
        command,
        cwd=cwd,
        stdout=log,
        stderr=OperatingSystem.merged_output,
        new_process_group=True
    )
    reason = None
    stop_at = None
    forced = False
    def kill_tree() -> None:
        if process.poll() is not None:
            return
        OperatingSystem.kill_process_group(process, log=log)
        process.wait()

    try:
        while process.poll() is None:
            now = time.monotonic()
            if reason is None:
                if (case.parent / "STOP").exists() or (case / "STOP").exists():
                    reason = "operator STOP"
                elif now - started >= timeout_seconds:
                    reason = "case deadline exceeded"
                if reason is not None:
                    stop_at = now
                    # The worker owns its fresh directory. Do not race with
                    # that creation when activation has not started yet.
                    if case.is_dir():
                        (case / "STOP").touch()
            if stop_at is not None and now - stop_at >= cleanup_seconds:
                forced = True
                kill_tree()
                break
            time.sleep(0.1)
    except BaseException:
        # A controller interrupt must not orphan its case or Minecraft stack.
        # Request cleanup first. Escalation still targets only this child tree.
        if case.is_dir():
            (case / "STOP").touch()
        try:
            process.wait(timeout=cleanup_seconds)
        except subprocess.TimeoutExpired:
            kill_tree()
        raise
    return WorkerResult(process.wait(), round(time.monotonic() - started, 3), reason, forced)
