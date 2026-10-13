"""Durable JSON files and cross-process locks.

Managed runs and body leases both use these helpers. Writes are atomic,
and reads retry brief Windows sharing violations.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid


class BodyBusy(RuntimeError):
    """Another process owns the resource. No new action was sent."""


def read_json(path: Path) -> dict:
    """Read a complete receipt and retry brief Windows replacement locks.

    An absent or corrupt receipt still raises. The retry covers only a
    transient sharing violation, which can occur during atomic replacement.
    """
    for attempt in range(10):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.02)


def atomic_json(path: Path, value: dict) -> None:
    """Flush a JSON receipt before atomic replacement. Keep old data on error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(5):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise RuntimeError(f"Receipt replacement is locked: {path}.")
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


class ProcessLock:
    """Hold a byte lock until close or process exit. Never remove the lock file.

    Windows uses msvcrt. Other hosts use flock. A dead PID needs no timeout or
    operator takeover. The operating system releases its file handle.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.stream = None

    def acquire(self, *, wait: bool = False, cancel=None) -> "ProcessLock":
        """Acquire once, or raise BodyBusy. A waiting caller checks cancellation."""
        from pm_workflows.runtime import WorkflowTerminated
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            stream.write(b"\0")
            stream.flush()
        while True:
            if cancel is not None and cancel.is_set():
                stream.close()
                raise WorkflowTerminated()
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.stream = stream
                return self
            except OSError as error:
                if not wait:
                    stream.close()
                    raise BodyBusy(f"Resource is owned: {self.path}.") from error
                time.sleep(0.05)

    def close(self) -> None:
        """Release this handle after external cleanup."""
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self) -> "ProcessLock":
        return self.acquire()

    def __exit__(self, *args) -> None:
        self.close()

