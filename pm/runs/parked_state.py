"""Portable kernel and environment state for a parked workflow.

Checkpoints retain file bytes, not mutable paths into a later runtime branch.
Restore archives an existing directory before installing the recorded version.
"""
from __future__ import annotations

import base64
from pathlib import Path
from uuid import uuid4


def capture_tree(root: Path) -> dict[str, str | None]:
    """Capture exact bytes under one owned directory. Operating-system locks are excluded."""
    return {path.relative_to(root).as_posix(): base64.b64encode(path.read_bytes()).decode("ascii") if path.is_file() else None
            for path in sorted(root.rglob("*")) if not path.name.endswith(".lock")}


def restore_tree(root: Path, files: dict[str, str | None], *, owner: Path) -> None:
    """Install a recorded directory while preserving the later branch beside it.

    All targets must remain under the explicit owner directory. Invalid paths
    raise before any move. The old directory remains available for debugging.
    """
    root, owner = root.resolve(), owner.resolve()
    if root == owner or not root.is_relative_to(owner):
        raise ValueError("A parked runtime directory is outside its owner.")
    decoded = {}
    for name, value in files.items():
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError("A parked file escaped its runtime directory.")
        decoded[target.relative_to(root)] = base64.b64decode(value, validate=True) if value is not None else None
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = root.with_name(root.name + ".restoring-" + uuid4().hex)
    staging.mkdir()
    for name, value in decoded.items():
        target = staging / name
        if value is None:
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    if root.exists():
        archive = root.with_name(root.name + ".branch-" + uuid4().hex)
        if not archive.resolve().is_relative_to(owner):
            raise ValueError("The parked branch archive escaped its owner.")
        root.rename(archive)
    staging.rename(root)


def capture_kernel(runtime_root: Path, namespace: str, run_id: str) -> dict:
    """Retain the kernel journal and its artifact workspace at a wait boundary."""
    return {"kernel": capture_tree(runtime_root / "kernel" / run_id),
            "workspace": capture_tree(runtime_root / namespace / run_id)}


def restore_kernel(runtime_root: Path, namespace: str, run_id: str, state: dict) -> None:
    """Restore the exact parked journal before the kernel resumes."""
    restore_tree(runtime_root / "kernel" / run_id, state["kernel"], owner=runtime_root)
    restore_tree(runtime_root / namespace / run_id, state["workspace"], owner=runtime_root)
