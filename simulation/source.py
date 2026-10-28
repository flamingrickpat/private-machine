"""Freeze the actual participant source and installed dependency origins per attempt.

The participant processes run from this copy, so a developer edit during a live
attempt cannot change workflow scripts under the running cognition. The receipt
records the repository revision, the tracked and unignored file hashes, the model
configuration hash, and the installed distribution origins.

Ordering: every file is copied before it is hashed, and the receipt is written
last, so a ``source.json`` that exists describes files that are already present.
Credentials stay in local data and never enter the copy. The package overview is
in ``simulation/__init__.py``.
"""
from __future__ import annotations

import hashlib
from importlib.metadata import distribution
from pathlib import Path
import shutil
import subprocess

from pydantic import JsonValue

from pm.host.durable_files import atomic_json
from pm.paths import REPO_ROOT


def freeze_source(destination: Path, config: Path) -> dict[str, JsonValue]:
    """Copy tracked and unignored source files without changing the repository.

    Child processes run from this copy. Later developer edits cannot change
    workflow scripts during an active attempt. Credentials stay in local data.
    Returns the receipt that was written beside the copy. Raises
    ``FileExistsError`` when the destination already exists.
    """
    def git(*arguments: str) -> bytes:
        return subprocess.check_output(["git", *arguments], cwd=REPO_ROOT)
    paths = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")
    destination.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for raw in paths:
        if not raw:
            continue
        relative = Path(raw.decode("utf-8"))
        source = REPO_ROOT / relative
        if not source.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        # Hash the copy, not the original: the receipt must describe the bytes the
        # participant will actually import.
        hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    saved_config = destination.parent / "model-config.yaml"
    shutil.copyfile(config, saved_config)
    dependencies: dict[str, dict[str, JsonValue]] = {}
    for name in ("pm-coder", "pm-workflows", "websockets"):
        installed = distribution(name)
        dependencies[name] = {"version": installed.version, "direct_url": installed.read_text("direct_url.json")}
    receipt: dict[str, JsonValue] = {"revision": git("rev-parse", "HEAD").decode().strip(),
               "status": git("status", "--porcelain").decode(), "files": hashes,
               "dependencies": dependencies, "source": str(destination.resolve()),
               "config": str(saved_config.resolve()), "config_sha256": hashlib.sha256(saved_config.read_bytes()).hexdigest()}
    atomic_json(destination.parent / "source.json", receipt)
    return receipt
