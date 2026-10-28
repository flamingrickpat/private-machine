"""Freeze the actual participant source and installed dependency origins per attempt."""
from __future__ import annotations

import hashlib
from importlib.metadata import distribution
from pathlib import Path
import shutil
import subprocess

from pm.host.durable_files import atomic_json
from pm.paths import REPO_ROOT


def freeze_source(destination: Path, config: Path) -> dict:
    """Copy tracked and unignored source files without changing the repository.

    Child processes run from this copy. Later developer edits cannot change
    workflow scripts during an active attempt. Credentials stay in local data.
    """
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=REPO_ROOT)
    paths = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0")
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
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
        hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    saved_config = destination.parent / "model-config.yaml"
    shutil.copyfile(config, saved_config)
    dependencies = {}
    for name in ("pm-coder", "pm-workflows", "websockets"):
        installed = distribution(name)
        dependencies[name] = {"version": installed.version, "direct_url": installed.read_text("direct_url.json")}
    receipt = {"revision": git("rev-parse", "HEAD").decode().strip(),
               "status": git("status", "--porcelain").decode(), "files": hashes,
               "dependencies": dependencies, "source": str(destination.resolve()),
               "config": str(saved_config.resolve()), "config_sha256": hashlib.sha256(saved_config.read_bytes()).hexdigest()}
    atomic_json(destination.parent / "source.json", receipt)
    return receipt
