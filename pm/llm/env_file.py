"""Read credentials from the local, git-ignored ``.env`` file.

The process environment always wins. When a variable is not set there, the
config falls back to this file, so local runs, tests, and gates need no manual
export. Values are returned to the caller only; nothing is written into
``os.environ``, so resolving a credential never changes global process state.

The file is ``PM_ENV_FILE`` when that variable is set, otherwise ``.env`` in
the repository root. A missing file means no values. Lines are ``KEY=value``;
blank lines, ``#`` comments, an ``export`` prefix, and surrounding quotes are
accepted.
"""
from __future__ import annotations

import os
from pathlib import Path

from pm.paths import REPO_ROOT


def env_file_path() -> Path:
    return Path(os.environ["PM_ENV_FILE"]) if os.environ.get("PM_ENV_FILE") else REPO_ROOT / ".env"


def read_env_file(path: Path | None = None) -> dict[str, str]:
    """Return the key-value pairs of one env file, or an empty mapping."""
    path = path or env_file_path()
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def credential(name: str) -> str:
    """Return a credential from the process environment, then the env file, or ""."""
    return os.environ.get(name) or read_env_file().get(name, "")


__all__ = ["credential", "env_file_path", "read_env_file"]
