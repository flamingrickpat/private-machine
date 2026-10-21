"""Write gate evidence files and digest their sources.

Every work-item gate keeps its receipts as indented UTF-8 JSON.
These helpers keep that format in one place.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


def save_json(path: Path, value: Any, *, default: Callable[[Any], Any] | None = None) -> None:
    """Write one evidence file immediately, so a later failure keeps it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=default), encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file without changing it."""
    return sha256_bytes(path.read_bytes())
