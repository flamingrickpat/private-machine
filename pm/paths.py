"""Repository locations that runtime modules share."""
from __future__ import annotations

from pathlib import Path

#: The repository root. Workflow manifests name skills relative to it.
REPO_ROOT = Path(__file__).resolve().parent.parent

__all__ = ["REPO_ROOT"]
