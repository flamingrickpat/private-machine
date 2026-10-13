"""Versioned MemoryStore checkpoints. Replacement failure preserves the old file.

Invalid external snapshots fail validation. Filesystem errors propagate to
the debugger. This module does not retry a failed commit or publish memory.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pm.records.global_workspace import GwtTurnRecord


class CommitReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    commit_id: str
    added_ids: tuple[int, ...]
    updated_ids: tuple[int, ...]
    tick_id: int


class Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_id: Literal["pm.ghost-snapshot.v1"] = "pm.ghost-snapshot.v1"
    state: dict
    current_knoxel_id: int
    current_tick_id: int
    reserved_tick_id: int
    revisions: dict[int, int]
    gwt_turns: dict[int, GwtTurnRecord]
    applied_commits: dict[str, CommitReceipt]
    managed_runs: dict[str, dict] = Field(default_factory=dict)
    consolidations: dict[str, dict] = Field(default_factory=dict)


def write_checkpoint(path: Path, checkpoint: Checkpoint, check_cancelled) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(checkpoint.model_dump(mode="json"), stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        check_cancelled()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_checkpoint(path: Path) -> Checkpoint:
    return Checkpoint.model_validate_json(Path(path).read_text(encoding="utf-8"))
