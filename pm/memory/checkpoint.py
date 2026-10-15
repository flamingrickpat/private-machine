"""Versioned MemoryStore checkpoints. Replacement failure preserves the old file.

Invalid external snapshots fail validation. Filesystem errors propagate to
the debugger. This module does not retry a failed commit or publish memory.
"""
from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
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


class CheckpointSerializer(ABC):
    """Encode checkpoint bytes. File ownership and atomic replacement stay outside."""

    @abstractmethod
    def dumps(self, checkpoint: Checkpoint) -> bytes:
        """Encode a checkpoint or raise before any accepted state changes."""

    @abstractmethod
    def loads(self, data: bytes) -> Checkpoint:
        """Decode and validate external bytes. Invalid data raises an error."""


class JsonCheckpointSerializer(CheckpointSerializer):
    """Keep the existing JSON format and encode it in one allocation."""

    def dumps(self, checkpoint: Checkpoint) -> bytes:
        """Return UTF-8 JSON. Non-finite numbers raise ValueError."""
        return json.dumps(checkpoint.model_dump(mode="json"), ensure_ascii=False, allow_nan=False).encode("utf-8")

    def loads(self, data: bytes) -> Checkpoint:
        """Validate UTF-8 checkpoint JSON without changing the stored schema."""
        return Checkpoint.model_validate_json(data)


JSON_SERIALIZER = JsonCheckpointSerializer()


def write_checkpoint(path: Path, checkpoint: Checkpoint, check_cancelled: Callable[[], None], *,
                     serializer: CheckpointSerializer = JSON_SERIALIZER) -> None:
    """Sync encoded bytes before atomic replacement. Cancellation preserves the old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = serializer.dumps(checkpoint)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        check_cancelled()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_checkpoint(path: Path, *, serializer: CheckpointSerializer = JSON_SERIALIZER) -> Checkpoint:
    """Read through the selected serializer. File and validation errors propagate."""
    return serializer.loads(Path(path).read_bytes())
