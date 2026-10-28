"""Versioned MemoryStore checkpoints. Replacement failure preserves the old file.

Invalid external snapshots fail validation. Filesystem errors propagate to
the debugger. This module does not retry a failed commit or publish memory.
"""
from __future__ import annotations

import json
import os
import pickle
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pm.records.global_workspace import GwtTurnRecord
from pm.records.cycle_state import CycleStateModel
from pm.memory.yaml_codec import ghost_state_from_dict, ghost_state_to_dict
from pm.memory.causal import CausalHistory


class CommitReceipt(BaseModel):
    """The durable result of one accepted commit.

    ``added_ids`` are the records the commit created and ``updated_ids`` the
    heads it superseded. A repeated commit ID returns its original receipt, so
    these fields must describe the accepted state, not the retry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    commit_id: str
    added_ids: tuple[int, ...]
    updated_ids: tuple[int, ...]
    tick_id: int


class Checkpoint(BaseModel):
    """One complete save of an accepted store.

    ``schema_id`` selects the reader for a legacy file. The counters, revisions,
    workspace turns, receipts, and causal history are saved together, so a
    restored store cannot mix two revisions of accepted memory.
    """

    model_config = ConfigDict(extra="forbid")
    schema_id: Literal["pm.ghost-snapshot.v1", "pm.ghost-snapshot.v2"] = "pm.ghost-snapshot.v2"
    state: CycleStateModel
    current_knoxel_id: int
    current_tick_id: int
    reserved_tick_id: int
    revisions: dict[int, int]
    gwt_turns: dict[int, GwtTurnRecord]
    applied_commits: dict[str, CommitReceipt]
    managed_runs: dict[str, dict] = Field(default_factory=dict)
    consolidations: dict[str, dict] = Field(default_factory=dict)
    causal_history: CausalHistory | None = None

    @field_validator("state", mode="before")
    @classmethod
    def read_legacy_state(cls, value: CycleStateModel | dict) -> CycleStateModel:
        """Parse legacy JSON state once. Typed pickle state keeps its records."""
        return ghost_state_from_dict(value) if isinstance(value, dict) else value


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
        data = checkpoint.model_dump(mode="json", exclude={"state"})
        data["state"] = ghost_state_to_dict(checkpoint.state)
        return json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")

    def loads(self, data: bytes) -> Checkpoint:
        """Validate UTF-8 checkpoint JSON without changing the stored schema."""
        return Checkpoint.model_validate_json(data)


JSON_SERIALIZER = JsonCheckpointSerializer()


class PickleCheckpointSerializer(CheckpointSerializer):
    """Save typed records directly. Load pickle files from trusted character homes."""

    def dumps(self, checkpoint: Checkpoint) -> bytes:
        """Encode the complete state with the highest pickle protocol."""
        return pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL)

    def loads(self, data: bytes) -> Checkpoint:
        """Restore the typed state without record conversion or validation loops."""
        return pickle.loads(data)


PICKLE_SERIALIZER = PickleCheckpointSerializer()


def write_checkpoint(path: Path, checkpoint: Checkpoint, check_cancelled: Callable[[], None], *,
                     serializer: CheckpointSerializer = PICKLE_SERIALIZER) -> None:
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


def read_checkpoint(path: Path, *, serializer: CheckpointSerializer | None = None) -> Checkpoint:
    """Read pickle or legacy JSON bytes. An explicit serializer overrides detection."""
    data = Path(path).read_bytes()
    selected = serializer or (PICKLE_SERIALIZER if data.startswith(b"\x80") else JSON_SERIALIZER)
    return selected.loads(data)
