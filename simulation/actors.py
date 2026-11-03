"""Ground simulated users in public history before a scenario starts.

The simulation controller owns these profiles. Each profile preserves the
actual user and companion names, complete public dialogue with source IDs,
and the source world's clock. It contains no companion thoughts, appraisals,
or other private state. Run creation freezes it beside the participant fork.
Generation reads only the current scene; future scenario facts stay with the
reviewer. See ``simulation.__init__`` for controller and participant ownership.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pm.memory.store import MemoryStore
from pm.records.enums import FeatureType, InterlocusType


def world_time(value: datetime) -> datetime:
    """Normalize persisted world times; legacy naive seed times are UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class HistoricalTurn(BaseModel):
    """One exact public utterance from the source checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    record_id: int = Field(description="Original public dialogue knoxel ID.")
    speaker: str
    at: datetime
    text: str


class ActorProfile(BaseModel):
    """Frozen public grounding for the user of one controlled companion."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_id: Literal["pm.simulation-actor.v1"] = "pm.simulation-actor.v1"
    speaker_name: str = Field(min_length=1, description="Real user name from the character home.")
    companion_name: str = Field(min_length=1, description="Actual AI companion name; never a transport ID.")
    relationship: Literal["user_and_ai_companion"] = "user_and_ai_companion"
    source_checkpoint_sha256: str
    continuity_at: datetime = Field(description="Latest saved public dialogue, cognitive cycle, or world-clock time.")
    dialogue: list[HistoricalTurn] = Field(min_length=1, description="Complete public history; text is unchanged.")


class ActorSummary(BaseModel):
    """Small browser projection of the selected source's public grounding."""

    speaker_name: str
    companion_name: str
    dialogue_turns: int
    speaker_turns: int
    continuity_at: datetime
    suggested_start_time: datetime


class ConversationTurn(BaseModel):
    """Public run dialogue, without controller receipts or workflow internals."""

    speaker: str
    at: str
    text: str


class DeliveryEvidence(BaseModel):
    """Public delivery metadata available to the director, never the actor."""

    kind: Literal["actor", "speech"]
    speaker: str
    timestamp: datetime
    content: str
    delivery_id: str | None = None
    target: str | None = None
    source: str | None = None


class SavedClock(BaseModel):
    """World time read from the participant's saved clock."""

    now: datetime


class SavedClockReceipt(BaseModel):
    """Read only clock time; timers and delivery state remain participant-owned."""

    clock: SavedClock


def profile_from_store(store: MemoryStore, checkpoint: Path) -> ActorProfile:
    """Read public dialogue and continuity without changing the checkpoint."""
    dialogue: list[HistoricalTurn] = []
    for record in store.all_features:
        if record.feature_type != FeatureType.Dialogue or record.interlocus != InterlocusType.Public:
            continue
        dialogue.append(HistoricalTurn(record_id=record.id, speaker=record.source,
                                       at=world_time(record.timestamp_world_begin), text=record.content))
    dialogue.sort(key=lambda row: (row.at, row.record_id))
    speaker = store.ghost_config.user_name
    if not any(row.speaker == speaker for row in dialogue):
        raise ValueError(f"No public {speaker} utterances exist to ground this simulation actor.")
    continuity = dialogue[-1].at
    if store.states:
        continuity = max(continuity, world_time(store.states[-1].timestamp))
    clock_path = checkpoint.parent / "runtime" / "clock.json"
    if clock_path.exists():
        # A home clock is a saved external JSON boundary, not model-generated time.
        receipt = SavedClockReceipt.model_validate_json(clock_path.read_text(encoding="utf-8"))
        continuity = max(continuity, world_time(receipt.clock.now))
    with checkpoint.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return ActorProfile(speaker_name=speaker, companion_name=store.ghost_config.companion_name,
                        source_checkpoint_sha256=digest, continuity_at=continuity, dialogue=dialogue)


def load_profile(home: Path) -> ActorProfile:
    """Load an accepted source checkpoint and derive its public actor profile."""
    checkpoint = home / "ghost.pickle"
    return profile_from_store(MemoryStore.from_checkpoint(checkpoint), checkpoint)


def suggested_start(profile: ActorProfile) -> datetime:
    """Choose the next 23:58 after source continuity, rather than today's date."""
    proposed = datetime.combine(profile.continuity_at.date(), time(23, 58), tzinfo=profile.continuity_at.tzinfo)
    if proposed <= profile.continuity_at:
        proposed += timedelta(days=1)
    return proposed


def summarize(profile: ActorProfile) -> ActorSummary:
    """Expose names, evidence counts, and the default date to the browser."""
    return ActorSummary(speaker_name=profile.speaker_name, companion_name=profile.companion_name,
                        dialogue_turns=len(profile.dialogue),
                        speaker_turns=sum(row.speaker == profile.speaker_name for row in profile.dialogue),
                        continuity_at=profile.continuity_at, suggested_start_time=suggested_start(profile))
