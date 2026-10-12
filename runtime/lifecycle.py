"""Durable startup and shutdown evidence for the companion.

The companion must appraise a real offline interval, and it must not invent a
clean stop that never happened. This module records two system events:

- a startup event with the boot identity, the last known activity, and the
  elapsed gap;
- a clean shutdown event, written only when the module terminates normally.

An abrupt exit leaves no clean shutdown event, so the next startup reports an
uncertain end interval. Replaying the same boot identity or shutdown identity
reuses the accepted event instead of writing a duplicate autobiographical
event.
"""
from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from pm.records.memory_records import Feature
from pm.records.base import Provenance
from pm.records.enums import FeatureType, InterlocusType, SourceKind

UNCERTAIN_END = "uncertain"
CLEAN_END = "clean"


def _system_feature(content: str, *, event: str, boot_id: str, **metadata) -> Feature:
    return Feature(
        content=content,
        feature_type=FeatureType.SystemMessage,
        causal=True,
        override_source="system",
        interlocus=InterlocusType.PrivateReportable,
        provenance=Provenance(source_kind=SourceKind.System, source_ref=boot_id),
        metadata={"origin": "lifecycle", "event": event, "boot_id": boot_id,
                  "memory_kind": "executed_event", **metadata},
    )


def _find_system_event(ghost, event: str, boot_id: str = "") -> Feature | None:
    matches = [
        knoxel for knoxel in ghost.all_knoxels.values()
        if isinstance(knoxel, Feature)
        and knoxel.feature_type == FeatureType.SystemMessage
        and str(knoxel.metadata.get("origin", "")) == "lifecycle"
        and str(knoxel.metadata.get("event", "")) == event
        and (not boot_id or str(knoxel.metadata.get("boot_id", "")) == boot_id)
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (item.timestamp_creation, item.id))
    return matches[-1]


def last_known_activity(ghost) -> datetime | None:
    """Return the newest timestamp in accepted memory, or None when empty."""
    stamps = [
        getattr(knoxel, "timestamp_world_begin", None) or getattr(knoxel, "timestamp_creation", None)
        for knoxel in ghost.all_knoxels.values()
    ]
    stamps = [stamp for stamp in stamps if stamp is not None]
    return max(stamps) if stamps else None


def last_completed_shutdown(ghost) -> Feature | None:
    """Return the most recent clean shutdown event, if any."""
    return _find_system_event(ghost, "clean_shutdown")


def _describe_gap(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.1f} seconds"
    if seconds < 5400:
        return f"{seconds / 60.0:.1f} minutes"
    return f"{seconds / 3600.0:.1f} hours"


def record_startup(ghost, *, boot_id: str | None = None, now: datetime | None = None) -> dict:
    """Record one boot event and return its evidence.

    The result has ``knoxel_id``, ``text``, ``boot_id``, ``end_kind``, and
    ``gap_seconds``. A repeated call with the same boot identity returns the
    accepted event without creating a duplicate.
    """
    boot_id = boot_id or uuid4().hex
    existing = _find_system_event(ghost, "startup", boot_id)
    if existing is not None:
        return {
            "knoxel_id": int(existing.id),
            "text": str(existing.content),
            "boot_id": boot_id,
            "end_kind": str(existing.metadata.get("end_kind", UNCERTAIN_END)),
            "gap_seconds": float(existing.metadata.get("gap_seconds", 0.0)),
            "duplicate": True,
        }
    now = now or datetime.utcnow()
    activity = last_known_activity(ghost)
    gap_seconds = max(0.0, (now - activity).total_seconds()) if activity else 0.0
    shutdown = last_completed_shutdown(ghost)
    gap_text = _describe_gap(gap_seconds)
    if shutdown is not None:
        end_kind = CLEAN_END
        text = (
            f"I started again after a clean stop. My last known activity was "
            f"{gap_text} before this startup."
        )
    else:
        end_kind = UNCERTAIN_END
        text = (
            f"I started again, but the last stop was not clean, so the end is "
            f"uncertain. At least {gap_text} passed since my last known activity."
        )
    feature = _system_feature(
        text, event="startup", boot_id=boot_id,
        end_kind=end_kind, gap_seconds=gap_seconds,
        last_activity=activity.isoformat() if activity else "",
        clean_shutdown_id=int(shutdown.id) if shutdown is not None else None,
    )
    stored = ghost.store_knoxel(feature)
    return {
        "knoxel_id": int(stored.id),
        "text": str(stored.content),
        "boot_id": boot_id,
        "end_kind": end_kind,
        "gap_seconds": gap_seconds,
        "duplicate": False,
    }


def record_shutdown(ghost, *, boot_id: str | None = None, now: datetime | None = None) -> int | None:
    """Record a clean shutdown and return its knoxel id.

    A repeated call for the same boot identity reuses the accepted event. The
    return value is None when no boot identity is available and no startup
    event exists, because a shutdown without a startup identity is ambiguous.
    """
    if boot_id is None:
        startup = _find_system_event(ghost, "startup")
        boot_id = str(startup.metadata.get("boot_id", "")) if startup is not None else ""
    existing = _find_system_event(ghost, "clean_shutdown", boot_id) if boot_id else None
    if existing is not None:
        return int(existing.id)
    now = now or datetime.utcnow()
    text = f"I stopped cleanly at {now.isoformat()}."
    feature = _system_feature(text, event="clean_shutdown", boot_id=boot_id, stopped_at=now.isoformat())
    return int(ghost.store_knoxel(feature).id)


__all__ = [
    "CLEAN_END",
    "UNCERTAIN_END",
    "last_completed_shutdown",
    "last_known_activity",
    "record_shutdown",
    "record_startup",
]
