"""Capture named record collections without changing accepted runtime state.

Callers choose the collection and its intended reader. A display hint does not
grant access to private records. Embeddings are omitted from exported copies;
timestamps, appraisal fields, codelet metadata, and other fields stay intact.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Literal

from pm.memory.snapshot import Snapshot

SCHEMA = "pm.record-package.v1"


def without_embeddings(value):
    """Copy serialized data recursively, excluding only embedding fields."""
    if isinstance(value, dict):
        return {key: without_embeddings(item) for key, item in value.items()
                if key not in {"embedding", "embeddings"}}
    if isinstance(value, list):
        return [without_embeddings(item) for item in value]
    return value


def capture_records(snapshot: Snapshot, record_ids: Iterable[int], *, title: str,
                    view: Literal["records", "chat", "gwt"] = "records",
                    perspective: str = "developer", related_ids: Iterable[int] = (),
                    tick_id: int | None = None) -> dict:
    """Return a detached collection from one frozen snapshot.

    Keep caller order and remove duplicate ids. Related sources accompany the
    selected records without becoming collection members. Missing records or
    GWT ticks raise KeyError. A chat collection rejects private record types.
    This function performs no inference, file writes, or memory commits.
    """
    if view not in {"records", "chat", "gwt"}:
        raise ValueError("Use the records, chat, or gwt display hint.")
    records = snapshot.serialize_knoxels()
    members = list(dict.fromkeys(record_ids))
    included = list(dict.fromkeys([*members, *related_ids]))
    if view == "chat" and any(records[key]["data"].get("feature_type") != "Dialogue" for key in members):
        raise ValueError("A chat collection can contain only dialogue records.")
    gwt = snapshot.serialize_gwt()[tick_id] if tick_id is not None else None
    if view == "gwt" and (gwt is None or members != list(gwt["knoxel_ids"])):
        raise ValueError("A GWT collection must contain the selected tick's exact members in order.")
    return {"schema_id": SCHEMA, "title": title, "view": view, "perspective": perspective,
            "captured_at": datetime.now(timezone.utc).isoformat(), "record_ids": members,
            "records": [without_embeddings(records[key]) for key in included],
            "revisions": {str(key): snapshot.revisions[key] for key in included},
            "gwt": without_embeddings(gwt)}


def capture_gwt(snapshot: Snapshot, tick_id: int) -> dict:
    """Capture the selected GWT turn and its exact members. Missing ticks raise."""
    turn = snapshot.serialize_gwt()[tick_id]
    return capture_records(snapshot, turn["knoxel_ids"], title=f"GWT content of tick {tick_id}",
                           view="gwt", tick_id=tick_id)


def dump_package(package: dict, path: Path) -> None:
    """Write a captured collection as UTF-8 JSON. The caller owns its path.

    Filesystem errors propagate. Write only after capture succeeds, and expose
    the file to review after this function returns. No runtime state changes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(without_embeddings(package), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
