"""Capture named record collections without changing accepted runtime state.

Callers choose the collection and its intended reader. A display hint does not
grant access to private records. Embeddings are omitted from exported copies;
timestamps, appraisal fields, codelet metadata, and other fields stay intact.

A ``cycle`` collection (:func:`capture_cycle`) holds one accepted cycle
state: the CSM items with their activation and endorsers, the workspace
shards with their items and tails, the codelet registry state, the stimulus
groups, the decisions, and the dispatches. It also holds the turn record of
the tick (shard changes, codelet runs, structures, the decision record) and
every record that the state or the turn names.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Literal

from pm.memory.snapshot import Snapshot
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord

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
                    view: Literal["records", "chat", "gwt", "cycle"] = "records",
                    perspective: str = "developer", related_ids: Iterable[int] = (),
                    tick_id: int | None = None) -> dict:
    """Return a detached collection from one frozen snapshot.

    Keep caller order and remove duplicate ids. Related sources accompany the
    selected records without becoming collection members. Missing records or
    GWT ticks raise KeyError. A chat collection rejects private record types.
    This function performs no inference, file writes, or memory commits.
    """
    if view not in {"records", "chat", "gwt", "cycle"}:
        raise ValueError("Use the records, chat, gwt, or cycle display hint.")
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


def cycle_record_ids(state: CycleState, turn: GwtTurnRecord | None) -> list[int]:
    """Return every record id that one cycle state and its turn record name, in a stable order.

    The order is: the claimed stimuli, the workspace items and tails, the CSM
    items and their merged records, the structures, the codelet percepts of
    the registry, the dispatched actions, the broadcast experiences, and the
    records of the turn (created percepts and structures, the experience, and
    the decision record).
    """
    ids = [*state.claimed_ids(), *(key for decision in state.decisions for key in decision.stimulus_ids),
           *state.workspace.item_ids(), *state.workspace.tail_ids()]
    for item in state.csm.items.values():
        ids.extend([item.record_id, *item.merged_ids])
    ids.extend(state.structure_ids)
    for runtime in state.codelets.codelets.values():
        ids.extend([*runtime.last_percept_ids, *runtime.precursor_ids])
    ids.extend(ref.action_id for ref in state.dispatches if ref.action_id is not None)
    ids.extend(ref.experience_id for ref in state.broadcasts if ref.experience_id is not None)
    if turn is not None:
        ids.extend([*turn.trace.percept_ids, *turn.trace.structure_ids,
                    *(key for key in (turn.experience_id, turn.decision_id) if key is not None)])
    return list(dict.fromkeys(ids))


def capture_cycle(snapshot: Snapshot, state: CycleState, *, title: str = "",
                  perspective: str = "developer") -> dict:
    """Return a ``cycle`` collection of one accepted cycle state from one frozen snapshot.

    ``state`` is the cycle state of the store for the same tick as the
    snapshot. The collection members are the records that the state and its
    turn record name (:func:`cycle_record_ids`). ``cycle`` holds the state
    itself. A named record that the snapshot lacks raises KeyError. The
    function performs no inference, file writes, or memory commits.
    """
    gwt = snapshot.serialize_gwt()
    turn = GwtTurnRecord.model_validate(gwt[state.tick_id]) if state.tick_id in gwt else None
    package = capture_records(snapshot, cycle_record_ids(state, turn), title=title or f"Cycle {state.tick_id}",
                              view="cycle", perspective=perspective,
                              tick_id=state.tick_id if turn is not None else None)
    package["cycle"] = without_embeddings(state.model_dump(mode="json"))
    return package


def dump_package(package: dict, path: Path) -> None:
    """Write a captured collection as UTF-8 JSON. The caller owns its path.

    Filesystem errors propagate. Write only after capture succeeds, and expose
    the file to review after this function returns. No runtime state changes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(without_embeddings(package), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
