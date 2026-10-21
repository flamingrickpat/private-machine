"""Capture named record collections without changing accepted runtime state.

Callers choose the collection and its intended reader. A display hint does not
grant access to private records. Embeddings are omitted from exported copies;
timestamps, appraisal fields, codelet metadata, and other fields stay intact.

A ``cycle`` collection (:func:`capture_cycle`) holds one accepted cycle
state: the CSM items with their activation and endorsers, the workspace
shards with their items and tails, the codelet registry state, the stimulus
groups, the decisions, and the dispatches. It also holds the turn record of
the tick (shard changes, codelet runs, structures, the decision record) and
every record that the state or the turn names. With the previous cycle
state, it also holds the mental state before the tick. ``decoded_vectors``
names the axes of each mental-state vector of a member record, so a reader
sees the mini-appraisal of a codelet percept without the vector layout.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Literal

from pm.memory.snapshot import Snapshot
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.mental_state import (
    AppraisalGeneral,
    AppraisalSocial,
    StateCognition,
    StateCore,
    StateEmotions,
    StateExpressionExt,
    StateNeeds,
    StateNeurochemical,
    StateRelationship,
)

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


#: The parts of a mental-state vector, in the order of ``FullMentalState.to_list``.
VECTOR_PARTS = (("appraisal_general", AppraisalGeneral), ("appraisal_social", AppraisalSocial),
                ("state_neurochemical", StateNeurochemical), ("state_core", StateCore),
                ("state_emotions", StateEmotions), ("state_cognition", StateCognition), ("state_needs", StateNeeds),
                ("state_relationship", StateRelationship), ("state_expression_ext", StateExpressionExt))
#: Values with a smaller magnitude count as zero in a decoded vector.
DECODE_EPSILON = 1e-6


def decode_vector(vector: list[float]) -> dict[str, float]:
    """Return the nonzero axes of one mental-state vector by name, for example ``state_emotions.joy``.

    The positions come from the state models. The relationship part names
    its entity in ``entity_id``; that position is not an axis and stays out.
    """
    named: dict[str, float] = {}
    for part, model in VECTOR_PARTS:
        for name, position in model.get_vector_positions().items():
            if name == "entity_id" or position >= len(vector):
                continue
            value = float(vector[position])
            if abs(value) > DECODE_EPSILON:
                named[f"{part}.{name}"] = round(value, 4)
    return named


def decoded_vectors(records: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    """Return the decoded appraisal and delta vectors of each serialized record that has them, by record id."""
    decoded = {}
    for wrapped in records:
        data = wrapped["data"]
        if "mental_state_appraisal_vec" not in data and "mental_state_delta_vec" not in data:
            continue
        decoded[str(data["id"])] = {
            "appraisal": decode_vector(data["mental_state_appraisal_vec"]) if "mental_state_appraisal_vec" in data else {},
            "delta": decode_vector(data["mental_state_delta_vec"]) if "mental_state_delta_vec" in data else {}}
    return decoded


def cycle_record_ids(state: CycleState, turn: GwtTurnRecord | None) -> list[int]:
    """Return every record id that one cycle state and its turn record name, in a stable order.

    The order is: the claimed stimuli, the workspace items and tails, the CSM
    items and their merged records, the structures, the codelet percepts of
    the registry, the dispatched actions, the broadcast experiences, and the
    records of the turn (created percepts and structures, the lines that the
    codelet selector read as echoes, the experience, the decision record, and
    every other record that the turn names: the appraisal, the goals, the
    expectations, the actions, and the selected memories).
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
        ids.extend([*turn.trace.percept_ids, *turn.trace.structure_ids, *turn.trace.echo_ids,
                    *(key for key in (turn.experience_id, turn.decision_id) if key is not None),
                    *turn.knoxel_ids])
    return list(dict.fromkeys(ids))


def capture_cycle(snapshot: Snapshot, state: CycleState, *, title: str = "",
                  perspective: str = "developer", previous: CycleState | None = None) -> dict:
    """Return a ``cycle`` collection of one accepted cycle state from one frozen snapshot.

    ``state`` is a cycle state of the store, and the snapshot holds its
    records. The collection members are the records that the state and its
    turn record name (:func:`cycle_record_ids`). ``cycle`` holds the state
    itself. ``previous`` is the cycle state before ``state``; its mental
    state becomes ``mental_state_before``. ``decoded_vectors`` names the
    axes of the mental-state vectors of the members (:func:`decode_vector`).
    ``entity_names`` maps the id of each entity shard to the entity name.
    A named record that the snapshot lacks raises KeyError. The function
    performs no inference, file writes, or memory commits.
    """
    gwt = snapshot.serialize_gwt()
    turn = GwtTurnRecord.model_validate(gwt[state.tick_id]) if state.tick_id in gwt else None
    package = capture_records(snapshot, cycle_record_ids(state, turn), title=title or f"Cycle {state.tick_id}",
                              view="cycle", perspective=perspective,
                              tick_id=state.tick_id if turn is not None else None)
    package["cycle"] = without_embeddings(state.model_dump(mode="json"))
    package["mental_state_before"] = (without_embeddings(previous.mental_state.model_dump(mode="json"))
                                      if previous is not None else None)
    package["decoded_vectors"] = decoded_vectors(package["records"])
    # The reader names each entity shard by its entity, not by its id.
    package["entity_names"] = {str(key): snapshot.get_entity_name(key) for key in state.workspace.entity_shards}
    return package


def dump_package(package: dict, path: Path) -> None:
    """Write a captured collection as UTF-8 JSON. The caller owns its path.

    Filesystem errors propagate. Write only after capture succeeds, and expose
    the file to review after this function returns. No runtime state changes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(without_embeddings(package), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
