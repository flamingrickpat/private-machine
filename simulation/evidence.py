"""Freeze readable runtime evidence beside each simulation attempt.

``capture_participant`` copies one participant's ghost, logs, and runtime folder
into the attempt, then writes the record packages the browser shows: the public
conversation, the private activity, the raw branch history, one package per
cycle, and the transaction ledger. The caller fences routing and drains runnable
work first, so the copy is a quiescent instant rather than a moving target.

Embeddings stay out of the readable packages. A failed attempt is captured too,
because its evidence is what explains the failure. The package overview is in
``simulation/__init__.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

from pydantic import JsonValue

from pm.host.durable_files import atomic_json
from pm.introspection.record_packages import capture_records, capture_cycle, dump_package
from pm.memory.store import MemoryStore


def capture_participant(home: Path, destination: Path, first_record_id: int) -> dict[str, JsonValue]:
    """Capture chat, private activity, exact cycle inputs, logs, and raw history.

    The caller first fences routing and drains runnable work. Failed runtime
    checkpoints remain inspectable. Embeddings stay outside readable packages.
    ``first_record_id`` is the run's baseline record ID: records below it belong
    to the forked source and stay out of the participant's activity packages.
    Returns the relative paths of the captured evidence. Raises
    ``FileExistsError`` when the destination already exists.
    Relative artifact paths use forward slashes for the browser.
    """
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint = home / "ghost.pickle"
    shutil.copyfile(checkpoint, destination / "ghost.pickle")
    for name in ("logs", "runtime"):
        source = home / name
        if source.exists():
            shutil.copytree(source, destination / name, ignore=shutil.ignore_patterns("snapshots", "*.lock"))
    raw = MemoryStore.from_checkpoint(destination / "ghost.pickle", raw=True)
    accepted = MemoryStore.from_checkpoint(destination / "ghost.pickle")
    clean = raw.causal_history.accepted()
    visible = accepted.snapshot(accepted._visible_history)
    new_ids = [key for key in visible.revisions if key >= first_record_id]
    dialogue = [row["id"] for row in accepted.dialogue_history() if row["id"] >= first_record_id]
    dump_package(capture_records(visible, dialogue, title="Public conversation", view="chat", perspective="Jesse"),
                 destination / "chat.json")
    dump_package(capture_records(visible, new_ids, title="Private activity and runtime records"),
                 destination / "activity.json")
    raw_view = raw.snapshot(raw._visible_history)
    dump_package(capture_records(raw_view, [key for key in raw_view.revisions if key >= first_record_id],
                                 title="Raw branch history for debugging"), destination / "raw-activity.json")
    cycles: list[dict[str, JsonValue]] = []
    for index, state in enumerate(accepted.states):
        turn = accepted.gwt_turns.get(state.tick_id)
        if turn is None or not any(key >= first_record_id for key in turn.knoxel_ids):
            continue
        package = capture_cycle(visible, state, previous=accepted.states[index - 1] if index else None)
        path = destination / "cycles" / f"{state.tick_id}.json"
        dump_package(package, path)
        cycles.append({"tick_id": state.tick_id, "slot": turn.slot_id, "summary": turn.summary,
                       "path": path.relative_to(destination).as_posix()})
    transactions: list[dict[str, JsonValue]] = [{**item.model_dump(mode="json"), "accepted": key in clean}
                                                for key, item in raw.causal_history.transactions.items()]
    atomic_json(destination / "transactions.json", {"transactions": transactions})
    return {"home": str(home), "chat": "chat.json", "activity": "activity.json", "cycles": cycles,
            "transactions": "transactions.json", "checkpoint": "ghost.pickle",
            "raw_activity": "raw-activity.json",
            "prompt_files": [path.relative_to(destination).as_posix() for path in (destination / "logs").rglob("*.md")],
            "telemetry": "runtime/telemetry.jsonl"}
