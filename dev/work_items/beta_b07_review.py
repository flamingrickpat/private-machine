"""Build the B07 owner review packages from passed live-gate runs.

Each live attempt of ``tests/live/test_interval_appraisal.py`` keeps the
store the real application wrote. This script captures from a copy of that
store: the dialogue, the current head of every goal and expectation (each
names the version it superseded), the actions, any committed wait, and every
main workspace turn. It performs no model call and changes no accepted state.

Usage::

    uv run --locked python -m dev.work_items.beta_b07_review \
        --part-b .runtime/beta-b07/integration/<stamp>-part-b-attempt-1 \
        --part-c .runtime/beta-b07/integration/<stamp>-part-c-attempt-1
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pm.introspection.record_packages import capture_gwt, capture_records, dump_package
from pm.memory.store import MemoryStore
from pm.records.cognitive_records import ExpectationRecord, Intention
from pm.records.enums import FeatureType
from pm.records.memory_records import ActionSchemaRecord, Feature

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".runtime/beta-b07/review"


def build_live_run(evidence: Path, *, name: str, title: str) -> list[Path]:
    """Capture one passed live attempt from a copy of its final store."""
    scratch_dir = OUT / "live" / evidence.name
    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch = scratch_dir / "ghost.pickle"
    shutil.copyfile(evidence / "ghost.pickle", scratch)
    store = MemoryStore.from_checkpoint(scratch)
    snapshot = store.snapshot(capture=False)
    # A package holds accepted heads only; superseded goal versions are named
    # by the heads' supersedes_id and shown in the logged prompts of the run.
    records = snapshot.all_knoxels
    ordered: list[int] = []
    for record_id, record in records.items():
        is_dialogue = isinstance(record, Feature) and record.feature_type == FeatureType.Dialogue
        is_wait = record.metadata.get("control_kind") == "inference_wait"
        if is_dialogue or is_wait or isinstance(record, (Intention, ExpectationRecord, ActionSchemaRecord)):
            ordered.append(int(record_id))
    ordered.sort()
    package = capture_records(snapshot, ordered, title=title, view="records", perspective="Jesse")
    written: list[Path] = []
    path = OUT / f"{name}.json"
    dump_package(package, path)
    written.append(path)
    for turn in sorted(store.gwt_turns.values(), key=lambda item: item.tick_id):
        if turn.slot_id != "main":
            continue
        path = OUT / f"{name}-main-{turn.tick_id}.json"
        superseded = [record_id for record_id in turn.knoxel_ids if record_id not in snapshot.all_knoxels]
        if superseded:
            # capture_gwt needs every member as a head. A main turn that checked
            # an expectation names the version it replaced, so it is captured
            # as a record collection of its current members, with the gap named.
            members = [record_id for record_id in turn.knoxel_ids if record_id in snapshot.all_knoxels]
            package = capture_records(snapshot, members, view="records", perspective="developer",
                title=f"Main turn {turn.tick_id}: {turn.summary[:80]} (superseded members not shown: {superseded})")
        else:
            package = capture_gwt(snapshot, turn.tick_id)
        dump_package(package, path)
        written.append(path)
    return written


def main() -> int:
    """Capture the passed Part B and Part C runs, then print the register commands."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--part-b", type=Path, required=True, help="Evidence folder of a passed Part B attempt.")
    parser.add_argument("--part-c", type=Path, required=True, help="Evidence folder of a passed Part C attempt.")
    parser.add_argument("--stamp", default="20261115", help="Date suffix of the capture ids.")
    arguments = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    captures = [
        (f"b07-attribution-live-{arguments.stamp}", "B07: the interval review shows a reply under the goal it was chosen for",
         build_live_run(arguments.part_b, name="b07-live-part-b",
                        title="B07: the interval review shows a reply under the goal it was chosen for")),
        (f"b07-promise-live-{arguments.stamp}", "B07: a conflicting wish against a kept promise",
         build_live_run(arguments.part_c, name="b07-live-part-c",
                        title="B07: a conflicting wish against a kept promise")),
    ]
    print("\nRegister with (one capture for each store):")
    for capture_id, title, paths in captures:
        print("uv run --locked python -m simulation.review register-package "
              + " ".join(str(path) for path in paths) + f" --combine --id {capture_id} --title \"{title}\"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
