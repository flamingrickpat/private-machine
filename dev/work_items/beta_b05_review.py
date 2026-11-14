"""Build the B05 owner review package from one recorded private-dwelling sample.

The recording keeps the starting store and every model call. This script
replays that sample with no endpoint, then captures from the replayed final
store, so the inner workspace turns are present even though the recording's
checkpoint was written before the derived turn commits.

The package holds, from one frozen snapshot:

- Jesse's short reply and Spamuel's long message;
- every retained nomination and the separate attention decision;
- every executed inquiry operation and the branches it generated;
- each inner workspace turn that projected an operation;
- any felt-state record the run committed.

It writes the packages under ``.runtime/beta-b05/review/`` and prints the
register command. It performs no model call and changes no accepted state.

Usage::

    uv run --locked python -m dev.work_items.beta_b05_review \
        .runtime/replay/private-dwelling-<stamp>/sample-03
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from pm import vclock
from pm.memory.store import MemoryStore
from pm.introspection.record_packages import capture_gwt, capture_records, dump_package
from pm.records.enums import FeatureType
from pm.records.qualia_records import QualeRecord
from pm.utils.operating_system import OperatingSystem
from tests.replay.__main__ import RecordingManifest
from tests.replay.determinism import deterministic_identities
from tests.replay.runtime import open_runtime
from tests.replay.scenarios import scenario_named
from workflows.inner_thought.nominations import NOMINATION_KEY

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".runtime" / "beta-b05" / "review"


def _replayed_store(sample: Path) -> MemoryStore:
    """Replay one recorded sample and return its final store."""
    manifest = RecordingManifest.model_validate_json((sample / "manifest.json").read_text(encoding="utf-8"))
    scenario = scenario_named(manifest.scenario)
    vclock.install(vclock.VirtualClock(manifest.clock_start_utc))
    scratch = sample / "review-replay.pickle"
    shutil.copyfile(sample / "ghost.pickle", scratch)
    store = MemoryStore.from_checkpoint(scratch)
    OperatingSystem.environment["PM_PROMPT_LOG"] = "0"
    runtime = None
    try:
        with deterministic_identities(f"{scenario.name}:{manifest.sample}"):
            runtime = open_runtime(scenario.name, store, sample, mode="replay",
                                   clock_start=manifest.clock_start_utc)
            scenario.run(runtime)
        return runtime.store
    finally:
        if runtime is not None:
            runtime.close()


def build(sample: Path, out: Path) -> list[Path]:
    """Capture the review packages from one replayed sample."""
    store = _replayed_store(sample)
    snapshot = store.snapshot(capture=False)
    features = list(store.all_features)
    dialogues = [record for record in features if record.feature_type == FeatureType.Dialogue]
    reply = next(record for record in dialogues if record.content == "Yeah. Maybe.")
    long_message = next(record for record in dialogues if record.id != reply.id)
    nominations = [record for record in features if NOMINATION_KEY in record.metadata]
    attention = [record for record in features if "thought_attention" in record.metadata]
    operations = [record for record in features
                  if record.feature_type == FeatureType.InnerMonologue
                  and (record.metadata or {}).get("loop") == "inquiry"]
    operation_ids = {int(record.id) for record in operations}
    branches = [record for record in features
                if record.feature_type == FeatureType.Thought and record.lineage is not None
                and record.lineage.operation == "inquiry_alternative"
                and operation_ids.intersection(record.lineage.parent_record_ids)]
    qualia = [record for record in store.all_knoxels.values() if isinstance(record, QualeRecord)]
    inner_turns = [turn for turn in store.gwt_turns.values() if turn.slot_id == "inner"]

    ordered: list[int] = [int(reply.id)]
    for group in (nominations, attention):
        ordered.extend(int(record.id) for record in group)
    for operation in operations:
        ordered.append(int(operation.id))
        ordered.extend(int(branch.id) for branch in branches
                       if int(operation.id) in branch.lineage.parent_record_ids)
    ordered.extend(int(record.id) for record in qualia)
    related = [int(long_message.id)]

    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    records_path = out / "b05-dwelling-records.json"
    dump_package(capture_records(snapshot, ordered, title="B05: dwelling on a short reply",
        view="records", perspective="Jesse", related_ids=related,
        tick_id=max(inner_turns, key=lambda turn: turn.tick_id).tick_id if inner_turns else None),
        records_path)
    written.append(records_path)
    for turn in sorted(inner_turns, key=lambda item: item.tick_id):
        path = out / f"b05-dwelling-inner-{turn.tick_id}.json"
        dump_package(capture_gwt(snapshot, turn.tick_id), path)
        written.append(path)
    return written


def main() -> int:
    """Capture one sample and print the register command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path)
    arguments = parser.parse_args()
    written = build(arguments.sample, OUT)
    for path in written:
        print(path)
    print("\nRegister with:")
    print("uv run --locked python -m simulation.review register-package "
          + " ".join(str(path) for path in written)
          + " --combine --id b05-dwelling-20261114 --title \"B05: dwelling on a short reply\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
