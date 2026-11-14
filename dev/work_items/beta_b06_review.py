"""Build the B06 owner review packages from recorded ToM and dwelling samples.

Each recording keeps the starting store and every model call. This script
replays a copy of the sample with no endpoint, then captures from the
replayed final store, so the inner workspace turns are present even though the
recording's checkpoint was written before the derived turn commits.

The perspective-morning package holds: Jesse's synthetic morning lines, the
theory-of-mind action, every executed inquiry operation with its recorded
reason and progress, the finding and any perspective revision it kept, the
recorded outcome facts (revision claims, the real owner's rehearsal
dispositions, evidence-request counts, published action requests and public
utterances), and the inner turns of the episode.

The private-dwelling package holds: Jesse's ambiguous reply and Spamuel's long
message, every retained nomination and attention decision, every executed
operation, each noticing step with its recorded activity facts, any felt-state
record, the outcome's noticing readout, and the inner turns.

It writes the packages under ``.runtime/beta-b06/review/`` and prints the
register command. It performs no model call and changes no accepted state.

Usage::

    uv run --locked python -m dev.work_items.beta_b06_review \
        .runtime/replay/perspective-morning-<stamp>/sample-03 \
        .runtime/replay/private-dwelling-<stamp>/sample-01
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pm import vclock
from pm.memory.store import MemoryStore
from pm.introspection.record_packages import capture_gwt, capture_records, dump_package
from pm.records.enums import FeatureType
from pm.records.memory_records import ActionSchemaRecord
from pm.records.qualia_records import QualeRecord
from pm.utils.operating_system import OperatingSystem
from tests.replay.__main__ import RecordingManifest
from tests.replay.determinism import deterministic_identities
from tests.replay.runtime import open_runtime
from tests.replay.scenarios import scenario_named
from workflows.inner_thought.metacognition import METACOGNITION_OPERATIONS
from workflows.inner_thought.nominations import NOMINATION_KEY

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".runtime" / "beta-b06" / "review"


def _replayed_store(sample: Path) -> MemoryStore:
    """Replay one recorded sample on a copy and return its final store."""
    scratch_dir = OUT / "replays" / sample.parent.name / sample.name
    scratch_dir.mkdir(parents=True, exist_ok=True)
    for item in sample.iterdir():
        target = scratch_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copyfile(item, target)
    manifest = RecordingManifest.model_validate_json(
        (scratch_dir / "manifest.json").read_text(encoding="utf-8"))
    scenario = scenario_named(manifest.scenario)
    vclock.install(vclock.VirtualClock(manifest.clock_start_utc))
    scratch = scratch_dir / "review-replay.pickle"
    shutil.copyfile(scratch_dir / "ghost.pickle", scratch)
    store = MemoryStore.from_checkpoint(scratch)
    OperatingSystem.environment["PM_PROMPT_LOG"] = "0"
    runtime = None
    try:
        with deterministic_identities(f"{scenario.name}:{manifest.sample}"):
            runtime = open_runtime(scenario.name, store, scratch_dir, mode="replay",
                                   clock_start=manifest.clock_start_utc)
            scenario.run(runtime)
        return runtime.store
    finally:
        if runtime is not None:
            runtime.close()


def _operation_rows(features: list) -> tuple[list, list]:
    """Return executed inquiry operations, and the noticing steps among them."""
    operations = [record for record in features
                  if record.feature_type == FeatureType.InnerMonologue
                  and (record.metadata or {}).get("loop") == "inquiry"]
    operations.sort(key=lambda record: record.id)
    noticing = [record for record in operations if _operation_of(record) in METACOGNITION_OPERATIONS]
    return operations, noticing


def _operation_of(record) -> str:
    """Return the operation name from the record's stored step payload."""
    step = (record.metadata or {}).get("inquiry_step") or {}
    if isinstance(step, dict) and "operation" in step:
        return str(step["operation"])
    return (record.lineage.operation if record.lineage is not None else "")


def _inner_turns(store: MemoryStore) -> list:
    """Return the inner GWT turns in tick order."""
    return sorted((turn for turn in store.gwt_turns.values() if turn.slot_id == "inner"),
                  key=lambda item: item.tick_id)


def build_perspective(sample: Path, out: Path) -> list[Path]:
    """Capture the morning-routine theory-of-mind review package."""
    store = _replayed_store(sample)
    snapshot = store.snapshot(capture=False)
    features = list(store.all_features)
    lines = [record for record in features
             if record.feature_type == FeatureType.Dialogue
             and record.provenance is not None
             and record.provenance.source_ref == "replay-synthetic-situation"]
    actions = [record for record in store.all_knoxels.values() if isinstance(record, ActionSchemaRecord)]
    operations, _ = _operation_rows(features)
    findings = [record for record in operations
                if (record.metadata or {}).get("private_inquiry_finding") is True]
    outcome = json.loads((sample / "outcome.json").read_text(encoding="utf-8"))
    ordered: list[int] = [int(record.id) for record in lines]
    for action in actions:
        ordered.append(int(action.id))
    for operation in operations:
        ordered.append(int(operation.id))
    package = capture_records(snapshot, ordered,
        title="B06: morning-routine theory of mind", view="records", perspective="Jesse",
        related_ids=sorted({int(source_id)
                            for record in operations
                            for source_id in (record.metadata or {}).get("inquiry_step", {}).get("evidence_ids", [])}))
    package["facts"] = {
        "deliberate_operations": outcome.get("deliberate_operations"),
        "deliberate_status": outcome.get("deliberate_status"),
        "finding_id": outcome.get("finding_id"),
        "mental_status": outcome.get("mental_status"),
        "thought_requests": outcome.get("thought_requests"),
        "thought_requests_with_perspective": outcome.get("thought_requests_with_perspective"),
        "revision_claims": outcome.get("revision_claims"),
        "owner_dispositions": outcome.get("dispositions"),
        "has_observation_claim": outcome.get("has_observation_claim"),
        "action_requests": outcome.get("action_requests"),
        "public_utterances": outcome.get("public_utterances"),
    }
    written: list[Path] = []
    path = out / "b06-perspective-morning.json"
    dump_package(package, path)
    written.append(path)
    for turn in _inner_turns(store):
        path = out / f"b06-perspective-morning-inner-{turn.tick_id}.json"
        dump_package(capture_gwt(snapshot, turn.tick_id), path)
        written.append(path)
    return written


def build_dwelling(sample: Path, out: Path) -> list[Path]:
    """Capture the dwelling-and-noticing review package."""
    store = _replayed_store(sample)
    snapshot = store.snapshot(capture=False)
    features = list(store.all_features)
    dialogues = [record for record in features if record.feature_type == FeatureType.Dialogue]
    reply = next(record for record in dialogues if record.content == "Yeah. Maybe.")
    long_message = next(record for record in dialogues if record.id != reply.id)
    nominations = [record for record in features if NOMINATION_KEY in record.metadata]
    attention = [record for record in features if "thought_attention" in record.metadata]
    operations, noticing = _operation_rows(features)
    operation_ids = {int(record.id) for record in operations}
    branches = [record for record in features
                if record.feature_type == FeatureType.Thought and record.lineage is not None
                and record.lineage.operation == "inquiry_alternative"
                and operation_ids.intersection(record.lineage.parent_record_ids)]
    qualia = [record for record in store.all_knoxels.values() if isinstance(record, QualeRecord)]
    outcome = json.loads((sample / "outcome.json").read_text(encoding="utf-8"))
    ordered: list[int] = [int(reply.id)]
    for group in (nominations, attention):
        ordered.extend(int(record.id) for record in group)
    for operation in operations:
        ordered.append(int(operation.id))
        ordered.extend(int(branch.id) for branch in branches
                       if int(operation.id) in branch.lineage.parent_record_ids)
    ordered.extend(int(record.id) for record in qualia)
    package = capture_records(snapshot, ordered,
        title="B06: noticing dwelling with recorded activity", view="records", perspective="Jesse",
        related_ids=[int(long_message.id)])
    package["facts"] = {
        "noticing_steps": outcome.get("noticing_steps"),
        "repeated_concern": outcome.get("repeated_concern"),
        "stop_reason": outcome.get("stop_reason"),
    }
    written: list[Path] = []
    path = out / "b06-private-dwelling.json"
    dump_package(package, path)
    written.append(path)
    for turn in _inner_turns(store):
        path = out / f"b06-private-dwelling-inner-{turn.tick_id}.json"
        dump_package(capture_gwt(snapshot, turn.tick_id), path)
        written.append(path)
    return written


def main() -> int:
    """Capture one perspective and one dwelling sample, then print registration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("perspective_sample", type=Path,
                        help="Sample directory of a perspective-morning recording.")
    parser.add_argument("dwelling_sample", type=Path,
                        help="Sample directory of a private-dwelling recording.")
    arguments = parser.parse_args()
    out = OUT
    perspective = build_perspective(arguments.perspective_sample, out)
    dwelling = build_dwelling(arguments.dwelling_sample, out)
    for path in [*perspective, *dwelling]:
        print(path)
    print("\nRegister with (two captures; the samples are separate stores):")
    print("uv run --locked python -m simulation.review register-package "
          + " ".join(str(path) for path in perspective)
          + " --combine --id b06-tom-20261114 "
          + "--title \"B06: morning-routine theory of mind\"")
    print("uv run --locked python -m simulation.review register-package "
          + " ".join(str(path) for path in dwelling)
          + " --combine --id b06-metacognition-20261114 "
          + "--title \"B06: noticing dwelling with recorded activity\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
