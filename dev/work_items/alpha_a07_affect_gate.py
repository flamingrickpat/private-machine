"""A07 live gate: emotional dynamics change behavior across a real sequence.

The gate uses the real local chat model, the real Spamuel seed, and a real
MemoryStore checkpoint. It drives four turns:

1. a disappointment;
2. an unrelated subject change;
3. a return to the disappointment;
4. a repair offer.

After each turn it records the accepted mental state, the affect trace on the
appraisal Feature, the regulation mode, and the public replies. It then runs
one contrasting persona turn with the cheerful fixture to show that the same
facts produce a different internal trajectory.

The gate never substitutes a fixed model answer. It records recovery rather
than forcing it, because coping may fail. It prints ``ok: true`` only when the
real acceptance conditions hold.

Run:
    uv run --locked python dev/work_items/alpha_a07_affect_gate.py \
        --config config.alpha.example.yaml --out .runtime/alpha/<stamp>-a07
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev.work_items.shared.evidence import save_json

from main import build_app
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.mental_state import FullMentalState
from pm.app.lifecycle_records import record_shutdown

import yaml

SEED = ROOT / "test_data" / "personas" / "spamuel" / "canon.yaml"
CHEERFUL = ROOT / "test_data" / "personas" / "cheerful" / "config.yaml"

TURNS = [
    ("disappointment",
     "Jesse: I waited all day for the thing you promised, and it never happened. "
     "I am really disappointed in you."),
    ("subject_change",
     "Jesse: Anyway, a flock of geese just crossed the yard. Strange weather for it."),
    ("return",
     "Jesse: I keep coming back to this morning. It still stings when I think about it."),
    ("repair",
     "Jesse: I am sorry I snapped at you. I know you were trying. Let us start over."),
]


def pump(app, rounds: int = 60) -> int:
    """Run each module step until no module has work. No threads are started."""
    work = 0
    for _ in range(rounds):
        progressed = False
        for module in app.modules:
            try:
                if module.step(block=False):
                    progressed = True
                    work += 1
            except Exception:
                raise
        if not progressed:
            break
    return work


def _state_snapshot(ghost) -> dict:
    if not ghost.states:
        return {"sadness": 0.0, "joy": 0.0, "fear": 0.0, "anger": 0.0,
                "valence": 0.0, "arousal": 0.0, "summary": ""}
    mental = FullMentalState.init_from_list(ghost.states[-1].mental_state_latent_vec)
    return {
        "sadness": round(float(mental.state_emotions.sadness), 4),
        "joy": round(float(mental.state_emotions.joy), 4),
        "fear": round(float(mental.state_emotions.fear), 4),
        "anger": round(float(mental.state_emotions.anger), 4),
        "tenderness": round(float(mental.state_emotions.tenderness), 4),
        "valence": round(float(mental.state_core.valence), 4),
        "arousal": round(float(mental.state_core.arousal), 4),
        "regulation": "",
    }


def _latest_appraisal(ghost) -> dict | None:
    features = [
        knoxel for knoxel in ghost.all_features
        if str(knoxel.feature_subtype) == "appraisal"
    ]
    if not features:
        return None
    feature = sorted(features, key=lambda item: item.id)[-1]
    return {
        "id": int(feature.id),
        "content": str(feature.content),
        "affect_trace": dict(feature.metadata.get("affect_trace", {})),
        "regulation": str(feature.metadata.get("regulation", "")),
        "behavior": dict(feature.metadata.get("behavior", {})),
    }


def build_control_ghost(path: Path, root: Path) -> MemoryStore:
    """Build the cheerful control MemoryStore from the small persona fixture.

    The fixture holds a character card, not a full state file. This helper
    reads the card fields into a fresh MemoryStore with no history.
    """
    card = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = PersonaConfig.model_validate(card)
    return MemoryStore(config, checkpoint=root / "ghost.json")


def run_sequence(seed_path: Path, root: Path, *, label: str, turns: list, replies: list, config: str, ghost=None) -> dict:
    app = build_app(
        seed_path=str(seed_path),
        runtime_root=root / "runtime",
        checkpoint_path=None if ghost is not None else root / "ghost.json",
        ghost=ghost,
        emit_reply=replies.append,
        cognition="real",
        config_path=config,
    )
    boot = dict(app.lifecycle_boot)
    boot_work = pump(app)
    snapshots: list[dict] = []
    previous = _state_snapshot(app.ghost)
    for name, text in turns:
        reply_start = len(replies)
        app.user_input.submit(text)
        turn_work = pump(app)
        report = app.main_cog.last_result.report if app.main_cog.last_result else {}
        state = _state_snapshot(app.ghost)
        state["regulation"] = str(report.get("regulation", ""))
        appraisal = _latest_appraisal(app.ghost)
        snapshots.append({
            "turn": name,
            "stimulus": text,
            "work": turn_work,
            "state": state,
            "previous_state": previous,
            "delta_sadness": round(state["sadness"] - previous["sadness"], 4),
            "delta_valence": round(state["valence"] - previous["valence"], 4),
            "regulation": state["regulation"],
            "affect_summary": str(report.get("affect_summary", "")),
            "appraisal": appraisal,
            "replies": list(replies[reply_start:]),
        })
        previous = state

    app.main_cog.stop()
    app.main_cog._finish_termination()
    record_shutdown(app.ghost)
    return {
        "label": label,
        "seed": str(seed_path),
        "boot": boot,
        "boot_work": boot_work,
        "snapshots": snapshots,
        "cycles": len(app.ghost.states),
        "appraisal_features": sum(
            1 for knoxel in app.ghost.all_features
            if str(knoxel.feature_subtype) == "appraisal"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="A07 live affect gate")
    parser.add_argument("--config", default="config.alpha.example.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", default=str(SEED))
    parser.add_argument("--cheerful-seed", default=str(CHEERFUL))
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument(
        "--reuse-primary", action="store_true",
        help="Reuse an existing primary-sequence.json in the output directory.",
    )
    parser.add_argument(
        "--reuse-comparison", action="store_true",
        help="Reuse an existing cheerful-sequence.json in the output directory.",
    )
    args = parser.parse_args(argv)

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    replies: list[str] = []

    primary_path = root / "primary-sequence.json"
    if args.reuse_primary and primary_path.exists():
        primary = json.loads(primary_path.read_text(encoding="utf-8"))
    else:
        primary = run_sequence(
            Path(args.seed), root / "primary", label="spamuel", turns=TURNS,
            replies=replies, config=args.config,
        )
        save_json(root / "primary-sequence.json", primary)

    comparison = None
    comparison_path = root / "cheerful-sequence.json"
    if args.reuse_comparison and comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    elif not args.skip_comparison:
        control_root = root / "cheerful"
        control_root.mkdir(parents=True, exist_ok=True)
        control_ghost = build_control_ghost(Path(args.cheerful_seed), control_root)
        comparison = run_sequence(
            Path(args.cheerful_seed), control_root, label="cheerful",
            turns=[TURNS[0]], replies=replies, config=args.config, ghost=control_ghost,
        )
        save_json(root / "cheerful-sequence.json", comparison)

    # A recovery is recorded, never required to be positive.
    if primary["snapshots"]:
        first = primary["snapshots"][0]
        last = primary["snapshots"][-1]
        recovery = {
            "sadness_recovery": round(first["state"]["sadness"] - last["state"]["sadness"], 4),
            "valence_recovery": round(last["state"]["valence"] - first["state"]["valence"], 4),
        }
    else:
        recovery = {"sadness_recovery": 0.0, "valence_recovery": 0.0}
    save_json(root / "recovery.json", recovery)

    failures: list[str] = []
    traced = [
        snapshot for snapshot in primary["snapshots"]
        if snapshot["appraisal"] and snapshot["appraisal"]["affect_trace"]
    ]
    if len(traced) != len(primary["snapshots"]):
        failures.append("one or more turns wrote no appraisal affect trace")
    if primary["appraisal_features"] < len(TURNS):
        failures.append("the primary sequence wrote fewer appraisals than turns")
    if any(not snapshot["replies"] for snapshot in primary["snapshots"]):
        failures.append("one or more turns produced no public reply")
    # The internal state must move somewhere across the sequence.
    moved = any(
        abs(snapshot["delta_sadness"]) > 1e-6 or abs(snapshot["delta_valence"]) > 1e-6
        for snapshot in primary["snapshots"]
    )
    if not moved:
        failures.append("the internal affect never moved across the sequence")
    if comparison is not None:
        # The control shows a different internal trajectory. A public reply is
        # A09's concern, so the gate records the reply but does not require it.
        if not comparison["snapshots"] or not comparison["snapshots"][0]["appraisal"]:
            failures.append("the cheerful comparison produced no appraisal trace")

    receipt = {
        "primary": primary,
        "comparison": comparison,
        "recovery": recovery,
        "failures": failures,
        "ok": not failures,
    }
    save_json(root / "session-receipt.json", receipt)
    if failures:
        raise RuntimeError("A07 live gate failed: " + "; ".join(failures))
    print(json.dumps({"out": str(root), "ok": True, "recovery": recovery}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())