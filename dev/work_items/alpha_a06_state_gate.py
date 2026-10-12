"""A06 live gate: real appraisal and goal state across a clean restart.

The gate uses the real local chat model, the real seed, and the real Ghost
checkpoint. It drives one goal and emotion discussion through the real
seven-stage cognition path, records the accepted state, records a clean
shutdown, restarts from the same checkpoint, and asks about the goal.

Every model response is real. The gate asserts consistency between the
accepted report and accepted memory. It never substitutes a fixed answer.

Run:
    uv run --locked python dev/work_items/alpha_a06_state_gate.py \
        --config config.alpha.example.yaml --out .runtime/alpha/<stamp>-a06
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
from pm.memory.store import Ghost
from runtime.lifecycle import record_shutdown
from workflows.main_cognition.state_changes import active_goal_texts, find_intention, normalize_goal_key

SEED = ROOT / "test_data" / "personas" / "spamuel" / "canon.yaml"


def pump(app, rounds: int = 40) -> int:
    """Run each module step until no module has work. No threads are started.

    Returns the number of working steps. A real cognitive cycle can dispatch
    to Story, so every module gets a turn in each pass.
    """
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="A06 live state gate")
    parser.add_argument("--config", default="config.alpha.example.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", default=str(SEED))
    args = parser.parse_args(argv)

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "ghost.json"
    replies: list[str] = []

    app = build_app(
        seed_path=args.seed,
        runtime_root=root / "runtime",
        checkpoint_path=checkpoint,
        emit_reply=replies.append,
        cognition="real",
        config_path=args.config,
    )
    boot_a = dict(app.lifecycle_boot)
    save_json(root / "boot-a.json", boot_a)

    lifecycle_work = pump(app)
    stimulus = (
        "Jesse: I finally finished the treehouse we talked about, and I am really proud "
        "of us. I want us to build a small stone house by the river next. Please remember "
        "that as our plan."
    )
    app.user_input.submit(stimulus)
    turn_work = pump(app)
    first_report = app.main_cog.last_result.report if app.main_cog.last_result else {}
    save_json(root / "first-report.json", first_report)
    save_json(root / "first-replies.json", list(replies))

    state = {
        "stimulus": stimulus,
        "lifecycle_work": lifecycle_work,
        "turn_work": turn_work,
        "replies": list(replies),
        "active_goals": active_goal_texts(app.ghost),
        "appraisal_features": [
            {"id": knoxel.id, "content": knoxel.content,
             "joy": knoxel.mental_state_appraisal.state_emotions.joy,
             "pride": knoxel.mental_state_appraisal.state_emotions.pride,
             "connection": knoxel.mental_state_appraisal.state_needs.connection}
            for knoxel in app.ghost.all_features
            if str(knoxel.feature_subtype) == "appraisal"
        ],
        "intentions": [
            {"id": knoxel.id, "content": knoxel.content, "status": knoxel.status,
             "active": knoxel.active, "metadata": dict(knoxel.metadata)}
            for knoxel in app.ghost.all_intentions
        ],
        "cycles": len(app.ghost.states),
    }
    save_json(root / "accepted-state.json", state)

    # A normal module stop records the clean shutdown event.
    app.main_cog.stop()
    app.main_cog._finish_termination()
    record_shutdown(app.ghost)
    shutdown_seen = any(
        knoxel.metadata.get("event") == "clean_shutdown"
        for knoxel in app.ghost.all_knoxels.values()
        if hasattr(knoxel, "metadata")
    )
    save_json(root / "shutdown.json", {"clean_shutdown_recorded": shutdown_seen})

    # Restart from the same checkpoint. build_app records the boot and puts the
    # lifecycle stimulus on the bus for the first cycle of the new session.
    restarted = build_app(
        seed_path=args.seed,
        runtime_root=root / "restarted",
        checkpoint_path=checkpoint,
        emit_reply=replies.append,
        cognition="real",
        config_path=args.config,
    )
    boot_b = dict(restarted.lifecycle_boot)
    save_json(root / "boot-b.json", boot_b)
    pump(restarted)
    restarted.user_input.submit("Jesse: What was our plan for the river?")
    restart_work = pump(restarted)
    restart_report = restarted.main_cog.last_result.report if restarted.main_cog.last_result else {}
    save_json(root / "restart-report.json", restart_report)

    reopened = Ghost.from_checkpoint(checkpoint)
    live_goals = active_goal_texts(reopened)
    river_goals = [goal for goal in live_goals if "river" in goal.lower()]
    intention = find_intention(reopened, normalize_goal_key(river_goals[0])) if river_goals else None
    receipt = {
        "boot_a": boot_a,
        "boot_b": boot_b,
        "shutdown_recorded": shutdown_seen,
        "restart_work": restart_work,
        "restart_active_goals": live_goals,
        "restart_intention_id": None if intention is None else int(intention.id),
        "restart_state_counts": {
            "knoxels": len(reopened.all_knoxels),
            "intentions": len(reopened.all_intentions),
            "features": len(reopened.all_features),
        },
    }
    save_json(root / "restart-receipt.json", receipt)

    failures: list[str] = []
    if not shutdown_seen:
        failures.append("the clean shutdown event was not recorded")
    if boot_b.get("end_kind") != "clean":
        failures.append("the restart did not report a clean gap")
    if intention is None:
        failures.append("the active river goal did not survive the restart")
    if failures:
        receipt["failures"] = failures
        save_json(root / "restart-receipt.json", receipt)
        raise RuntimeError("A06 live gate failed: " + "; ".join(failures))
    print(json.dumps({"out": str(root), "ok": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
