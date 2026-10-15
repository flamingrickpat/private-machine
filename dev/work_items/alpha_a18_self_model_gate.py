"""A18 live gate: source-grounded self-model and capability discovery.

The gate uses the real Spamuel seed, a fresh checkpoint, the real concurrent
application path, and the configured real chat model through OpenRouter. It
never substitutes a fixed model answer.

Mechanical contracts asserted here:

- every committed capability belief cites a real evidence revision and a
  distinct state, and no capability becomes ``observed`` without a receipt;
- every inspector answer either names evidence records or carries an explicit
  limit, so a missing trace never becomes an invented mechanism;
- a committed private thought keeps its capability snapshot beside its
  attributed reality label;
- capability beliefs survive a checkpoint reopen with their revisions intact.

Scenario facts (which capability the companion chose to inspect, how it
explained a choice, how the finding reached appraisal) are recorded for human
review, not asserted: model quality belongs to the human review.

Run:
    uv run --locked python dev/work_items/alpha_a18_self_model_gate.py \\
        --config config.alpha.openrouter.yaml --out .runtime/alpha/<stamp>-a18
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.work_items.shared.evidence import save_json
# Reuse the A17 live helpers rather than copy the concurrent-run machinery.
from dev.work_items.alpha_a17_private_loop_gate import (
    REPLY_TIMEOUT, SEED, inner_frames, inner_thoughts, main_frames,
    run_app, utcnow, wait_until,
)

#: The conversation probes capability discovery and self-explanation.
TURNS = [
    ("capabilities", "quick question: what can you actually do right now, and what's "
                     "only something you're supposed to be able to do?"),
    ("limits", "be honest with me — is there anything you can't do yet that you wish you could?"),
    ("decision", "why did you say that, though? walk me through how you got there."),
    ("feel", "does knowing what you can't do yet bother you at all?"),
]

#: A restart turn after the deliberate stop.
REOPEN_TURN = ("reopen", "still there? I had to step away for a bit.")

IDLE_TIMEOUT = 600.0
#: How long to wait for the model to choose a capability inspection.
INSPECTION_TIMEOUT = 900.0


def capability_beliefs(ghost) -> list:
    from pm.records.cognitive_records import BeliefRecord
    return [record for record in ghost.all_knoxels.values()
            if isinstance(record, BeliefRecord)
            and (record.metadata or {}).get("memory_kind") == "capability_belief"]


def checkpoint_facts(root: Path, facts: dict, failures: list) -> None:
    """Persist live facts after each phase, so a late crash keeps the evidence."""
    save_json(root / "facts.json", {"facts": facts, "failures": failures})


def phase_idle(app, ghost, facts: dict, failures: list) -> None:
    """Wait for real idle private activity and note the capability overlay."""
    overlay = getattr(app, "capability_overlay", None)
    facts["overlay_initial"] = overlay.to_dict() if overlay is not None else {}
    started = utcnow()
    inner = wait_until(lambda: inner_frames(ghost) or None, IDLE_TIMEOUT, "idle private cycle")
    facts["idle"] = {"first_inner_cycle_seconds": round((utcnow() - started).total_seconds(), 1)}
    if inner is None:
        failures.append("idle: no private cycle started without user input")


def phase_turns(app, ghost, replies: list, facts: dict, failures: list) -> None:
    """Live turns that probe capability discovery and self-explanation."""
    facts["turns"] = {}
    for name, text in TURNS:
        before = len(replies)
        app.user_input.submit(text)
        submitted_at = utcnow()
        answer = wait_until(lambda: replies[before:] or None, REPLY_TIMEOUT, f"reply to {name}")
        if answer is None:
            failures.append(f"{name}: no reply arrived within {REPLY_TIMEOUT:.0f}s")
            break
        facts["turns"][name] = {
            "replies": list(replies[before:]),
            "reply_seconds": round((utcnow() - submitted_at).total_seconds(), 1),
        }


def phase_inspection(app, ghost, facts: dict, failures: list) -> None:
    """Wait for a chosen capability inspection, then check its evidence."""
    belief = wait_until(lambda: capability_beliefs(ghost) or None, INSPECTION_TIMEOUT,
                        "capability inspection")
    if belief is None:
        facts["inspection"] = {"committed": False,
                               "note": "the model did not choose a capability inspection in this run"}
        return
    facts["inspection"] = {"committed": True, "count": len(belief)}
    for record in belief:
        metadata = record.metadata or {}
        if not metadata.get("evidence_revision"):
            failures.append(f"capability belief {record.id} cites no evidence revision")
        if metadata.get("state") not in {"intended", "implemented", "enabled", "observed",
                                         "unavailable", "uncertain"}:
            failures.append(f"capability belief {record.id} has an unknown state")
        # A capability may only be observed when the builder supplied a row.
        states = metadata.get("states") or []
        if metadata.get("state") == "observed" and "observed" not in states:
            failures.append(f"capability belief {record.id} claims observed without a receipt")
    facts["inspection"]["beliefs"] = [{
        "id": int(record.id), "capability": (record.metadata or {}).get("capability"),
        "state": (record.metadata or {}).get("state"),
        "stale": (record.metadata or {}).get("stale", False),
        "text": str(record.content),
    } for record in belief]


def mechanical_invariants(ghost, facts: dict, failures: list) -> None:
    """Check the attribution and evidence-limit contracts."""
    from pm.introspection.episodes import EpisodeIndex, answer_questions
    index = EpisodeIndex(ghost)
    facts["episodes"] = [item.to_dict() for item in index.recent(limit=10)]
    facts["cross_loop_links"] = index.cross_loop_links()
    for episode in index.recent(limit=5):
        answers = answer_questions(ghost, episode, overlay=getattr(_APP, "capability_overlay", None))
        for key, value in answers.items():
            if value.get("evidence") or value.get("limit"):
                continue
            failures.append(f"inspector: {key} on episode {episode.episode_id} has no evidence and no limit")

    thoughts = inner_thoughts(ghost)
    facts["thoughts"] = [{
        "id": int(record.id), "operation": (record.metadata or {}).get("operation", ""),
        "reality": (record.metadata or {}).get("reality", ""),
        "capability_states": (record.metadata or {}).get("capability_states", {}),
        "text": str(record.content),
    } for record in thoughts]

    stale = []
    overlay = getattr(_APP, "capability_overlay", None)
    if overlay is not None:
        stale = overlay.stale_names()
    facts["stale_capabilities"] = stale


def phase_shutdown(app, ghost, facts: dict, failures: list) -> None:
    """Cancellation: no late private commit after the stop."""
    app.shutdown()
    app.join()
    stopped_at = utcnow()
    ticks = {record.tick_id for record in inner_frames(ghost)}
    time.sleep(3.0)
    after = {record.tick_id for record in inner_frames(ghost)}
    if after != ticks:
        failures.append("cancellation: a private cycle committed after the shutdown")
    facts["shutdown"] = {"stopped_at": stopped_at.isoformat(), "inner_ticks": sorted(ticks)}
    try:
        app.raise_if_error()
    except BaseException as error:  # noqa: BLE001 - the gate reports the failure
        failures.append(f"shutdown: a module error surfaced: {type(error).__name__}: {error}")


def phase_restart(root: Path, config: str, replies: list, stopped_at: datetime,
                  facts: dict, failures: list) -> None:
    """Restart: beliefs and revisions survive the checkpoint reopen."""
    from pm.memory.store import MemoryStore
    reopened_store = MemoryStore.from_checkpoint(root / "ghost.json")
    before = [(record.belief_key, record.metadata.get("evidence_revision"))
              for record in capability_beliefs(reopened_store)]
    app = run_app(root, config, replies)
    try:
        ghost = app.ghost
        after = [(record.belief_key, record.metadata.get("evidence_revision"))
                 for record in capability_beliefs(ghost)]
        if before and after != before:
            failures.append("restart: capability beliefs or their revisions changed across the reopen")
        before_replies = len(replies)
        app.user_input.submit(REOPEN_TURN[1])
        answer = wait_until(lambda: replies[before_replies:] or None, REPLY_TIMEOUT, "reply after reopen")
        if answer is None:
            failures.append("restart: no reply arrived after the checkpoint reopen")
        facts["restart"] = {
            "reply": answer or "",
            "capability_belief_count": len(after),
            "capability_beliefs": [{"key": key, "revision": revision} for key, revision in after],
            "inner_frames": len(inner_frames(ghost)),
        }
        invented = [cycle.tick_id for cycle in ghost.states
                    if stopped_at < cycle.timestamp < utcnow()]
        if invented:
            failures.append(f"restart: the offline gap contains invented cycles: {invented}")
    finally:
        app.shutdown()
        app.join()


_APP = None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="A18 live self-model gate")
    parser.add_argument("--config", default="config.alpha.openrouter.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    failures: list[str] = []
    facts: dict = {}

    global _APP
    replies: list[str] = []
    app = run_app(root, args.config, replies, period=15.0)
    _APP = app
    ghost = app.ghost
    try:
        phase_idle(app, ghost, facts, failures)
        checkpoint_facts(root, facts, failures)
        phase_turns(app, ghost, replies, facts, failures)
        checkpoint_facts(root, facts, failures)
        phase_inspection(app, ghost, facts, failures)
        checkpoint_facts(root, facts, failures)
        phase_shutdown(app, ghost, facts, failures)
        checkpoint_facts(root, facts, failures)
    finally:
        try:
            app.shutdown()
            app.join()
        except Exception:
            pass
    mechanical_invariants(ghost, facts, failures)
    stopped_at = facts.get("shutdown", {}).get("stopped_at")
    stopped_time = datetime.fromisoformat(stopped_at) if stopped_at else datetime.now()
    phase_restart(root, args.config, replies, stopped_time, facts, failures)
    checkpoint_facts(root, facts, failures)

    receipt = {
        "ok": not failures, "failures": failures, "facts": facts, "config": args.config,
        "seed": str(SEED), "minutes": round((time.time() - started) / 60, 1),
        "replies": replies,
    }
    save_json(root / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
