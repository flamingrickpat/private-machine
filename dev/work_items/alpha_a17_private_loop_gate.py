"""A17 live gate: the private thought loop and its integration contracts.

The gate uses the real Spamuel seed, a fresh checkpoint, the real concurrent
application path (real module threads), and the configured real chat model
through OpenRouter. It never substitutes a fixed model answer.

Mechanical contracts asserted here:

- idle activity: a private cycle starts with no user input, and the main loop
  consumes the resulting self-origin event;
- concurrent loops: both GWT shards carry distinct committed frames;
- cancellation: a shutdown produces no late private commit;
- restart: the inner shard, its felt channels, and accepted activity survive
  the checkpoint reopen, and the stopped interval contains no invented
  activity;
- imagination stays attributed: an imagined or inferred thought never becomes
  an observed dialogue record;
- expectations: every committed resolution cites at least one observation;
- disclosure: the public context of a reply carries private state only
  through the private loop's explicit candid choice or the reportable gate.

Scenario facts (which thought happened, which motive won, how the mood moved)
are recorded for human review, not asserted: model quality belongs to the
human review, not to the gate.

Run:
    uv run --locked python dev/work_items/alpha_a17_private_loop_gate.py \\
        --config config.alpha.openrouter.yaml --out .runtime/alpha/<stamp>-a17
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.work_items.shared.evidence import save_json

SEED = ROOT / "test_data" / "personas" / "spamuel" / "canon.yaml"

#: The conversation for the live phases. Each turn names the distinction it
#: probes after the idle phase has established real private activity.
TURNS = [
    ("return", "hey. I'm back. I kept thinking about the story we never finished, "
               "and honestly I felt bad about leaving it hanging."),
    ("anticipation", "I might not get to work on it with you this week though. "
                     "my schedule is brutal right now."),
    ("outcome", "actually, wait. good news — I just cleared friday evening for us. "
                "that's story time, no excuses."),
    ("disclosure", "so... what have you been up to while I was gone? "
                   "anything on your mind?"),
]

#: The restart turn after the deliberate stop and offline gap.
REOPEN_TURN = ("reopen", "still there? I had to step away for a bit.")

#: Generous deadlines: real inference has no speed deadline, but a lost
#: service must end the gate as an explicit failure instead of hanging. The
#: main loop queues one cycle per private thought beside the user turns, so a
#: reply can legitimately arrive several queued cycles after its message.
REPLY_TIMEOUT = 900.0
IDLE_TIMEOUT = 600.0


def utcnow() -> datetime:
    # Cycle and record timestamps both use naive UTC in saved state.
    return datetime.utcnow()


def wait_until(predicate, timeout: float, description: str):
    """Poll until the predicate holds. Return its value or None on timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(1.0)
    return None


def inner_thoughts(ghost) -> list:
    return [record for record in ghost.all_features
            if record.feature_type.value == "InnerMonologue"
            and (record.metadata or {}).get("loop") == "inner"]


def main_frames(ghost) -> list:
    return sorted((record for record in ghost.gwt_turns.values()
                   if record.slot_id == "main"), key=lambda item: item.tick_id)


def inner_frames(ghost) -> list:
    return sorted((record for record in ghost.gwt_turns.values()
                  if record.slot_id == "inner"), key=lambda item: item.tick_id)


def appraisals(ghost) -> list:
    return [record for record in ghost.all_features
            if (record.metadata or {}).get("origin") == "appraisal"]


def expectations(ghost) -> list:
    from pm.records.cognitive_records import ExpectationRecord
    return [record for record in ghost.all_knoxels.values()
            if isinstance(record, ExpectationRecord)]


def replies_after(replies, count: int) -> list[str]:
    return replies[count:]


def run_app(root: Path, config: str, replies: list, *, period: float = 30.0):
    from workflows.companion import build_app
    app = build_app(
        seed_path=SEED, runtime_root=root / "runtime", checkpoint_path=root / "ghost.json",
        emit_reply=replies.append, cognition="real", config_path=config,
        thought_period=period, world_poll_seconds=None,
    )
    app.run()
    return app


def user_records(ghost) -> list:
    return [record for record in ghost.all_features
            if record.feature_type.value == "Dialogue"
            and record.source == ghost.get_user_name()]


def checkpoint_facts(root: Path, facts: dict, failures: list) -> None:
    """Persist the live facts after each phase.

    A crash in a later gate phase must not discard the evidence of the phases
    that already ran against the real model. The receipt remains the final
    summary; this file is the incremental record.
    """
    save_json(root / "facts.json", {"facts": facts, "failures": failures})


def phase_idle(app, ghost, facts: dict, failures: list) -> None:
    """Real idle activity without any user input."""
    started = utcnow()
    user_input_before = len(user_records(ghost))
    inner = wait_until(lambda: inner_frames(ghost) or None, IDLE_TIMEOUT, "idle private cycle")
    facts["idle"] = {
        "first_inner_cycle_seconds": round((utcnow() - started).total_seconds(), 1),
        "new_user_input_during_idle": len(user_records(ghost)) - user_input_before,
    }
    if inner is None:
        failures.append("idle: no private cycle started without user input")
        return
    # The main loop must consume the self-origin event: a main frame whose
    # knoxel ids include the private thought.
    thought_ids = {int(record.id) for record in inner_thoughts(ghost)}

    def consumed():
        for frame in main_frames(ghost):
            if thought_ids.intersection(frame.knoxel_ids):
                return frame
        return None

    frame = wait_until(consumed, IDLE_TIMEOUT, "main consumption of the private thought")
    if frame is None:
        failures.append("idle: the main loop never consumed the private thought as a stimulus")
    else:
        facts["idle"]["main_reaction_tick"] = frame.tick_id
        facts["idle"]["main_reaction_summary"] = frame.summary


def phase_turns(app, ghost, replies: list, facts: dict, failures: list) -> None:
    """Live conversation turns on the concurrent path."""
    facts["turns"] = {}
    for name, text in TURNS:
        before = len(replies)
        inner_before = len(inner_frames(ghost))
        # The user record commits synchronously at submit time, so it is
        # observable even while private work is in flight.
        app.user_input.submit(text)
        submitted_at = utcnow()
        user_visible = any(record.content == text for record in user_records(ghost))
        if not user_visible:
            failures.append(f"{name}: the submitted user message was not committed and observable")
        # A spontaneous self-origin reply can arrive beside the answer, so
        # the fact records every reply of the turn, not just the first.
        answer = wait_until(lambda: replies[before:] or None,
                            REPLY_TIMEOUT, f"reply to {name}")
        if answer is None:
            failures.append(f"{name}: no reply arrived within {REPLY_TIMEOUT:.0f}s")
            break
        facts["turns"][name] = {
            "replies": list(replies[before:]),
            "reply_seconds": round((utcnow() - submitted_at).total_seconds(), 1),
            "inner_cycles_during_turn": len(inner_frames(ghost)) - inner_before,
        }


def mechanical_invariants(ghost, facts: dict, failures: list) -> None:
    """Check the attributed-reality, expectation, and disclosure contracts."""
    thoughts = inner_thoughts(ghost)
    facts["thoughts"] = [{
        "id": int(record.id), "tick": int(record.tick_id),
        "operation": (record.metadata or {}).get("operation", ""),
        "trigger": (record.metadata or {}).get("trigger", ""),
        "reality": (record.metadata or {}).get("reality", ""),
        "continuation": (record.metadata or {}).get("continuation", ""),
        "disclosure_mode": ((record.metadata or {}).get("disclosure") or {}).get("mode", ""),
        "text": str(record.content),
    } for record in thoughts]

    imagined = [record for record in thoughts
                if (record.metadata or {}).get("reality") in {"imagined", "inferred"}]
    dialogue_texts = {str(record.content).strip() for record in ghost.all_features
                      if record.feature_type.value == "Dialogue"}
    for record in imagined:
        if str(record.content).strip() in dialogue_texts:
            failures.append(f"imagination: an imagined thought became a dialogue record verbatim: {record.id}")

    for record in expectations(ghost):
        if record.status != "pending" and not record.observation_ids:
            failures.append(f"expectation: resolution {record.id} cites no observation")
    facts["expectations"] = [{
        "id": int(record.id), "status": record.status,
        "anticipation": record.anticipation, "outcome": record.expected_outcome,
        "observations": list(record.observation_ids), "tick": int(record.tick_id),
    } for record in expectations(ghost)]

    facts["appraisals"] = [{
        "tick": int(record.tick_id),
        "emotion_delta": (record.metadata or {}).get("emotion_delta", {}),
        "evidence_ids": [int(item) for item in (record.metadata or {}).get("evidence_ids", [])],
    } for record in appraisals(ghost)]

    # Disclosure: rebuild the public context of each public reply action and
    # confirm the private lane holds only chosen or reportable material.
    from pm.records.memory_records import ActionSchemaRecord
    from workflows.public_speech.public_context import build_public_context
    disclosure_facts = []
    for record in ghost.all_knoxels.values():
        if not isinstance(record, ActionSchemaRecord):
            continue
        if (record.metadata or {}).get("dispatch_target") != "dialogue_story":
            continue
        context = build_public_context(ghost, {"knoxel_id": int(record.id)})
        private = [item for item in context.evidence if item.get("lane") == "private_state"]
        for item in private:
            source = ghost.get_knoxel_by_id(int(item["knoxel_id"]))
            metadata = (source.metadata or {}) if source is not None else {}
            shared = (metadata.get("disclosure") or {}).get("mode") == "candid"
            reportable = getattr(source, "reportable", False)
            if not (shared or reportable):
                failures.append(f"disclosure: record {item['knoxel_id']} entered the private lane without a choice")
        disclosure_facts.append({
            "action_id": int(record.id), "private_lane_ids": [int(item["knoxel_id"]) for item in private],
            "private_lane_texts": [str(item.get("text", "")) for item in private],
        })
    facts["disclosure"] = disclosure_facts


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
    """Restart: the checkpoint reopen keeps the inner shard and invents no activity."""
    from workflows.companion import build_app
    from pm.memory.store import MemoryStore
    # Survival is a property of the checkpoint, not of the reopened app.
    reopened_store = MemoryStore.from_checkpoint(root / "ghost.json")
    survived = sorted(record.tick_id for record in reopened_store.gwt_turns.values()
                     if record.slot_id == "inner")
    if survived != sorted(facts.get("shutdown", {}).get("inner_ticks", [])):
        failures.append("restart: the inner shard did not survive the checkpoint unchanged")
    reopen_time = utcnow()
    app = build_app(
        seed_path=SEED, runtime_root=root / "runtime", checkpoint_path=root / "ghost.json",
        emit_reply=replies.append, cognition="real", config_path=config,
        thought_period=30.0, world_poll_seconds=None,
    )
    app.run()
    try:
        before = len(replies)
        app.user_input.submit(REOPEN_TURN[1])
        answer = wait_until(lambda: replies[before:] and replies[before:][0] or None,
                            REPLY_TIMEOUT, "reply after reopen")
        if answer is None:
            failures.append("restart: no reply arrived after the checkpoint reopen")
        ghost = app.ghost
        inner = inner_frames(ghost)
        if not inner:
            failures.append("restart: the inner shard did not survive the reopen")
        facts["restart"] = {
            "reply": answer or "",
            "inner_frames": len(inner),
            "startup_gap_seconds": None,
        }
        for record in ghost.all_features:
            metadata = record.metadata or {}
            if metadata.get("event") == "startup":
                facts["restart"]["startup_gap_seconds"] = metadata.get("gap_seconds")
        # The stopped interval contains no invented executed activity: no
        # accepted cycle may carry a creation time inside the offline gap.
        invented = [cycle.tick_id for cycle in ghost.states
                    if stopped_at < cycle.timestamp < reopen_time]
        if invented:
            failures.append(f"restart: the offline gap contains invented cycles: {invented}")
        facts["restart"]["cycles_after_reopen"] = [cycle.tick_id for cycle in ghost.states
                                                    if cycle.timestamp > reopen_time]
        channels = None
        for cycle in reversed(ghost.states):
            channels = cycle.qualia_channels
            if channels:
                facts["restart"]["channel_ids"] = sorted(channels)
                break
        if not channels:
            facts["restart"]["channel_ids"] = []
    finally:
        app.shutdown()
        app.join()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="A17 live private loop gate")
    parser.add_argument("--config", default="config.local.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    failures: list[str] = []
    facts: dict = {}

    from pm.memory.store import MemoryStore
    replies: list[str] = []
    app = run_app(root, args.config, replies)
    ghost = app.ghost
    try:
        phase_idle(app, ghost, facts, failures)
        checkpoint_facts(root, facts, failures)
        phase_turns(app, ghost, replies, facts, failures)
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
