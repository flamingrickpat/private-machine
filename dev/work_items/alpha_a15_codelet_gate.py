"""A15 live gate: selective codelets and inspectable attention competition.

The gate uses the real Spamuel seed, a fresh runtime checkpoint, the real
application path, and the configured real chat model. It never substitutes a
fixed model answer. It drives five turns on one checkpoint:

1. ``disappointment``: Jesse returns after a long real silence, disappointed.
   Competing codelets nominate; some win and some wait.
2. ``follow_up``: an immediate second message. Cooldowns hold back codelets
   that ran in turn 1, and the absence codelet stays closed.
3. ``thought``: a private thought arrives through the same ``thought.created``
   path that ``InnerThoughtLoop`` uses. The gate authors the thought text,
   because the A17 inner loop does not exist yet. Its proposals must never
   claim an outside origin.
4. ``stale``: a concurrent writer changes an active goal between proposal and
   commit. The goal nomination must commit as ``stale`` and lose, while the
   cycle still publishes a reply.
5. ``danger``: an urgent event. When the appraisal reaches danger (an
   appraised bodily threat), no LLM codelet may run. When it does not, the
   gate records that the danger path was not reached instead of claiming it.

A second fresh checkpoint runs turn 1 with a changed catalogue, so the two
attention frames can be compared.

Run:
    uv run --locked python dev/work_items/alpha_a15_codelet_gate.py \\
        --config config.local.yaml --out .runtime/alpha/<stamp>-a15
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml

from dev.work_items.shared.evidence import save_json
from pm.app.lifecycle_records import record_shutdown
from pm.bus.envelopes import new_envelope
from pm.records.cognitive_records import Intention
from pm.records.enums import FeatureType, IntentionType, SourceKind
from pm.records.memory_records import new_feature
from workflows.companion import build_app
from workflows.main_cognition.codelet_schedule import DEFAULT_CATALOGUE, in_danger
from workflows.main_cognition.stage_commits import active_intentions

SEED = ROOT / "test_data" / "personas" / "spamuel" / "canon.yaml"

TURNS = [
    ("disappointment", "user",
     "hey. I know it's been a while. Last time we talked you said you'd help me finish our story, "
     "and then nothing. I'm honestly kind of disappointed."),
    ("follow_up", "user",
     "I'm not angry. I just thought that story mattered more to you."),
    ("thought", "thought",
     "I keep hearing how flat Jesse sounded. Maybe I really did let him down, "
     "and maybe he is starting to give up on me."),
    ("stale", "user",
     "okay. can we still finish it together this week?"),
    ("danger", "user",
     "wait, I smell smoke from the kitchen and the alarm just went off!"),
]

# Words that would break the illusion if they reached public speech.
MACHINERY = re.compile(r"\b(codelet|nomination|attention record|activation|as an ai|language model)\b", re.I)
# Recovery texts from public_speech. A cycle that ends in one of them did not
# produce a companion reply, even though something was published.
RECOVERY = re.compile(r"Please ask again\.|I could not complete the decision", re.I)


def pump(app, rounds: int = 80) -> int:
    """Run each module step until no module has work. No threads are started."""
    work = 0
    for _ in range(rounds):
        progressed = False
        for module in app.modules:
            if module.step(block=False):
                progressed = True
                work += 1
        if not progressed:
            break
    return work


def submit_thought(app, text: str) -> int:
    """Commit a private thought and announce it exactly as InnerThoughtLoop does."""
    ghost = app.ghost
    changes = ghost.begin_batch(f"a15-thought-{time.time_ns()}")
    new_feature(ghost, text, FeatureType.InnerMonologue, source=ghost.get_companion_name(),
                changes=changes, source_kind=SourceKind.Self)
    receipt = ghost.commit(changes)
    knoxel_id = receipt.added_ids[0]
    app.bus.put(new_envelope("inner_thought", "data", "thought.created", "", {"knoxel_id": knoxel_id}))
    return knoxel_id


def ensure_goal(app) -> dict:
    """Return the first active goal, creating a labelled fixture goal if none exists.

    The real goals stage may already have added one. A fixture goal records
    its origin, so it cannot pass as a goal the companion chose.
    """
    goals = active_intentions(app.ghost)
    if goals:
        return {"id": int(goals[0].id), "content": goals[0].content, "fixture": False}
    changes = app.ghost.begin_batch(f"a15-goal-fixture-{time.time_ns()}")
    changes.create(Intention(
        content="Finish the shared story with Jesse.", entity_id=0, affective_valence=0.0,
        incentive_salience=0.7, intention_type=IntentionType.WorldState, status="active", active=True,
        metadata={"goal_key": "finish the shared story with jesse", "origin": "a15_gate_fixture"},
    ))
    receipt = app.ghost.commit(changes)
    return {"id": int(receipt.added_ids[0]), "content": "Finish the shared story with Jesse.", "fixture": True}


def concurrent_goal_writer(app, goal_id: int, log: dict):
    """Wrap publish_output so a real second writer changes the goal first.

    This is the race that stale rejection protects against: the cycle read
    the goal at one revision, and another writer committed a newer revision
    before the cycle's commit.
    """
    original = app.main_cog.publish_output

    def publish(report, batch):
        ghost = app.ghost
        goal = ghost.get_knoxel_by_id(goal_id)
        before = int(ghost.revisions.get(goal_id, 0))
        changes = ghost.begin_batch(f"a15-concurrent-{time.time_ns()}")
        goal.incentive_salience = min(1.0, float(goal.incentive_salience) + 0.05)
        changes.replace(goal)
        ghost.commit(changes)
        log.update({"goal_id": goal_id, "revision_before": before,
                    "revision_after": int(ghost.revisions.get(goal_id, 0))})
        app.main_cog.publish_output = original
        return original(report, batch)

    app.main_cog.publish_output = publish


def turn_evidence(app, name: str, replies: list[str], started: int, seconds: float) -> dict:
    ghost = app.ghost
    result = app.main_cog.last_result
    report = result.report if result else {}
    frames = sorted(ghost.all_attention_frames, key=lambda item: item.id)
    frame = frames[-1] if frames else None
    records = [item for item in ghost.all_attention_knoxels if frame is not None and item.tick_id == frame.tick_id]
    trace = report.get("codelet_trace", {}) or {}
    return {
        "turn": name,
        "seconds": round(seconds, 1),
        "tick": int(frame.tick_id) if frame else None,
        "regulation": report.get("regulation", ""),
        "affect_summary": report.get("affect_summary", ""),
        "signals": trace.get("signals", {}),
        "catalogue": {"version": trace.get("catalogue_version"), "digest": trace.get("catalogue_digest")},
        "codelet_runs": trace.get("runs", []),
        "codelet_cost": trace.get("cost", {}),
        "competition": trace.get("competition", {}),
        "attention_records": [{
            "id": int(item.id), "codelet": item.from_codelet, "selected": item.selected,
            "status": item.metadata.get("status"), "reason": item.metadata.get("reason"),
            "origin": item.metadata.get("origin"), "reality": item.metadata.get("reality"),
            "score": item.score, "targets": list(item.target_knoxel_ids), "text": item.content,
        } for item in sorted(records, key=lambda item: item.id)],
        "frame": None if frame is None else {
            "id": int(frame.id), "focus": list(frame.focus_knoxel_ids),
            "winning_attention_ids": list(frame.winning_attention_ids),
            "control_mode": frame.control_mode, "reason": frame.frame_reason,
            "stale_nominations": frame.metadata.get("stale_nominations", []),
        },
        "subjective_experience": report.get("subjective_experience", ""),
        "replies": list(replies[started:]),
    }


def run_sequence(root: Path, config: str, turns, *, catalogue: str | None = None) -> dict:
    replies: list[str] = []
    app = build_app(seed_path=str(SEED), runtime_root=root / "runtime", checkpoint_path=root / "ghost.json",
                    emit_reply=replies.append, cognition="real", config_path=config)
    if catalogue:
        app.main_cog.codelet_catalogue = catalogue
    boot_work = pump(app)
    evidence = {"boot_work": boot_work, "turns": [], "fixtures": {}}
    for name, kind, text in turns:
        started = len(replies)
        if name == "stale":
            goal = ensure_goal(app)
            evidence["fixtures"]["stale_goal"] = goal
            writer: dict = {}
            evidence["fixtures"]["concurrent_writer"] = writer
            concurrent_goal_writer(app, goal["id"], writer)
        clock = time.perf_counter()
        if kind == "thought":
            evidence["fixtures"]["thought_id"] = submit_thought(app, text)
        else:
            # The terminal and the console submit the plain message text.
            app.user_input.submit(text)
        pump(app)
        item = turn_evidence(app, name, replies, started, time.perf_counter() - clock)
        item["stimulus"] = {"kind": kind, "text": text}
        evidence["turns"].append(item)
        save_json(root / f"turn-{len(evidence['turns'])}-{name}.json", item)
    app.main_cog.stop()
    app.main_cog._finish_termination()
    record_shutdown(app.ghost)
    return evidence


def variant_catalogue(path: Path) -> Path:
    """Write a changed catalogue: no memory echo, a louder curiosity, two places."""
    data = yaml.safe_load(DEFAULT_CATALOGUE.read_text(encoding="utf-8"))
    data["version"] = f"{data['version']}-variant-curious"
    data["policy"]["main"]["capacity"] = 2
    data["codelets"] = [item for item in data["codelets"] if item["name"] != "disappointment_echo"]
    for item in data["codelets"]:
        if item["name"] == "curiosity_probe":
            item["base_activation"] = 0.9
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def by_status(turn: dict, status: str) -> list[dict]:
    return [item for item in turn["attention_records"] if item["status"] == status]


def run_status(turn: dict, codelet: str) -> tuple[str, str]:
    for item in turn["codelet_runs"]:
        if item["codelet"] == codelet:
            return item["status"], item["reason"]
    return "absent", ""


def evaluate(primary: dict, variant: dict | None) -> tuple[list[str], dict]:
    """Return gate failures and facts. Each check reads committed evidence.

    Public speech quality belongs to A09. A missing reply or a recovery
    fallback is recorded in ``facts["speech"]`` as an immersion finding, not
    as an A15 failure. A leak of codelet machinery into speech is an A15
    failure, because codelet text reaches the workspace.
    """
    failures: list[str] = []
    facts: dict = {"speech": {}}
    turns = {item["turn"]: item for item in primary["turns"]}
    for item in primary["turns"]:
        fallback = [reply for reply in item["replies"] if RECOVERY.search(reply)]
        facts["speech"][item["turn"]] = ("no reply" if not item["replies"]
                                          else "recovery fallback" if fallback else "companion reply")
        leaked = [reply for reply in item["replies"] if MACHINERY.search(reply)]
        if leaked:
            failures.append(f"{item['turn']}: a public reply names internal machinery: {leaked[0][:120]!r}")
        frame = item["frame"] or {}
        winners = sorted(record["id"] for record in item["attention_records"] if record["selected"])
        if sorted(frame.get("winning_attention_ids", [])) != winners:
            failures.append(f"{item['turn']}: frame winners differ from selected AttentionRecords")
    first = turns["disappointment"]
    codelets = {record["codelet"] for record in first["attention_records"]}
    llm_proposals = [run for run in first["codelet_runs"] if run["kind"] == "llm" and run["status"] == "ran"]
    facts["t1_codelets"] = sorted(codelets)
    facts["t1_deferred"] = [(record["codelet"], record["reason"]) for record in by_status(first, "deferred")]
    if len(codelets) < 3:
        failures.append("disappointment: fewer than three codelets nominated")
    if not llm_proposals:
        failures.append("disappointment: no LLM codelet produced a proposal")
    if not by_status(first, "deferred"):
        failures.append("disappointment: no nomination was deferred, so no competition is visible")
    if run_status(first, "absence_return")[0] != "ran":
        failures.append("disappointment: absence_return did not run after the real silence")
    second = turns["follow_up"]
    facts["t2_absence"] = run_status(second, "absence_return")
    if run_status(second, "absence_return")[1] != "gate":
        failures.append("follow_up: absence_return was not held by its idle-time gate")
    cooled = [run["codelet"] for run in second["codelet_runs"] if run["reason"] == "cooldown"]
    facts["t2_cooldown"] = cooled
    ran_first = {run["codelet"] for run in first["codelet_runs"] if run["status"] in {"ran", "silent"}}
    if not set(cooled) & ran_first:
        failures.append("follow_up: no codelet that ran in turn 1 waited out a cooldown")
    thought = turns["thought"]
    origins = {record["origin"] for record in thought["attention_records"]}
    facts["t3_origins"] = sorted(origins)
    if "outside" in origins:
        failures.append("thought: a thought-started cycle committed an outside-origin nomination")
    if run_status(thought, "thought_echo")[0] not in {"ran", "silent"}:
        failures.append("thought: thought_echo did not run for a private thought")
    stale = turns["stale"]
    stale_records = by_status(stale, "stale")
    facts["t4_stale"] = [(record["codelet"], record["targets"]) for record in stale_records]
    writer = primary["fixtures"].get("concurrent_writer", {})
    if writer.get("revision_after", 0) <= writer.get("revision_before", 0):
        failures.append("stale: the concurrent writer did not change the goal revision")
    if not stale_records or any(record["selected"] for record in stale_records):
        failures.append("stale: no stale nomination was recorded, or a stale one won")
    danger = turns["danger"]
    calls = danger["codelet_cost"].get("llm_calls", 0)
    reached = in_danger(danger["signals"])
    facts["t5_regulation"] = danger["regulation"]
    facts["t5_llm_calls"] = calls
    facts["t5_danger_reached"] = reached
    if reached and calls:
        failures.append("danger: the danger path still ran LLM codelets")
    painful = [item["turn"] for item in primary["turns"]
               if item["regulation"] == "reactive" and not in_danger(item["signals"])]
    facts["reactive_without_danger"] = painful
    if variant is not None:
        changed = variant["turns"][0]
        primary_winners = sorted(record["codelet"] for record in first["attention_records"] if record["selected"])
        variant_winners = sorted(record["codelet"] for record in changed["attention_records"] if record["selected"])
        facts["variant"] = {"primary_winners": primary_winners, "variant_winners": variant_winners,
                            "primary_digest": first["catalogue"]["digest"],
                            "variant_digest": changed["catalogue"]["digest"]}
        if first["catalogue"]["digest"] == changed["catalogue"]["digest"]:
            failures.append("variant: the changed catalogue has the same digest")
        if primary_winners == variant_winners:
            failures.append("variant: the changed catalogue selected the same attention frame")
    return failures, facts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="A15 live codelet gate")
    parser.add_argument("--config", default="config.local.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-variant", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    primary = run_sequence(root / "primary", args.config, TURNS)
    save_json(root / "primary.json", primary)
    variant = None
    if not args.skip_variant:
        path = variant_catalogue(root / "variant-catalogue.yaml")
        variant = run_sequence(root / "variant", args.config, TURNS[:1], catalogue=str(path))
        save_json(root / "variant.json", variant)
    failures, facts = evaluate(primary, variant)
    receipt = {"ok": not failures, "failures": failures, "facts": facts, "config": args.config,
               "seed": str(SEED), "minutes": round((time.time() - started) / 60, 1)}
    save_json(root / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
