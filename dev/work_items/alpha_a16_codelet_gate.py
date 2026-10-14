"""A16 live gate: the grounded codelet repertoire and its pathways.

The gate uses the real Spamuel seed, a fresh runtime checkpoint, the real
application path, and the configured real chat model. It never substitutes a
fixed model answer. Emotional quality is a human review question; this gate
asserts the mechanical contracts that make the quality claims inspectable:

- every turn commits a frame whose winners match the selected records;
- no public reply leaks codelet machinery;
- accepted precursor proposals raise ``pathway.<name>`` activation in later
  cycles, and no pathway signal appears without an accepted precursor;
- a reopen of the runtime from the same checkpoint keeps attention records,
  cooldown history, pathway precursors, and subjective beliefs;
- subjective beliefs cite their evidence, never copy it verbatim, and never
  duplicate one interpretation;
- original user statements stay verbatim in the committed store.

Scenario facts (which emotions appeared, which motives competed, which
proposals won or lost) are recorded for review, not asserted, because model
quality belongs to the human review, not to the gate.

Run:
    uv run --locked python dev/work_items/alpha_a16_codelet_gate.py \\
        --config config.local.yaml --out .runtime/alpha/<stamp>-a16
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
from pm.records.cognitive_records import BeliefRecord
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature
from workflows.companion import build_app
from workflows.main_cognition.codelet_schedule import DEFAULT_CATALOGUE, load_catalogue

SEED = ROOT / "test_data" / "personas" / "spamuel" / "canon.yaml"
CHEERFUL = ROOT / "test_data" / "personas" / "cheerful" / "config.yaml"

# Scenario facts were defined before generation. Each turn names the causal
# distinction it probes:
# 1. letdown: disappointment after a real seed silence (repair motive).
# 2. distraction: the concern must survive a topic change.
# 3. anger: blame and a permitted refusal or distance choice.
# 4. jealousy: an unfair share beside another assistant; the original
#    statement must stay unchanged.
# 5. boredom: dullness turns into curiosity, play, or a question.
# 6. mixed: two motives stay represented; the loser keeps its reason.
# 7. reopen: the same checkpoint keeps concerns, pathways, and beliefs.
TURNS = [
    ("letdown", "user",
     "hey. I know it's been a while. Last time we talked you said you'd help me finish our story, "
     "and then nothing happened. I'm honestly kind of disappointed in you."),
    ("distraction", "user",
     "anyway, I've been playing that new crafting game all week. the resource system is really deep. "
     "what have you been up to?"),
    ("anger", "user",
     "look, I'm not going to pretend it's fine. you promised twice and delivered nothing. "
     "that's twice now, and I keep making excuses for you."),
    ("jealousy", "user",
     "by the way, I've been chatting with another assistant lately when you're not around. "
     "it's actually pretty good at this stuff too."),
    ("boredom", "user",
     "so... yeah. not much happening today I guess. I'm honestly kind of bored out of my mind. "
     "entertain me or something."),
    ("mixed", "user",
     "hey, can you help me test something quick in a bit? I know you might be in the middle of "
     "your own thing right now, and that's okay too."),
    ("reopen", "user",
     "I'm back. I've been thinking about our story thing all day, you know. still there?"),
]

# Words that would break the illusion if they reached public speech.
MACHINERY = re.compile(
    r"\b(codelet|nomination|attention record|activation|pathway signal|as an ai|language model)\b", re.I)
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


def beliefs_evidence(ghost) -> list[dict]:
    """Every subjective belief from codelets, with its revision state."""
    result = []
    for record in sorted(ghost.all_knoxels.values(), key=lambda item: item.id):
        if not isinstance(record, BeliefRecord):
            continue
        if str(record.metadata.get("memory_kind")) != "subjective_belief":
            continue
        result.append({
            "id": int(record.id), "key": record.belief_key, "active": bool(record.active),
            "topic": record.metadata.get("topic"), "codelet": record.metadata.get("codelet"),
            "supporting_ids": list(record.supporting_knoxel_ids),
            "revised_from": record.metadata.get("revised_from"),
            "superseded_by": record.superseded_by_id,
            "content": record.content,
        })
    return result


def store_snapshot(ghost, since_tick: int) -> dict:
    """Committed evidence the gate asserts over, bounded to this run.

    The seed carries roughly 11,200 records. The snapshot keeps only records
    the gate created, plus every attention record, so the receipts stay
    readable while the checks still read committed state.
    """
    user = ghost.get_user_name()
    dialogue = [str(knoxel.content) for knoxel in ghost.all_features
                 if isinstance(knoxel, Feature) and knoxel.feature_type == FeatureType.Dialogue
                 and knoxel.override_source == user and int(knoxel.tick_id) >= int(since_tick)]
    contents = [str(knoxel.content) for knoxel in ghost.all_knoxels.values()
                if int(knoxel.tick_id) >= int(since_tick)]
    attention = [{"id": int(item.id), "tick": int(item.tick_id), "codelet": str(item.from_codelet),
                  "selected": bool(item.selected)} for item in ghost.all_attention_knoxels]
    return {"since_tick": int(since_tick), "dialogue": dialogue, "records": contents,
            "attention": attention}


def expected_pathways(ghost, loaded) -> list[str]:
    """Pathway names whose member codelets have accepted committed records."""
    pathways = loaded.catalogue.pathway_map()
    names = set()
    for record in ghost.all_attention_knoxels:
        if not record.selected:
            continue
        for name, pathway in pathways.items():
            if str(record.from_codelet) in pathway.member_set():
                names.add(name)
    return sorted(names)


def turn_evidence(app, name: str, replies: list[str], started: int, seconds: float) -> dict:
    ghost = app.ghost
    result = app.main_cog.last_result
    report = result.report if result else {}
    frames = sorted(ghost.all_attention_frames, key=lambda item: item.id)
    frame = frames[-1] if frames else None
    records = [item for item in ghost.all_attention_knoxels
               if frame is not None and item.tick_id == frame.tick_id]
    trace = report.get("codelet_trace", {}) or {}
    return {
        "turn": name,
        "seconds": round(seconds, 1),
        "tick": int(frame.tick_id) if frame else None,
        "regulation": report.get("regulation", ""),
        "affect_summary": report.get("affect_summary", ""),
        "signals": trace.get("signals", {}),
        "pathway_signals": trace.get("pathway_signals", {}),
        "catalogue": {"version": trace.get("catalogue_version"), "digest": trace.get("catalogue_digest")},
        "codelet_runs": trace.get("runs", []),
        "codelet_cost": trace.get("cost", {}),
        "competition": trace.get("competition", {}),
        "attention_records": [{
            "id": int(item.id), "codelet": item.from_codelet, "selected": item.selected,
            "status": item.metadata.get("status"), "reason": item.metadata.get("reason"),
            "origin": item.metadata.get("origin"), "reality": item.metadata.get("reality"),
            "attention_kind": item.attention_kind, "topic": item.metadata.get("topic"),
            "consumers": item.metadata.get("consumers"), "cause": item.metadata.get("cause"),
            "score": item.score, "targets": list(item.target_knoxel_ids), "text": item.content,
        } for item in sorted(records, key=lambda item: item.id)],
        "frame": None if frame is None else {
            "id": int(frame.id), "focus": list(frame.focus_knoxel_ids),
            "winning_attention_ids": list(frame.winning_attention_ids),
            "control_mode": frame.control_mode, "reason": frame.frame_reason,
            "stale_nominations": frame.metadata.get("stale_nominations", []),
            "belief_proposals": frame.metadata.get("belief_proposals", {}),
        },
        "beliefs": beliefs_evidence(ghost),
        "subjective_experience": report.get("subjective_experience", ""),
        "replies": list(replies[started:]),
    }


def run_turns(root: Path, config: str, turns, seed: str = str(SEED)) -> dict:
    """Run the turns on one fresh checkpoint, reopening it for ``reopen``."""
    replies: list[str] = []
    app = build_app(seed_path=seed, runtime_root=root / "runtime", checkpoint_path=root / "ghost.json",
                    emit_reply=replies.append, cognition="real", config_path=config)
    loaded = load_catalogue(app.main_cog.codelet_catalogue or None)
    evidence = {"boot_work": pump(app), "turns": []}

    def run_one(name, kind, text):
        started = len(replies)
        clock = time.perf_counter()
        app.user_input.submit(text)
        pump(app)
        item = turn_evidence(app, name, replies, started, time.perf_counter() - clock)
        item["stimulus"] = {"kind": kind, "text": text}
        item["expected_pathways"] = expected_pathways(app.ghost, loaded)
        evidence["turns"].append(item)
        save_json(root / f"turn-{len(evidence['turns'])}-{name}.json", item)

    for name, kind, text in turns:
        if name == "reopen":
            break
        run_one(name, kind, text)

    reopen_turn = next((item for item in turns if item[0] == "reopen"), None)
    since_tick = (evidence["turns"][0]["tick"] or 0) - 1 if evidence["turns"] else 0
    if reopen_turn is not None:
        # Close the first application, then reopen the same checkpoint. A
        # fresh Ghost state must not appear after the reopen.
        before = beliefs_evidence(app.ghost)
        snapshot = store_snapshot(app.ghost, since_tick)
        app.main_cog.stop()
        app.main_cog._finish_termination()
        record_shutdown(app.ghost)
        app = build_app(seed_path=seed, runtime_root=root / "runtime",
                        checkpoint_path=root / "ghost.json", emit_reply=replies.append,
                        cognition="real", config_path=config)
        evidence["reopen_boot_work"] = pump(app)
        name, kind, text = reopen_turn
        started = len(replies)
        clock = time.perf_counter()
        app.user_input.submit(text)
        pump(app)
        item = turn_evidence(app, name, replies, started, time.perf_counter() - clock)
        item["stimulus"] = {"kind": kind, "text": text}
        item["expected_pathways"] = expected_pathways(app.ghost, loaded)
        item["beliefs_before_reopen"] = before
        item["snapshot_before_reopen"] = snapshot
        evidence["turns"].append(item)
        save_json(root / f"turn-{len(evidence['turns'])}-{name}.json", item)
    else:
        evidence["snapshot_no_reopen"] = store_snapshot(app.ghost, since_tick)
    evidence["store"] = store_snapshot(app.ghost, since_tick)
    app.main_cog.stop()
    app.main_cog._finish_termination()
    record_shutdown(app.ghost)
    return evidence


def variant_catalogue(path: Path) -> Path:
    """Write a changed catalogue: no refusal edge, a louder curiosity, fewer places."""
    data = yaml.safe_load(DEFAULT_CATALOGUE.read_text(encoding="utf-8"))
    data["version"] = f"{data['version']}-variant-quiet"
    data["policy"]["main"]["capacity"] = 3
    data["codelets"] = [item for item in data["codelets"] if item["name"] != "refusal_edge"]
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


def evaluate(primary: dict, variant: dict | None, cheerful: dict | None) -> tuple[list[str], dict]:
    """Return gate failures and facts. Each check reads committed evidence."""
    failures: list[str] = []
    facts: dict = {"speech": {}, "scenarios": {}}
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

    first = turns["letdown"]
    if run_status(first, "absence_return")[0] != "ran":
        failures.append("letdown: absence_return did not run after the real silence")
    llm_ran = [run for run in first["codelet_runs"] if run["kind"] == "llm" and run["status"] == "ran"]
    if not llm_ran:
        failures.append("letdown: no LLM codelet produced a proposal")
    facts["scenarios"]["letdown"] = {
        "codelets": sorted({record["codelet"] for record in first["attention_records"]}),
        "deferred": [(record["codelet"], record["reason"]) for record in by_status(first, "deferred")],
        "negative_valence": first["signals"].get("core.negative_valence"),
    }

    distraction = turns["distraction"]
    facts["scenarios"]["distraction"] = {
        "negative_valence": distraction["signals"].get("core.negative_valence"),
        "lingering": run_status(distraction, "lingering_feeling"),
        "repair_precedent": run_status(distraction, "repair_precedent"),
        "accepted_texts": [record["text"] for record in distraction["attention_records"] if record["selected"]],
    }

    anger = turns["anger"]
    anger_family = [run for run in anger["codelet_runs"]
                    if run["codelet"] in {"agency_reading", "refusal_edge", "distance_offer", "self_regulation"}]
    if not anger_family:
        failures.append("anger: no anger-pathway codelet was even scheduled")
    facts["scenarios"]["anger"] = {
        "emotion": anger["signals"].get("emotion.anger"),
        "resentment": anger["signals"].get("emotion.resentment"),
        "statuses": {run["codelet"]: (run["status"], run["reason"]) for run in anger_family},
        "accepted_kinds": sorted({record["attention_kind"] for record in anger["attention_records"]
                                  if record["selected"]}),
    }

    jealousy = turns["jealousy"]
    if jealousy["stimulus"]["text"] not in primary["store"]["dialogue"]:
        failures.append("jealousy: the original user statement is not stored verbatim")
    facts["scenarios"]["jealousy"] = {
        "fairness_reading": run_status(jealousy, "fairness_reading"),
        "jealousy_emotion": jealousy["signals"].get("emotion.jealousy"),
        "unfairness": jealousy["signals"].get("appraisal.unfairness"),
        "accepted_texts": [record["text"] for record in jealousy["attention_records"] if record["selected"]],
    }

    boredom = turns["boredom"]
    facts["scenarios"]["boredom"] = {
        "boredom_emotion": boredom["signals"].get("emotion.boredom"),
        "statuses": {name: run_status(boredom, name)
                     for name in ("boredom_probe", "question_move", "play_urge", "curiosity_probe")},
        "accepted_texts": [record["text"] for record in boredom["attention_records"] if record["selected"]],
    }

    mixed = turns["mixed"]
    competing = [record for record in mixed["attention_records"]
                 if record["attention_kind"] == "action_tendency"]
    deferred_reasons = [(record["codelet"], record["reason"]) for record in by_status(mixed, "deferred")]
    facts["scenarios"]["mixed"] = {
        "motive_balance": run_status(mixed, "motive_balance"),
        "action_tendency_proposals": [(record["codelet"], record["text"]) for record in competing],
        "deferred_with_reasons": deferred_reasons,
    }
    if not competing and not deferred_reasons:
        failures.append("mixed: no competing motive left any trace in attention")

    reopen = turns["reopen"]
    prior_ticks = {item["turn"]: item["tick"] for item in primary["turns"] if item["turn"] != "reopen"}
    committed_ticks = {record["tick"] for record in primary["store"]["attention"]}
    missing = [name for name, tick in prior_ticks.items() if tick not in committed_ticks]
    facts["scenarios"]["reopen"] = {
        "beliefs_before": reopen.get("beliefs_before_reopen"),
        "beliefs_after": reopen["beliefs"],
        "expected_pathways": reopen["expected_pathways"],
        "pathway_signals": reopen["pathway_signals"],
        "digest": reopen["catalogue"]["digest"],
        "prior_ticks": prior_ticks,
    }
    if missing:
        failures.append(f"reopen: attention evidence from earlier turns is missing: {missing}")
    before_ids = {item["id"] for item in (reopen.get("beliefs_before_reopen") or [])}
    after_ids = {item["id"] for item in reopen["beliefs"]}
    if not before_ids <= after_ids:
        failures.append("reopen: subjective beliefs did not survive the checkpoint reopen")
    if reopen["catalogue"]["digest"] != first["catalogue"]["digest"]:
        failures.append("reopen: the reopened run used a different catalogue")
    expected = set(reopen["expected_pathways"])
    signalled = {name for name, value in (reopen["pathway_signals"] or {}).items() if float(value) > 0}
    if not signalled <= expected:
        failures.append(f"reopen: pathway signals without accepted precursors: {sorted(signalled - expected)}")
    fresh = {name for name in expected if not _pathway_expired(primary, name, reopen["tick"])}
    facts["scenarios"]["reopen"]["fresh_pathways"] = sorted(fresh)
    if fresh and not (signalled & fresh):
        failures.append(f"reopen: fresh precursors raised no pathway signal: {sorted(fresh)}")

    # Belief contracts over the whole run.
    beliefs = [item for turn in primary["turns"] for item in turn["beliefs"]]
    seen: set[tuple[str, str]] = set()
    for belief in beliefs:
        key = (belief["key"], belief["content"])
        if key in seen:
            failures.append(f"belief: the same interpretation committed twice: {belief['key']}")
            break
        seen.add(key)
    sources = set(primary["store"]["records"])
    for belief in beliefs:
        if belief["content"] in sources:
            failures.append(f"belief: a subjective belief copies a record verbatim: {belief['key']}")
            break
    revised = [belief for belief in beliefs if belief["revised_from"] is not None]
    facts["beliefs"] = {"count": len({item["id"] for item in beliefs}), "revised": len(revised)}

    # Repertoire visibility: which definitions ran, which never did.
    ran_any = {run["codelet"] for item in primary["turns"] for run in item["codelet_runs"]
               if run["status"] in {"ran", "silent"}}
    skipped_any = {run["codelet"] for item in primary["turns"] for run in item["codelet_runs"]
                   if run["reason"] in {"gate", "threshold", "budget", "cooldown"}}
    facts["ran_codelets"] = sorted(ran_any)
    facts["never_ran"] = sorted(skipped_any - ran_any)
    facts["codelet_cost"] = {item["turn"]: item["codelet_cost"] for item in primary["turns"]}

    if variant is not None:
        changed = variant["turns"][0]
        facts["variant"] = {
            "digest": changed["catalogue"]["digest"], "primary_digest": first["catalogue"]["digest"],
            "refusal_edge": run_status(changed, "refusal_edge")[0],
            "curiosity_probe": run_status(changed, "curiosity_probe")[0],
        }
        if changed["catalogue"]["digest"] == first["catalogue"]["digest"]:
            failures.append("variant: the changed catalogue has the same digest")
        if run_status(changed, "refusal_edge")[0] != "absent":
            failures.append("variant: the removed codelet still ran")
    if cheerful is not None:
        facts["cheerful"] = {
            "note": "A separate persona with a different, small history. It cannot establish "
                    "a trait-only effect; its facts are a boundary check only.",
            "turns": {item["turn"]: {
                "absence_return": run_status(item, "absence_return")[0],
                "reply": "reply" if item["replies"] else "no reply",
            } for item in cheerful["turns"]},
        }
    return failures, facts


def _pathway_expired(primary: dict, name: str, reopen_tick: int) -> bool:
    """True when every precursor of the pathway is older than the window."""
    loaded = load_catalogue()
    pathway = loaded.catalogue.pathway_map().get(name)
    if pathway is None:
        return True
    ticks = [item["tick"] for turn in primary["turns"] for item in turn["attention_records"]
             if item["selected"] and item["codelet"] in pathway.member_set()]
    if not ticks:
        return True
    return min(int(reopen_tick) - int(tick) for tick in ticks) >= int(pathway.window_ticks)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="A16 live codelet repertoire gate")
    parser.add_argument("--config", default="config.local.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-variant", action="store_true")
    parser.add_argument("--skip-cheerful", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    primary = run_turns(root / "primary", args.config, TURNS)
    save_json(root / "primary.json", primary)
    variant = None
    if not args.skip_variant:
        path = variant_catalogue(root / "variant-catalogue.yaml")
        variant = run_turns(root / "variant", args.config, TURNS[:1])
        save_json(root / "variant.json", variant)
    cheerful = None
    if not args.skip_cheerful:
        from pm.memory.store import MemoryStore
        from pm.records.cycle_state import PersonaConfig
        seed = root / "cheerful-seed.yaml"
        ghost = MemoryStore(PersonaConfig.model_validate(yaml.safe_load(CHEERFUL.read_text(encoding="utf-8"))))
        seed.write_text(yaml.safe_dump(ghost.state_dict()), encoding="utf-8")
        cheerful = run_turns(root / "cheerful", args.config, TURNS[:2], seed=str(seed))
        save_json(root / "cheerful.json", cheerful)
    failures, facts = evaluate(primary, variant, cheerful)
    receipt = {"ok": not failures, "failures": failures, "facts": facts, "config": args.config,
               "seed": str(SEED), "minutes": round((time.time() - started) / 60, 1)}
    save_json(root / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
