"""Derive the story_continuation situation from the A19 review run.

The source run is ``.runtime/alpha/20261017-a19-review-01``. Its final checkpoint
holds the state after tick 816. This script removes every commit from the
submission of Jesse's third message onward. The result is the state that
existed just before ``user_input.submit`` for that message.

Run it once. ``run.py`` uses the written ``ghost.json`` and does not call this script.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / ".runtime" / "alpha" / "20261017-a19-review-01"

# The first knoxel created by the replayed turn. Commit ecfda87f added it at
# submission. Every later knoxel belongs to tick 816.
FIRST_REMOVED_ID = 11347
LAST_KEPT_TICK = 815
REMOVED_COMMITS = {
    "ecfda87f173a4cde8028e43bb0bb88b9",  # user input 11347
    "afef6e6cd9d54bfaa62ff57826130d83",  # main cognition tick 816
    "1db3a4e3b4764c86a40b025a8f3bb364",  # public reply 11370
    "public-ack:11370",
}
# The prompt logs of tick 816, from the submission to the published reply.
LOG_WINDOW = ("2026-10-17T17:18:00", "2026-10-17T17:21:00")


def rollback(ghost: dict) -> dict:
    """Remove tick 816 and the submitted message from a checkpoint dict."""
    commits = ghost["applied_commits"]
    missing = REMOVED_COMMITS - set(commits)
    if missing:
        raise ValueError(f"Source checkpoint lacks expected commits: {sorted(missing)}")
    # Commits that are kept must not touch removed knoxels. Otherwise the
    # rollback would leave a partial update behind.
    for key, commit in commits.items():
        if key in REMOVED_COMMITS:
            continue
        touched = set(commit.get("added_ids", [])) | set(commit.get("updated_ids", []))
        if any(item >= FIRST_REMOVED_ID for item in touched):
            raise ValueError(f"Kept commit {key} touches a removed knoxel.")
    ghost["applied_commits"] = {k: v for k, v in commits.items() if k not in REMOVED_COMMITS}
    state = ghost["state"]
    state["knoxels"] = [k for k in state["knoxels"] if k["data"]["id"] < FIRST_REMOVED_ID]
    state["cycles"] = [c for c in state["cycles"] if c["tick_id"] <= LAST_KEPT_TICK]
    ghost["revisions"] = {k: v for k, v in ghost["revisions"].items() if int(k) < FIRST_REMOVED_ID}
    ghost["gwt_turns"] = {k: v for k, v in ghost["gwt_turns"].items() if int(k) <= LAST_KEPT_TICK}
    ghost["current_knoxel_id"] = FIRST_REMOVED_ID - 1
    ghost["current_tick_id"] = LAST_KEPT_TICK
    ghost["reserved_tick_id"] = LAST_KEPT_TICK
    return ghost


def copy_recorded_logs() -> list[dict]:
    """Copy the original prompt logs of tick 816 and return a time-ordered index."""
    target = HERE / "recorded" / "llm-logs"
    target.mkdir(parents=True, exist_ok=True)
    index = []
    for path in sorted((SOURCE / "llm-logs").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        head = dict(line.strip("- ").split(": ", 1) for line in text.splitlines()[:16] if ": " in line)
        if not LOG_WINDOW[0] <= head.get("time", "") <= LOG_WINDOW[1]:
            continue
        shutil.copy2(path, target / path.name)
        body = text.split("### 1. system", 1)[-1].strip().splitlines()
        first = body[0].strip() if body else ""
        index.append({"time": head["time"], "file": f"recorded/llm-logs/{path.name}",
                      "route": head.get("route", ""), "thread": head.get("owner_thread", ""),
                      "first_line": first[:120]})
    return sorted(index, key=lambda row: row["time"])


def main() -> None:
    ghost = json.loads((SOURCE / "ghost.json").read_text(encoding="utf-8"))
    (HERE / "ghost.json").write_text(json.dumps(rollback(ghost), ensure_ascii=False), encoding="utf-8")
    receipt = json.loads((SOURCE / "receipt.json").read_text(encoding="utf-8"))
    situation = {
        "name": "story_continuation",
        "source_run": str(SOURCE.relative_to(ROOT)),
        "source_checkpoint_after_tick": 816,
        "start_state": "after tick 815 and public-ack:11346; before submission of the turn below",
        "session": SOURCE.name,
        "turn": "the unfinished one feels more like us. i still don't want to lock it in tonight though.",
        "config": "config.alpha.openrouter.yaml",
        "prior_chat": [m for m in receipt["messages"] if m["at"] < "2026-10-17T17:18:00.4"],
        "recorded_reply": next(m for m in receipt["messages"] if m["id"] == 11370),
        "recorded_logs": copy_recorded_logs(),
    }
    (HERE / "situation.json").write_text(json.dumps(situation, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
