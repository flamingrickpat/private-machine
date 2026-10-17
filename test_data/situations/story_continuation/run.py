"""Replay the story_continuation situation with real cognition, for debugging.

Start this file in PyCharm with the project interpreter (``.venv``) and the
repository root as the working directory. Set a breakpoint, for example in
``workflows/main_cognition/stages/triage/steps.py`` at ``run``.

Each start copies ``ghost.json`` into a fresh directory under
``.runtime/situations/story_continuation/``. The situation folder stays unchanged.
The script submits the recorded turn, waits for the public reply without a
deadline, and then stops the application. Prompt logs go to ``llm-logs/`` in
the fresh directory. Compare them with ``recorded/llm-logs/``.

The start state is the same on every start. LLM output is not deterministic,
and the clock is the real clock. Time-based signals such as
``time.idle_hours`` therefore differ from the original run.

Create a file named ``STOP`` in the run directory to end the wait early.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from workflows.companion import DEFAULT_SEED, build_app  # noqa: E402

SITUATION = json.loads((HERE / "situation.json").read_text(encoding="utf-8"))


def main() -> Path:
    """Run one replay and return its run directory."""
    out = ROOT / ".runtime" / "situations" / SITUATION["name"] / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    shutil.copy2(HERE / "ghost.json", out / "ghost.json")
    os.environ["PM_PROMPT_LOG_DIR"] = str(out / "llm-logs")
    print(f"run directory: {out}")
    # The same arguments as dev/work_items/alpha_a19_review.py. A long thought
    # period keeps the timed inner loop out of the replay.
    app = build_app(DEFAULT_SEED, out / "app", cognition="real", config_path=str(ROOT / SITUATION["config"]),
                    checkpoint_path=out / "ghost.json", thought_period=3600, world_poll_seconds=None,
                    record_lifecycle=False, emit_reply=lambda text: print(f"\nSpamuel: {text}\n"))
    try:
        app.run()
        print(f"Jesse: {SITUATION['turn']}")
        correlation = app.user_input.submit(SITUATION["turn"], session=SITUATION["session"])
        # No deadline: a paused breakpoint must not end the replay.
        while not (out / "STOP").exists():
            app.raise_if_error()
            with app.ghost.lock:
                done = any(row.metadata.get("exchange_correlation") == correlation
                           and row.metadata.get("exchange_status") == "resolved"
                           for row in app.ghost.all_features)
            if done:
                break
            time.sleep(0.5)
    finally:
        app.shutdown()
        app.join()
    app.raise_if_error()
    print(f"recorded reply: {SITUATION['recorded_reply']['text']}")
    return out


if __name__ == "__main__":
    main()
