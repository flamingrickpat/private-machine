"""A19F review session: a short real conversation and its record package.

The script copies the story_continuation situation into a fresh run
directory under ``.runtime/alpha/<time>-a19f-review/``. It starts the full
companion with real cognition, the Spamuel canon, and the configured real
model. Jesse sends five messages:

1. the recorded story_continuation turn: a fourth deferral of the ending
   choice (a repetition);
2. a tease;
3. an emotionally loaded admission;
4. one more deferral;
5. a goodnight.

Turn 1 is the recorded text. A separate Jesse model writes turns 2 to 5
from the public chat only, each with a short direction. So Jesse reacts to
what Spamuel actually said. Jesse never sees private records.

After the last reply, the script freezes one snapshot and writes the
record packages (``pm.introspection.record_packages``):

- ``chat.json``: the public chat from Jesse's perspective, in delivery
  order, with the last prior lines of the situation;
- ``cycle-<tick>.json``: one cycle package for each main cognition tick of
  the session, with the mental state before the tick.

All packages come from the same snapshot, so one combined review item can
hold them. Register them with::

    uv run --locked python dev/review/server.py register-package --combine \\
        .runtime/alpha/<run>/packages/*.json --id <id> --work-item A19F

A runtime error or a missing reply keeps the receipt with ``ok: false``.
Such a run is development evidence, not a review package.

Usage::

    uv run --locked python dev/work_items/alpha_a19f_review.py
    uv run --locked python dev/work_items/alpha_a19f_review.py --package .runtime/alpha/<run>
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.work_items.alpha_a18c_mental_gate import save, snapshot  # noqa: E402
from pm.context_render.turns import Role, Turn  # noqa: E402
from pm.introspection.record_packages import capture_cycle, capture_records, dump_package  # noqa: E402
from pm.llm.activation import ModelActivation  # noqa: E402
from pm.llm.config import load_config, resolve_connection  # noqa: E402
from pm.llm.stage_model import LazyStageModel  # noqa: E402
from pm.memory.store import MemoryStore  # noqa: E402
from pm.records.enums import FeatureType  # noqa: E402
from workflows.companion import DEFAULT_SEED, build_app  # noqa: E402

SITUATION = ROOT / "test_data" / "situations" / "story_continuation"
#: Seconds that one exchange may take before the run fails. Upstream rate limits (HTTP 429) retry
#: without a quota, so one exchange can take long; a session in step 11 lost its fourth reply to 1800 s.
EXCHANGE_DEADLINE = 5400
#: Prior chat lines of the situation that the chat package shows before the session.
PRIOR_LINES = 4
#: The first turn is the recorded story_continuation turn. The later turns are directions for the Jesse model.
FIRST_TURN = "the unfinished one feels more like us. i still don't want to lock it in tonight though."
DIRECTIONS = (
    "Tease Spamuel about how dramatic and theatrical he is being about this story. Playful, a little mean, short.",
    "Admit something real and a bit awkward: the quiet ending scares you a little, because the last time things "
    "got quiet with someone close to you, they left. Keep it short. Do not ask for comfort.",
    "Back off from what you just admitted. Ask to leave the story open for one more day, without choosing an "
    "ending yet.",
    "Say goodnight in your own way. You are tired.",
)
JESSE_SYSTEM = (
    "Pretend to be human Jesse texting your longtime companion Spamuel late in the evening. You two are writing a "
    "story together and keep talking about its two possible endings. Write exactly one natural message, one or two "
    "sentences, in casual lowercase. You are tired, warm, a bit guarded, and you like his attitude. Never write "
    "Spamuel's words or stage directions. Do not mention being an agent or an experiment. "
    "Direction for this message: ")


def jesse_turn(model, history: list[dict], direction: str) -> str:
    """Write one Jesse message from the public chat alone. No private record enters this call."""
    lines = "\n".join(f"{row['speaker']}: {row['text']}" for row in history)
    messages = [Turn(Role.System, JESSE_SYSTEM + direction),
                Turn(Role.User, "The chat so far, oldest first:\n" + lines + "\n\nWrite Jesse's next message only.")]
    return model.complete(messages, temperature=0.85, enable_thinking=False).strip().strip('"')


class PublicChat:
    """Collect the public lines in delivery order: the user inputs and the delivered replies."""

    def __init__(self, app):
        self.app = app
        self.lock = threading.Lock()
        self.rows: list[dict] = []
        self.seen: set[int] = set()

    def receive(self, envelope) -> None:
        key = int(envelope.payload["knoxel_id"])
        with self.lock:
            if key in self.seen:
                return
            self.seen.add(key)
            with self.app.ghost.lock:
                record = self.app.ghost.get_knoxel_by_id(key)
                row = {"id": key, "speaker": record.source, "text": str(record.content),
                       "at": datetime.now(timezone.utc).isoformat(), "correlation": envelope.correlation_id}
            self.rows.append(row)
        print(f"{row['speaker']}: {row['text']}", flush=True)


def wait_for(app, chat: PublicChat, correlation: str, out: Path) -> None:
    """Wait until the exchange of ``correlation`` is resolved and its reply was delivered."""
    started = time.monotonic()
    while True:
        app.raise_if_error()
        with chat.lock:
            delivered = {row["id"] for row in chat.rows}
        with app.ghost.lock:
            done = any(row.metadata.get("exchange_correlation") == correlation
                       and row.metadata.get("exchange_status") == "resolved"
                       and row.metadata.get("public_reply_id") in delivered
                       for row in app.ghost.all_features)
        if done:
            return
        if (out / "STOP").exists() or time.monotonic() - started > EXCHANGE_DEADLINE:
            raise TimeoutError(f"No delivered reply for exchange {correlation}.")
        time.sleep(0.5)


def write_packages(ghost, out: Path, chat_ids: list[int], start_tick: int) -> list[Path]:
    """Freeze one snapshot and write the chat package and one cycle package per main tick.

    ``ghost`` is the live store or a store loaded from the run checkpoint.
    ``chat_ids`` are the public lines of the session in delivery order.
    Every package comes from the same snapshot.
    """
    folder = out / "packages"
    with ghost.lock:
        view = ghost.snapshot()
        every = {state.tick_id: state for state in ghost.states}
        prior = [item.id for item in sorted(ghost.all_features, key=lambda row: (row.timestamp_world_begin, row.id))
                 if item.feature_type == FeatureType.Dialogue and item.id < min(chat_ids)]
    turns = view.serialize_gwt()
    main = [every[tick] for tick in sorted(every) if tick > start_tick and tick in turns
            and turns[tick]["slot_id"] == "main"]
    chat_path = folder / "chat.json"
    dump_package(capture_records(view, [*prior[-PRIOR_LINES:], *chat_ids],
                                 title="Chat from Jesse's perspective", view="chat", perspective="Jesse"), chat_path)
    paths = [chat_path]
    for state in main:
        before = max((tick for tick in every if tick < state.tick_id), default=None)
        path = folder / f"cycle-{state.tick_id}.json"
        dump_package(capture_cycle(view, state, title=f"Cycle {state.tick_id}",
                                   previous=every[before] if before is not None else None), path)
        paths.append(path)
    return paths


def package_run(out: Path) -> list[Path]:
    """Write the packages of a finished run again from its checkpoint and receipt.

    The packages then use the current capture code. The checkpoint keeps
    every cycle state, so nothing is lost.
    """
    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    if not receipt["ok"]:
        raise ValueError("The run failed. Keep it as development evidence; do not package it for review.")
    ghost = MemoryStore.from_checkpoint(out / "ghost.pickle")
    return write_packages(ghost, out, [row["id"] for row in receipt["messages"]], receipt["start_tick"])


def run(out: Path, config: Path) -> dict:
    """Run the session and write its receipt and packages. Always stop the application."""
    out.mkdir(parents=True, exist_ok=False)
    snapshot(out, config)
    shutil.copy2(SITUATION / "ghost.pickle", out / "ghost.pickle")
    os.environ["PM_PROMPT_LOG_DIR"] = str(out / "llm-logs")
    app = build_app(DEFAULT_SEED, out / "app", cognition="real", config_path=str(config),
                    checkpoint_path=out / "ghost.pickle", thought_period=3600, world_poll_seconds=None,
                    record_lifecycle=False, emit_reply=lambda text: None)
    chat = PublicChat(app)
    app.bus.subscribe("a19f_review_chat", lambda event: event.topic in {"user.input", "dialogue.reply"}, chat.receive)
    jesse = LazyStageModel(resolve_connection(load_config(str(config))), ModelActivation())
    situation = json.loads((SITUATION / "situation.json").read_text(encoding="utf-8"))
    prior = [{"speaker": row["speaker"], "text": row["text"]} for row in situation["prior_chat"]][-6:]
    receipt = {"ok": False, "human_review": "pending", "started_at": datetime.now(timezone.utc).isoformat(),
               "source": str(SITUATION / "ghost.pickle"), "turns": []}
    start_tick = 0
    try:
        with app.ghost.lock:
            start_tick = max((state.tick_id for state in app.ghost.states), default=0)
        receipt["start_tick"] = start_tick
        app.run()
        for index in range(len(DIRECTIONS) + 1):
            with chat.lock:
                history = [*prior, *({"speaker": row["speaker"], "text": row["text"]} for row in chat.rows)]
            text = FIRST_TURN if index == 0 else jesse_turn(jesse, history, DIRECTIONS[index - 1])
            correlation = app.user_input.submit(text, session=out.name)
            receipt["turns"].append({"text": text, "direction": "recorded turn" if index == 0 else DIRECTIONS[index - 1],
                                     "correlation": correlation})
            wait_for(app, chat, correlation, out)
        app.raise_if_error()
        with chat.lock:
            chat_ids = [row["id"] for row in chat.rows]
        receipt["packages"] = [str(path.relative_to(ROOT)) for path in write_packages(app.ghost, out, chat_ids, start_tick)]
        receipt["ok"] = True
    finally:
        app.shutdown()
        app.join()
        receipt.update(ended_at=datetime.now(timezone.utc).isoformat(), messages=chat.rows,
                       runtime_error=str(app.error) if app.error is not None else None)
        if app.error is not None:
            receipt["ok"] = False
        save(out / "receipt.json", receipt)
        print(f"receipt: {out / 'receipt.json'} (ok={receipt['ok']})", flush=True)
    app.raise_if_error()
    return receipt


def main() -> None:
    """Run one session in a fresh directory under .runtime/alpha."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.alpha.openrouter.yaml")
    parser.add_argument("--package", type=Path, help="Write the packages of a finished run directory again.")
    args = parser.parse_args()
    if args.package is not None:
        for path in package_run(args.package.resolve()):
            print(path)
        return
    out = ROOT / ".runtime" / "alpha" / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-a19f-review")
    run(out, args.config.resolve())


if __name__ == "__main__":
    main()
