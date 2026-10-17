"""Continue an accepted reflection checkpoint through real public conversation.

The source checkpoint stays unchanged. The application owns routing and speech.
A deadline or runtime error retains evidence and prevents review registration.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.work_items.alpha_a18c_mental_gate import save, snapshot
from dev.work_items.conversation_review import ChatCapture
from workflows.companion import DEFAULT_SEED, build_app


def run(source: Path, out: Path, config: Path, *, drain_only: bool = False) -> dict:
    """Copy accepted state and capture actual replies to three later inputs.

    Raise on missing speech or application failure. Always stop owned workers.
    The returned receipt contains delivery order, without human ratings.
    """
    out.mkdir(parents=True, exist_ok=False)
    snapshot(out, config)
    shutil.copy2(source, out / "ghost.json")
    os.environ["PM_PROMPT_LOG_DIR"] = str(out / "llm-logs")
    app = build_app(DEFAULT_SEED, out / "app", cognition="real", config_path=str(config),
                    checkpoint_path=out / "ghost.json", thought_period=3600,
                    world_poll_seconds=None, record_lifecycle=False, emit_reply=lambda text: None)
    chat = ChatCapture(app, out)
    parent_receipt = json.loads(source.with_name("receipt.json").read_text(encoding="utf-8"))
    chat.rows.extend(parent_receipt.get("messages", []))
    chat.seen.update(row["id"] for row in chat.rows)
    app.bus.subscribe("a19_review_chat", lambda event: event.topic in {"user.input", "dialogue.reply"}, chat.receive)
    receipt = {"ok": False, "source_checkpoint": str(source), "human_review": "pending",
               "started_at": datetime.now(timezone.utc).isoformat(), "messages": []}
    started = time.monotonic()

    def wait_for_exchanges(correlations: set[str]) -> None:
        # A reply to an older queued request must not complete a newer request.
        # Public arrival order alone does not identify the answered exchange.
        while True:
            app.raise_if_error()
            with chat.lock:
                delivered_ids = {row["id"] for row in chat.rows if row["speaker"] == "Spamuel"}
            with app.ghost.lock:
                sources = [row for row in app.ghost.all_features
                           if row.metadata.get("exchange_correlation") in correlations]
                if len(sources) == len(correlations) and all(
                        row.metadata.get("exchange_status") == "resolved"
                        and row.metadata.get("public_reply_id") in delivered_ids
                        for row in sources):
                    return
            if (out / "STOP").exists() or time.monotonic() - started > 1800:
                raise TimeoutError("The review conversation ended before public speech arrived.")
            time.sleep(0.5)

    try:
        with app.ghost.lock:
            pending = {row.metadata["exchange_correlation"] for row in app.ghost.all_features
                       if row.metadata.get("exchange_status") == "pending"}
        app.run()
        wait_for_exchanges(pending)
        turns = () if drain_only else (
            "gotta run. later.",
            "back. and yes, our story. can we try what each ending would feel like, without choosing yet?",
            "the unfinished one feels more like us. i still don't want to lock it in tonight though.",
        )
        for text in turns:
            correlation = app.user_input.submit(text, session=out.name)
            wait_for_exchanges({correlation})
        app.raise_if_error()
        receipt["ok"] = True
    finally:
        app.shutdown()
        app.join()
        receipt.update(ended_at=datetime.now(timezone.utc).isoformat(), messages=chat.rows,
                       runtime_error=str(app.error) if app.error is not None else None)
        if app.error is not None:
            receipt["ok"] = False
        save(out / "receipt.json", receipt)
    app.raise_if_error()
    return receipt


def main() -> None:
    """Read explicit evidence paths and run one isolated conversation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.alpha.openrouter.yaml")
    parser.add_argument("--drain-only", action="store_true")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if not source.is_relative_to(ROOT / ".runtime") or not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Review evidence belongs under .runtime.")
    run(source, out, args.config.resolve(), drain_only=args.drain_only)


if __name__ == "__main__":
    main()
