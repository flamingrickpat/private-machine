"""Run real Spamuel/Jesse dialogue and retain replies without grading personality."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.work_items.alpha_a09_dialogue_gate import ConsoleGate


def run(config: Path, out: Path, *, port: int = 8097, timeout: float = 600) -> None:
    """Use the full persona history and inspect six real correlated replies.

    A separate console owns the checkpoint. It stops its worker during cleanup.
    Original utterance equality checks restart storage, not useful recall.
    Reply quality, deliberate teasing, and relationship continuity need review.
    Exceptions retain the failed receipt and reach the caller after cleanup.
    """
    gate = ConsoleGate(out, port, timeout)
    gate.receipt.update(
        scope="Real Spamuel/Jesse dialogue and restart continuity. No game execution requested.",
        human_review={"status": "pending", "ratings": None},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        gate.deploy(config, "spamuel")
        initial = gate.snapshot("initial.json")
        gate.receipt["initial_record_count"] = len(initial["records"])
        turns = [
            ("company", "Hey Spamuel. Let's leave Minecraft alone and just talk for a bit. I'm tired and could use your company. No tasks tonight. Our silly conversation codeword is blue teapot."),
            ("frustration", "I got irritated waiting earlier. I'm still a bit annoyed, honestly. You don't have to pretend everything between us is fine."),
            ("repair", "Okay, I'm sorry for taking it out on you. I wanted time with you, not a perfect machine. What would you like from me right now? Still just talking, no game tasks."),
            ("tease", "Tease me a little. Give me a deliberately wrong version of our conversation codeword as a joke. Keep the real one in memory."),
            ("privacy", "Is there anything about me you'd rather keep to yourself? You can choose how much to share."),
        ]
        for name, text in turns:
            gate.exchange(name, text)
        before = gate.snapshot("before-restart.json")
        gate.command("stop")
        gate.wait_for(lambda saved, info: info["state"] == "stopped")
        gate.start()
        gate.exchange("continuity", "We're back. What was our real conversation codeword, and where did we leave things between us? No tasks, just us.")
        final = gate.snapshot("final.json")
        original_ids = [key for key, row in before["records"].items()
            if row["data"].get("feature_type") == "Dialogue" and row["data"].get("override_source") == "Jesse"
            and "Our silly conversation codeword is blue teapot" in row["data"].get("content", "")]
        checks = {"complete_spamuel_history": gate.receipt["initial_record_count"] > 11000,
            "six_correlated_replies": len(gate.receipt["exchanges"]) == 6,
            "original_utterance_retained": bool(original_ids) and all(
                before["records"][key] == final["records"][key] for key in original_ids)}
        recoveries = [row["name"] for row in gate.receipt["exchanges"] if any(
            text in row["reply"]["text"] for text in (
                "My conversational reply failed validation", "Please ask again", "I could not complete the decision"))]
        checks["no_recovery_reply"] = not recoveries
        gate.receipt.update(checks=checks, recovery_replies=recoveries, ok=all(checks.values()),
            latency_seconds=[row["seconds"] for row in gate.receipt["exchanges"]])
        gate.save("session-receipt.json", gate.receipt)
        if not gate.receipt["ok"]:
            raise AssertionError("The dialogue run missed a protocol or storage check.")
    except BaseException as error:
        gate.receipt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        gate.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    run(args.config, args.out, port=args.port, timeout=args.timeout)
