"""Lose publication after real cognition, then recover the same identities."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class PublicationLost(RuntimeError):
    """The gate intentionally interrupts an owned bus delivery."""


def run(args):
    """Use full history, real selection, real speech, and a real note worker.

    Only bus delivery fails. No model response or workflow result is replaced.
    Each restart loads the same checkpoint and retains accepted identities.
    The note request is an explicit operator boundary test.
    """
    from bus.bus import BusSystem
    from main import build_app
    from pm_core.ghost.service import Ghost
    from pm_core.model.knoxel_common import ActionSchemaKnoxel
    from runtime.body_lease import atomic_json
    from runtime.intents import replay_intents
    from runtime.managed_runs import ManagedRunSupervisor
    from runtime.publication import exchange_record
    from dev.work_items.alpha_a12_recovery_gate import wait_for

    out = args.out.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Keep live publication evidence below .runtime.")
    out.mkdir(parents=True, exist_ok=args.resume)
    seed = ROOT / "data/spamuel_agentic_canon.yaml"
    checkpoint = out / "ghost.json"
    receipt = {"item": "A12", "status": "running", "checks": {},
               "canon_sha256": hashlib.sha256(seed.read_bytes()).hexdigest(),
               "scope": "real companion selection and speech; operator-selected real note task"}
    if args.resume and (out / "receipt.json").is_file():
        from runtime.body_lease import read_json
        atomic_json(out / f"receipt-prior-{time.time_ns()}.json", read_json(out / "receipt.json"))

    def save():
        atomic_json(out / "receipt.json", receipt)

    def build():
        return build_app(seed, out / "app", cognition="real", config_path=str(args.config.resolve()),
                         checkpoint_path=checkpoint, record_lifecycle=False, emit_reply=lambda text: None)

    save()
    try:
        app = build()
        lost = []
        def lose_intent(envelope, recipient):
            if envelope.topic == "action.request":
                lost.append(json.loads(envelope.to_json()))
                raise PublicationLost("Accepted companion intent was not delivered.")
        app.bus.observer = lose_intent
        if args.resume and (out / "lost-intent.json").is_file():
            lost.append(json.loads((out / "lost-intent.json").read_text(encoding="utf-8")))
            correlation = lost[0]["correlation_id"]
        else:
            correlation = app.user_input.submit("Jesse: Spamuel, say hello to me in one short sentence. This is a chat request.",
                                                session="a12-publication-recovery")
            try:
                app.main_cog.step()
            except PublicationLost:
                pass
        if not lost:
            raise AssertionError("Real cognition did not reach accepted dispatch publication.")
        if lost[0]["target"] not in {"dialogue_story", "dialogue_assistant"}:
            raise AssertionError("The real model selected a non-dialogue action for this chat request.")
        atomic_json(out / "lost-intent.json", lost[0])
        if app.main_cog.last_result is not None:
            atomic_json(out / "cognition-report.json", app.main_cog.last_result.report)
        original_id = lost[0]["payload"]["knoxel_id"]
        frozen = Ghost.from_checkpoint(checkpoint)
        action = frozen.get_knoxel_by_id(original_id)
        if not isinstance(action, ActionSchemaKnoxel) or action.metadata["dispatch_status"] != "pending":
            raise AssertionError("The accepted companion intent was not checkpointed.")
        # Inspect the actual recovery envelope before delivering any workflow.
        bus, replays = BusSystem(), []
        bus.subscribe("capture", lambda envelope: envelope.topic == "action.request", replays.append)
        inspection = ManagedRunSupervisor(frozen, runtime_root=out / "inspection", workflows={}, clock=time.monotonic)
        replay_intents(frozen, bus, inspection)
        receipt["checks"]["accepted_intent_identity"] = len(replays) == 1 and (
            replays[0].payload["knoxel_id"] == original_id and replays[0].correlation_id == correlation
            and replays[0].payload["action_id"] == lost[0]["payload"]["action_id"])
        atomic_json(out / "replayed-intent.json", json.loads(replays[0].to_json()))

        app = build()
        speech = []
        def lose_speech(envelope, recipient):
            if envelope.topic == "dialogue.reply":
                speech.append(json.loads(envelope.to_json()))
                raise PublicationLost("Committed public reply was not delivered.")
        app.bus.observer = lose_speech
        selected = app.dialogue if lost[0]["target"] == "dialogue_story" else app.assistant
        try:
            selected.step()
        except PublicationLost:
            pass
        if not speech:
            raise AssertionError("The real dialogue workflow did not reach public reply publication.")
        atomic_json(out / "lost-speech.json", speech[0])
        reply_id = speech[0]["payload"]["knoxel_id"]
        frozen = Ghost.from_checkpoint(checkpoint)
        exchange = exchange_record(frozen, correlation)
        receipt["checks"]["reply_committed_before_delivery"] = (
            exchange.metadata["public_reply_id"] == reply_id
            and frozen.get_knoxel_by_id(reply_id).metadata["publication_status"] == "pending")
        receipt["reply"] = {"id": reply_id, "text": frozen.get_knoxel_by_id(reply_id).content,
                            "correlation": correlation, "action_id": original_id}
        app = build()
        delivered = []
        app.user_input.emit_reply = delivered.append
        app.user_input.step()
        app.reply_recovery.resume()
        app.user_input.step()
        receipt["checks"]["single_reply_delivery"] = delivered == [receipt["reply"]["text"]]
        receipt["checks"]["no_duplicate_accepted_intent"] = sum(
            isinstance(item, ActionSchemaKnoxel) and item.metadata.get("action_id") == action.metadata["action_id"]
            for item in app.ghost.all_knoxels.values()) == 1

        # A real native note supplies terminal progress and a verifiable effect.
        # Fail after its completed event is durable, before a consumer can ack.
        lost_progress, errors = [], []
        normal_sink = app.supervisor.event_sink
        def lose_progress(event):
            if event["event"] == "completed":
                lost_progress.append(dict(event))
                raise PublicationLost("Completed progress was not delivered.")
            normal_sink(event)
        app.supervisor.event_sink = lose_progress
        app.supervisor.on_error = errors.append
        task = app.supervisor.start(action_id="a12-progress-note", workflow_name="note_task",
            instructions="Write the exact UTF-8 note specified by note_spec.",
            arguments={"note_spec": {"path": "outbox/recovery-note.txt", "text": "A12 progress survives publication loss."}})
        wait_for(lambda: lost_progress, args.timeout)
        app.supervisor.join(task["run_id"], 30)
        final = app.supervisor.status(task["run_id"])
        actual = (Path(final["artifact_path"]) / "outbox/recovery-note.txt").read_bytes()
        receipt["checks"]["real_note_effect"] = actual == b"A12 progress survives publication loss."
        receipt["checks"]["publication_loss_preserves_completed_state"] = final["state"] == "completed" and not any(
            row["message"]["event"] == "failed" for row in final["events"])
        receipt["progress"] = {"original": lost_progress[0], "receipt": final,
                               "publication_errors": [str(error) for error in errors]}
        atomic_json(out / "lost-progress.json", lost_progress[0])
        app = build()
        progress = []
        app.bus.subscribe("progress-capture", lambda envelope: envelope.topic == "workflow.event", progress.append)
        app.supervisor.replay_events()
        original_event = lost_progress[0]["event_id"]
        receipt["checks"]["progress_identity"] = any(event.payload.get("event_id") == original_event for event in progress)
        receipt["checks"]["recovery_starts_no_worker"] = not app.supervisor._threads
        for event in progress:
            app.supervisor.acknowledge_event(event.payload["run_id"], event.payload["event_id"])
        count = len(progress)
        app.supervisor.replay_events()
        receipt["checks"]["acknowledged_progress_stops_replay"] = len(progress) == count
        if not all(receipt["checks"].values()):
            raise AssertionError("A live publication recovery contract failed.")
        receipt["status"] = "passed"
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(out / "receipt.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.alpha.example.yaml")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--resume", action="store_true", help="Continue after the retained accepted-intent boundary.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
