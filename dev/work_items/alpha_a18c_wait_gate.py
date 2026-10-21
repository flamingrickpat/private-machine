"""Exercise deliberate wait, external wake, timeout, and restart with real inference.

This focused gate uses the full Spamuel seed and the actual concurrent assembly.
It does not inject action choices or establish A18C's remaining mental-work
capabilities. Raw replies, bus events, checkpoints, and kernel artifacts remain
under the fresh output directory. Human immersion review stays separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from importlib.metadata import distribution
import json
from pathlib import Path
import subprocess
import sys
from threading import Lock
import time
import traceback
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.stage_model import LazyStageModel
from pm.llm.workflow_sessions import WorkflowModels
from workflows.companion import DEFAULT_SEED, build_app
from dev.work_items.a18b.runner import accepted_pause


def save(path: Path, data: object) -> None:
    """Write an inspectable receipt without changing retained earlier evidence."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def run(config_path: Path, out: Path, deadline_seconds: float = 1800, *, pause_only: bool = False) -> dict:
    """Run a fresh real gate. Raise on reuse of an existing evidence directory.

    Deadline and application errors end the gate with a failed receipt. Cleanup
    cancels active calls and joins owned work. No Minecraft fixture is needed:
    this gate disables world polling and asks for no game action.
    """
    out.mkdir(parents=True, exist_ok=False)
    config = load_config(config_path)
    effective = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    effective_path = out / "effective-config.yaml"
    effective_path.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    config = load_config(effective_path)
    owner = ModelActivation()
    model = LazyStageModel(resolve_connection(config), owner)
    models = WorkflowModels(config, owner)
    started = time.monotonic()
    end = started + deadline_seconds
    facts: dict = {}
    receipt = {"scope": "A18C wait only", "passed": False, "human_review": "pending", "facts": facts}
    replies: list[dict] = []
    reply_records: dict[str, list[int]] = {}
    routine_calls = [0]
    app = None
    log_lock = Lock()
    bus_log = (out / "bus.jsonl").open("a", encoding="utf-8")

    def observe(envelope, recipient):
        with log_lock:
            if envelope.topic == "dialogue.reply" and recipient == "user_input":
                reply_records.setdefault(envelope.correlation_id, []).append(envelope.payload["knoxel_id"])
            bus_log.write(json.dumps({"at": datetime.utcnow().isoformat(), "recipient": recipient,
                                       "envelope": json.loads(envelope.to_json())}, ensure_ascii=False) + "\n")
            bus_log.flush()

    def reply(text):
        with log_lock:
            replies.append({"at": datetime.utcnow().isoformat(), "text": text})
            save(out / "replies.json", replies)

    def start_app():
        result = build_app(DEFAULT_SEED, out / "app", cognition="real", config_path=str(effective_path),
            stage_model=model, workflow_models=models, checkpoint_path=out / "ghost.pickle",
            emit_reply=reply, world_poll_seconds=None,
            cognitive_control_config={"wait_seconds": 45, "maximum_wait_seconds": 60})
        # Instrument real model entry without replacing inference or results.
        # Main cognition phases call the model through ``llm_complete``
        # (A19F). The inner loop still runs its stages through ``run_stage``.
        def instrument(factory):
            def tools(environment, batch):
                bound = dict(factory(environment, batch))
                name = "llm_complete" if "llm_complete" in bound else "run_stage"
                original = bound[name]
                def counted(*args, **kwargs):
                    with log_lock:
                        routine_calls[0] += 1
                    return original(*args, **kwargs)
                return {**bound, name: counted}
            return tools
        for module in (result.main_cog, result.inner_thought):
            module.tools_factory = instrument(module.tools_factory)
        result.bus.observer = observe
        result.run()
        return result

    def until(predicate, label):
        while time.monotonic() < end:
            if (out / "STOP").exists():
                raise InterruptedError("The operator stopped this focused gate.")
            app.raise_if_error()
            found = predicate()
            if found:
                return found
            time.sleep(0.2)
        raise TimeoutError(f"Gate deadline while waiting for {label}.")

    def replied(correlation):
        with log_lock:
            ids = list(reply_records.get(correlation, []))
        return any(app.ghost.get_knoxel_by_id(key).metadata.get("publication_status") == "published"
                   for key in ids)

    def wait_record():
        return next((item for item in reversed(list(app.ghost.all_features))
                     if item.metadata.get("control_kind") == "inference_wait"
                     and item.metadata.get("status") == "waiting"), None)

    def submit(text):
        correlation = app.user_input.submit(text, session="a18c-wait-gate")
        facts.setdefault("inputs", []).append({"text": text, "correlation": correlation,
                                              "at": datetime.utcnow().isoformat()})
        save(out / "facts.json", facts)
        return correlation

    def quiet(label):
        correlation = submit("I'm stepping away to read for a bit. Please actually wait for a new stimulus now: "
               "stop active thought and leave it quiet until my next message or the configured timer. "
               "No acknowledgment or explanation is needed.")
        record = until(wait_record, label)
        pause = accepted_pause(app.ghost, correlation)
        if pause is None:
            raise AssertionError("The accepted wait does not pause the current pending exchange.")
        facts[label] = {"record_id": record.id, "metadata": dict(record.metadata), "accepted_pause": pause}
        save(out / "facts.json", facts)
        return record

    try:
        origins = {}
        for name in ("pm-workflows", "pm-coder"):
            dist = distribution(name)
            origins[name] = {"version": dist.version, "direct_url": json.loads(dist.read_text("direct_url.json") or "null")}
        save(out / "source.json", {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
            "seed": str(DEFAULT_SEED), "seed_sha256": hashlib.sha256(DEFAULT_SEED.read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "effective_config_sha256": hashlib.sha256(effective_path.read_bytes()).hexdigest(),
            "model": model.connection.model, "origins": origins})
        patch = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
        (out / "source.patch").write_bytes(patch)
        files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
        hashes = {}
        with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(set(files) - {""}):
                path = ROOT / name
                if path.is_file():
                    data = path.read_bytes()
                    hashes[name] = hashlib.sha256(data).hexdigest()
                    archive.writestr(name, data)
        save(out / "source-files.json", hashes)
        app = start_app()
        facts["initial_records"] = len(app.ghost.all_knoxels)
        if facts["initial_records"] < 11202:
            raise AssertionError("The gate did not import the full Spamuel history.")
        correlation = submit("heyo, taking a quick break. how's it going? :3")
        until(lambda: replied(correlation), "correlated ordinary conversational reply")
        quiet("first_wait")
        # Existing requests can still finish cleanup. The subsequent interval
        # checks new calls and accepted cycles, not transport cleanup latency.
        time.sleep(2)
        calls, cycles, stages = model.request_count, len(app.ghost.states), routine_calls[0]
        invocations = (app.main_cog.invocation_id, app.inner_thought.invocation_id)
        time.sleep(5)
        facts["quiet_interval"] = {"seconds": 5, "calls_before": calls, "calls_after": model.request_count,
                                   "routine_stages_before": stages, "routine_stages_after": routine_calls[0],
                                   "cycles_before": cycles, "cycles_after": len(app.ghost.states)}
        if (stages != routine_calls[0] or cycles != len(app.ghost.states)
                or invocations != (app.main_cog.invocation_id, app.inner_thought.invocation_id)):
            raise AssertionError("Routine inference or a cycle continued during an accepted wait.")
        if pause_only:
            receipt.update(passed=True, scope="ordinary conversation then accepted pause without another input")
            return receipt
        before = len(replies)
        correlation = submit("back! I changed my mind about reading. got a moment to chat?")
        until(lambda: replied(correlation), "correlated external-wake reply")
        source = next(item for item in app.ghost.all_features if item.metadata.get("exchange_correlation") == correlation)
        if not any(source.id in cycle.claimed_ids() for cycle in app.ghost.states):
            raise AssertionError("The wake message did not reach a committed cognition cycle.")
        facts["external_wake"] = {"source_id": source.id, "replies": replies[before:]}
        record = quiet("restart_wait")
        app.shutdown()
        app.join()
        app.raise_if_error()
        state_before = {item.id for item in app.ghost.all_features if item.metadata.get("loop") == "inner"}
        stopped_at = datetime.utcnow().isoformat()
        app = start_app()
        state_after = {item.id for item in app.ghost.all_features if item.metadata.get("loop") == "inner"}
        if state_before != state_after:
            raise AssertionError("Restart invented private work before the next running cycle.")
        if not app.cognitive_control.waiting():
            raise AssertionError("Restart lost the accepted wait before its recorded deadline.")
        facts["restart"] = {"stopped_at": stopped_at, "resumed_at": datetime.utcnow().isoformat(),
                            "wait_record_id": record.id, "private_ids_equal": True}
        wake = until(lambda: next((item for item in app.ghost.all_features
            if item.metadata.get("control_kind") == "inference_wake"
            and item.metadata.get("source_event_ids") == [record.id]
            and item.metadata.get("wake_status") == "consumed"), None), "timeout appraisal commit")
        if not any(wake.id in cycle.claimed_ids() for cycle in app.ghost.states):
            raise AssertionError("The timer wake has no accepted main cognition cycle.")
        facts["timeout"] = {"wake_record_id": wake.id, "metadata": dict(wake.metadata),
                            "source_kind": wake.provenance.source_kind.value}
        receipt["passed"] = True
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if app is not None:
            try:
                app.shutdown()
                app.join()
                if app.error is not None:
                    receipt.update(passed=False, application_error=f"{type(app.error).__name__}: {app.error}")
            except BaseException as error:
                receipt.update(passed=False, cleanup_error=f"{type(error).__name__}: {error}")
                (out / "cleanup-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        bus_log.close()
        save(out / "facts.json", facts)
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        save(out / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.alpha.openrouter.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, default=1800)
    parser.add_argument("--pause-only", action="store_true")
    args = parser.parse_args()
    receipt = run(args.config.resolve(), args.out.resolve(), args.deadline, pause_only=args.pause_only)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
