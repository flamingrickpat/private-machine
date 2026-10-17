"""Exercise deliberate recall through the real kernel with the full canon.

This gate supplies a mental task at the action boundary. It does not prove
that action selection chooses the task or that a human approves the result.
All source snapshots, original evidence, and errors stay in a fresh run.
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
import shutil
from threading import Event, Thread
import time
import traceback
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pm.bus.envelopes import new_envelope
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.stage_model import LazyStageModel
from pm.llm.workflow_sessions import WorkflowModels
from pm.records.enums import ActionType
from pm.records.memory_records import ActionSchemaRecord
from workflows.companion import DEFAULT_SEED, build_app


def save(path: Path, value: object) -> None:
    """Write a private receipt. The caller owns the evidence directory."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def snapshot(out: Path, config: Path) -> None:
    """Retain exact working files and installed origins before real inference."""
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    ).decode().split("\0")
    hashes = {}
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(files) - {""}):
            path = ROOT / name
            if path.is_file():
                data = path.read_bytes()
                hashes[name] = hashlib.sha256(data).hexdigest()
                archive.writestr(name, data)
    origins = {}
    for name in ("pm-workflows", "pm-coder"):
        installed = distribution(name)
        origins[name] = {"version": installed.version,
                         "direct_url": json.loads(installed.read_text("direct_url.json") or "null")}
    save(out / "source-files.json", hashes)
    save(out / "source.json", {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "seed_sha256": hashlib.sha256(DEFAULT_SEED.read_bytes()).hexdigest(), "origins": origins})


def run(config_path: Path, out: Path, *, exercise_cognition: bool = False, fault: str | None = None,
        recall_checkpoint: Path | None = None, scenario: str | None = None,
        episode_attempts: int = 5, observe_expiry: bool = False) -> dict:
    """Run one real session, then inspect its committed proposal and sources.

    No application threads start. The real mental workflow runs synchronously
    against the full seed. Its own deadline and cancellation own agent cleanup.
    The receipt retains an error if provider work or validation fails.
    """
    out.mkdir(parents=True, exist_ok=False)
    snapshot(out, config_path)
    if recall_checkpoint is not None:
        parent = json.loads(recall_checkpoint.with_name("receipt.json").read_text(encoding="utf-8"))
        if not parent.get("passed"):
            raise ValueError("The source recall gate needs a completed passing receipt before checkpoint reuse.")
        shutil.copy2(recall_checkpoint, out / "ghost.json")
        save(out / "parent-receipt.json", parent)
        source_cache = recall_checkpoint.parent / "embeddings"
        if source_cache.is_dir():
            shutil.copytree(source_cache, out / "embeddings")
            save(out / "inherited-cache.json", {
                "source": str(source_cache),
                "files": {str(path.relative_to(source_cache)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_cache.rglob("*") if path.is_file()}})
    config = load_config(config_path)
    effective = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if fault == "provider":
        for connection in effective["chat"].values():
            connection.pop("api_key_env", None)
            connection["api_key"] = "a18c-deliberately-invalid-key"
            connection["context_window"] = 100000
    effective["embedding"]["cache_dir"] = str(out / "embeddings")
    effective["embedding"]["legacy_cache_dir"] = str(config.embedding.legacy_cache_dir)
    effective_path = out / "effective-config.yaml"
    effective_path.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    config = load_config(effective_path)
    owner = ModelActivation()
    model = LazyStageModel(resolve_connection(config), owner)
    models = WorkflowModels(config, owner)
    started = time.monotonic()
    receipt = {"scope": "real deliberate recall at the action boundary", "passed": False,
               "human_review": "pending", "started_at": datetime.utcnow().isoformat()}
    app = None
    finished = Event()
    watcher = None
    events = []
    try:
        app = build_app(DEFAULT_SEED, out / "app", cognition="real", config_path=str(effective_path),
                        stage_model=model, workflow_models=models, checkpoint_path=out / "ghost.json",
                        world_poll_seconds=None, record_lifecycle=False,
                        cognitive_control_config={"episode_steps": 3, "stalled_steps": 2,
                                                  "wait_seconds": 45, "maximum_wait_seconds": 60,
                                                  "mental_timeout_seconds": 1 if fault == "timeout" else 600})
        def control_deadline():
            while not finished.wait(0.2):
                if (out / "STOP").exists() or time.monotonic() - started >= 1800:
                    receipt["gate_stopped"] = True
                    app.shutdown()
                    return
        watcher = Thread(target=control_deadline, name="mental-gate-deadline", daemon=True)
        watcher.start()
        app.bus.observer = lambda envelope, recipient: events.append(
            {"recipient": recipient, "envelope": json.loads(envelope.to_json())})
        receipt["seed_records"] = len(app.ghost.all_knoxels)
        if receipt["seed_records"] < 11202:
            raise AssertionError("The full Spamuel seed did not load.")
        instruction = ("Privately recall an actual hobby or creative interest from your history with Jesse. "
                       "Inspect original conversation evidence. Try a revised query if the first search misses. "
                       "Use what you remember to propose a small activity or goal and a temporary focus. "
                       "Keep the choice your own. Do not send Jesse a message.")
        if recall_checkpoint is None:
            changes = app.ghost.begin_batch("mental-gate-action", new_tick=True)
            payload = {"action_id": "mental-gate-action", "workflow_name": "focus_management",
                       "content": "Perform private mental work.", "arguments": {"instruction": instruction}}
            action = changes.create(ActionSchemaRecord(content=instruction, action_type=ActionType.ToolCall,
                metadata={"gate_supplied": True, "action_id": "mental-gate-action",
                          "workflow_name": "focus_management", "dispatch_status": "pending",
                          "dispatch_correlation": "mental-gate", "dispatch_target": "focus_management",
                          "dispatch_payload": payload}))
            app.ghost.commit(changes)
            envelope = new_envelope("main_cog", "data", "action.request", "mental-gate",
                                   {"knoxel_id": action.id, **payload}, target="focus_management")
            app.focus_management.handle_batch([envelope])
        else:
            receipt["reused_recall_checkpoint_sha256"] = hashlib.sha256(recall_checkpoint.read_bytes()).hexdigest()
        if fault:
            accepted = app.ghost.get_knoxel_by_id(action.id)
            outcome = app.ghost.get_knoxel_by_id(accepted.metadata["mental_result_id"])
            if accepted.metadata.get("mental_status") != "failed" or "mental_plan" in outcome.metadata:
                raise AssertionError("The failed session did not retain a failed action result.")
            receipt.update(passed=True, scope="real mental-agent failure and result recovery", fault=fault,
                           outcome_id=outcome.id, error_type=outcome.metadata["error_type"])
            reopened = build_app(DEFAULT_SEED, out / "reopened", cognition="real", config_path=str(effective_path),
                stage_model=model, workflow_models=models, checkpoint_path=out / "ghost.json",
                world_poll_seconds=None, record_lifecycle=False)
            pending = reopened.main_cog.mailbox.take_batch(reopened.main_cog.stop_event, 0)
            if not any(item.payload.get("knoxel_id") == outcome.id for item in pending):
                raise AssertionError("Restart did not queue the durable failed result for appraisal.")
            if reopened.focus_management.last_result is not None:
                raise AssertionError("Restart repeated the failed agent session.")
            reopened.focus_management.handle_batch([envelope])
            if reopened.focus_management.last_result is not None:
                raise AssertionError("A duplicate accepted action repeated the failed session.")
            receipt["action_type"] = "ActionSchemaKnoxel"
            receipt["duplicate_action_did_not_repeat_agent"] = True
            receipt["recovered_without_agent_repeat"] = True
            return receipt
        proposals = [item for item in app.ghost.all_features if item.metadata.get("agent_session")]
        if len(proposals) != 1:
            raise AssertionError("The mental action did not commit exactly one proposal.")
        proposal = proposals[0]
        plan = proposal.metadata["mental_plan"]
        evidence = [app.ghost.snapshot().serialize_knoxels()[key] for key in plan["evidence_ids"]]
        if not evidence:
            raise AssertionError("Recall returned no inspected evidence.")
        receipt.update(passed=True, proposal_id=proposal.id, plan=plan,
                       source_revisions=proposal.metadata["source_revisions"],
                       invocation=getattr(app.focus_management, "invocation_id", None))
        save(out / "evidence.json", evidence)
        if exercise_cognition:
            receipt["passed"] = False
            save(out / "receipt.json", receipt)
            from workflows.cognitive_control.attention import active_focus
            receipt["focus_before_followup"] = active_focus(app.ghost)
            if scenario is not None:
                texts = {
                    "urgent": "hey urgent, my smoke alarm just went off and I smell something burning in the kitchen. "
                              "I'm stepping away to check it. I need your attention here, not on the playlist game.",
                    "private_options": "I'm torn between continuing our creative playlist game and making something different. "
                        "I don't want a chat reply right now. Privately consider two distinct ideas and what you actually "
                        "know about my preferences. Keep the alternatives open until you've compared them. "
                        "It's fine to put it aside if it goes nowhere.",
                }
                correlation = app.user_input.submit(texts[scenario], session="a18c-mental-followup")
                receipt["followup"] = {"scenario": scenario, "text": texts[scenario], "correlation": correlation}
                incoming = app.main_cog.mailbox.take_batch(app.main_cog.stop_event, 0)
            else:
                incoming = [new_envelope("focus_management", "data", "thought.created", "mental-gate",
                                        {"knoxel_id": proposal.id})]
            app.main_cog.handle_batch(incoming)
            report = app.main_cog.last_result.report
            receipt["main_appraisal"] = {"invocation": app.main_cog.invocation_id,
                                          "focus_decision": report.get("focus_decision"),
                                          "goal_changes": report.get("goal_changes")}
            save(out / "main-report.json", report)
            receipt["coverage"] = {"main_appraisal": True,
                "focus_accepted": bool((report.get("focus_decision") or {}).get("accept"))}
            competition = report.get("codelet_trace", {}).get("competition", {})
            receipt["coverage"].update(
                nonzero_focus_bias=any(value != 0 for value in competition.get("deliberate_bias", {}).values()),
                danger=competition.get("danger"),
                direct_stimulus_selected=any(item["codelet"] == "direct_stimulus" and item["status"] == "selected"
                    for item in report.get("nominations", [])))
            receipt["episodes"] = []
            for number in range(episode_attempts):
                before = set(app.ghost.all_knoxels)
                goals_before = {key: item.model_dump(mode="json") for key, item in app.ghost.all_knoxels.items()
                                if type(item).__name__ == "Intention"}
                app.inner_thought.handle_batch([])
                goals_after = {key: item.model_dump(mode="json") for key, item in app.ghost.all_knoxels.items()
                               if type(item).__name__ == "Intention"}
                if goals_before != goals_after:
                    raise AssertionError("A private step changed accepted goals before main appraisal.")
                fresh = [item for item in app.ghost.all_features
                         if item.id not in before and item.metadata.get("episode")]
                step = {"attempt": number, "invocation": app.inner_thought.invocation_id,
                        "thoughts": [{"id": item.id, "text": item.content,
                                      "metadata": item.metadata} for item in fresh]}
                receipt["episodes"].append(step)
                save(out / "receipt.json", receipt)
                if not fresh:
                    continue
                current = fresh[-1]
                if not current.metadata["episode"]["closed"]:
                    request_count = model.request_count
                    thought_count = len([item for item in app.ghost.all_features if item.metadata.get("loop") == "inner"])
                    app.inner_thought.handle_batch([])
                    after_count = len([item for item in app.ghost.all_features if item.metadata.get("loop") == "inner"])
                    if after_count != thought_count or model.request_count != request_count:
                        raise AssertionError("A continuing private step advanced before main appraisal.")
                app.main_cog.handle_batch([new_envelope("inner_thought", "data", "thought.created", "",
                                                      {"knoxel_id": current.id})])
                if current.metadata["episode"].get("parked_until"):
                    app.inner_thought.last_environment = app.inner_thought.working_target.open("parking-inspection")
                    request = app.inner_thought.prepare_input([])
                    receipt["parked_sources_next_request"] = request["episode"]["suppressed_source_ids"]
                    if any(thought["metadata"]["episode"]["sequence"] > 0
                           for attempt in receipt["episodes"] for thought in attempt["thoughts"]):
                        break
                if current.metadata["episode"]["closed"] and current.metadata["episode"]["sequence"] > 0:
                    break
            all_steps = [thought for attempt in receipt["episodes"] for thought in attempt["thoughts"]]
            receipt["coverage"].update(
                continued_episode=any(item["metadata"]["episode"]["sequence"] > 0 for item in all_steps),
                closed_episode=any(item["metadata"]["episode"]["closed"] for item in all_steps),
                parked_concern=bool(receipt.get("parked_sources_next_request")))
            if scenario == "urgent" and not receipt["coverage"]["direct_stimulus_selected"]:
                raise AssertionError("The urgent outside stimulus lost its reserved attention place.")
            if scenario == "private_options" and not receipt["coverage"]["continued_episode"]:
                raise AssertionError("No real continuing episode occurred in the requested private comparison.")
            if observe_expiry:
                initial = receipt["focus_before_followup"]
                if not initial.get("accept"):
                    raise AssertionError("The focus expiry case started without effective advice.")
                while active_focus(app.ghost) and not finished.wait(0.2):
                    if receipt.get("gate_stopped"):
                        raise TimeoutError("The gate deadline preceded observed focus expiry.")
                receipt["coverage"]["real_time_focus_expiry"] = not bool(active_focus(app.ghost))
            receipt["passed"] = True
    except Exception as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        finished.set()
        if watcher is not None:
            watcher.join()
        if app is not None:
            app.shutdown()
            app.join()
        if receipt.get("gate_stopped"):
            receipt["passed"] = False
        save(out / "bus.json", events)
        receipt["elapsed_seconds"] = time.monotonic() - started
        save(out / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exercise-cognition", action="store_true")
    parser.add_argument("--fault", choices=("timeout", "provider"))
    parser.add_argument("--recall-checkpoint", type=Path)
    parser.add_argument("--scenario", choices=("urgent", "private_options"))
    parser.add_argument("--episode-attempts", type=int, default=5)
    parser.add_argument("--observe-expiry", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config.resolve(), arguments.out.resolve(),
                 exercise_cognition=arguments.exercise_cognition, fault=arguments.fault,
                 recall_checkpoint=arguments.recall_checkpoint.resolve() if arguments.recall_checkpoint else None,
                 scenario=arguments.scenario, episode_attempts=arguments.episode_attempts,
                 observe_expiry=arguments.observe_expiry)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)
