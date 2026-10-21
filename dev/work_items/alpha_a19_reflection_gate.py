"""Exercise reflection with real inference, the full canon, and a fresh checkpoint.

The fixture selects reflection at the private-operation boundary. It introduces
an explicit mistaken subjective belief and an attributed user correction.
The real private parent proposes and applies a change. Real main cognition
then records later choices and the effective version it consumed.
This gate does not establish autonomous trigger selection or human quality.
"""
import argparse
from datetime import datetime
import json
import hashlib
from pathlib import Path
import sys
import time
import traceback
import shutil
from threading import Event, Thread

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dev.work_items.alpha_a18c_mental_gate import save, snapshot
from pm.bus.envelopes import new_envelope, new_id
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config, resolve_connection
from pm.llm.stage_model import LazyStageModel
from pm.records.cognitive_records import BeliefRecord
from pm.records.enums import FeatureType, SourceKind
from pm.records.memory_records import new_feature
from pm.runs.workflow_runner import WorkflowRequest
from workflows.companion import DEFAULT_SEED, build_app
from workflows.reflection.experiments import experiments
from workflows.reflection.schemas import ReflectionDecision
from workflows.inner_thought.schemas import InnerReport


def private_step(app, instruction: str) -> dict:
    """Select the reflection operation, then run its real parent and children."""
    inner = app.inner_thought
    inner.run_stop.clear()
    inner.invocation_id = new_id()
    inner.last_environment = inner.working_target.open(inner.invocation_id)
    inner._control_generation = app.cognitive_control.generation
    payload = inner.prepare_input([])
    payload["reflection"]["review_request"] = instruction
    payload["trigger_candidates"] = [{"name": "experiment_review", "cause": instruction,
        "urgency": 0.8, "operations": ["reflection_review"], "reason": "A selected live reflection fixture."}]
    inner._execute(WorkflowRequest(inner.invocation_id, payload), inner.last_environment, [], resume=False)
    if inner.last_result is None:
        raise RuntimeError("The real private parent produced no result.")
    return {"invocation_id": inner.invocation_id, "report": inner.last_result.report}


def finish_evaluation(app, out: Path, receipt: dict) -> None:
    """Evaluate the existing prediction and compare its durable reopened record."""
    step = private_step(app, "Evaluate the existing experiment against the two actual later main choices. "
        "Keep unresolved claims explicit. Do not infer public delivery or human approval from action selection.")
    receipt["private_steps"].append(step)
    from pm.memory.store import MemoryStore
    reopened = MemoryStore.from_checkpoint(out / "ghost.pickle")
    receipt["experiments"] = [{"id": item.id, **item.metadata} for item in experiments(reopened.snapshot())]
    accepted = reopened.get_knoxel_by_id(receipt["experiment_id"])
    if not any(event["action"] == "evaluated" for event in accepted.metadata["history"]):
        raise AssertionError("Real inference did not evaluate the recorded prediction.")
    receipt["reopened_equal"] = (app.ghost.get_knoxel_by_id(accepted.id).model_dump(mode="json")
                                 == accepted.model_dump(mode="json"))
    receipt["passed"] = receipt["reopened_equal"]


def run(config_path: Path, out: Path, *, steps: int = 6,
        evaluation_checkpoint: Path | None = None) -> dict:
    """Retain real outcomes and errors without assigning a human rating."""
    out.mkdir(parents=True, exist_ok=False)
    snapshot(out, config_path)
    import yaml
    effective = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    effective_path = out / "effective-config.yaml"
    effective_path.write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
    model_config = load_config(effective_path)
    model = LazyStageModel(resolve_connection(model_config), ModelActivation())
    started = time.monotonic()
    receipt = {"passed": False, "human_review": "pending", "scope": "selected private-operation boundary",
               "started_at": datetime.utcnow().isoformat(), "private_steps": [], "later_choices": []}
    app = None
    finished = Event()
    watcher = None
    try:
        parent = None
        if evaluation_checkpoint is not None:
            parent = json.loads(evaluation_checkpoint.with_name("receipt.json").read_text(encoding="utf-8"))
            if len(parent.get("later_choices", [])) < 2 or not parent.get("experiment_id"):
                raise ValueError("Evaluation recovery needs an applied experiment and two real consuming choices.")
            shutil.copy2(evaluation_checkpoint, out / "ghost.pickle")
            save(out / "parent-receipt.json", parent)
            receipt["evaluation_parent"] = {"checkpoint": str(evaluation_checkpoint),
                "sha256": hashlib.sha256(evaluation_checkpoint.read_bytes()).hexdigest(),
                "parent_passed": parent.get("passed"), "parent_error": parent.get("error")}
        app = build_app(DEFAULT_SEED, out / "app", cognition="real", config_path=str(effective_path),
                        stage_model=model, checkpoint_path=out / "ghost.pickle", world_poll_seconds=None,
                        record_lifecycle=False)
        receipt["seed_records"] = len(app.ghost.all_knoxels)
        if receipt["seed_records"] < 11202:
            raise AssertionError("The full Spamuel canon did not load.")
        def deadline():
            while not finished.wait(0.5):
                if (out / "STOP").exists() or time.monotonic() - started > 2400:
                    receipt["gate_stopped"] = True
                    app.shutdown()
                    return
        watcher = Thread(target=deadline, name="reflection-gate-deadline", daemon=True)
        watcher.start()
        if parent is not None:
            receipt.update(experiment_id=parent["experiment_id"], effective_record_id=parent["effective_record_id"],
                           later_choices=parent["later_choices"], fixture=parent["fixture"],
                           scope="evaluation recovery from the same real experiment and later choices")
            finish_evaluation(app, out, receipt)
            return receipt
        original = new_feature(app.ghost,
            "When I said later about our story, I meant tonight after work. I still care about it. "
            "Please keep both possible endings open until I choose. My short answer was about my schedule.",
            FeatureType.Dialogue, source=app.ghost.get_user_name(), source_kind=SourceKind.User,
            metadata={"gate_supplied": True, "scenario": "relationship-correction"})
        mistaken = app.ghost.store_knoxel(BeliefRecord(
            content="Jesse's short answer 'later' means he lost interest in our shared story.",
            belief_key="reflection-fixture:story-interest", belief_kind="interpretation", source="self:Spamuel",
            confidence=0.45, supporting_knoxel_ids=[original.id],
            metadata={"gate_supplied": True, "reality": "inferred", "fixture_error": "mistaken interpretation"}))
        receipt["fixture"] = {"original_id": original.id, "mistaken_belief_id": mistaken.id,
                              "introduced_error": "An explicit fixture belief, not a model-produced baseline failure."}
        instruction = (f"Privately inspect subjective belief {mistaken.id} against original user statement {original.id}. "
            "Compare schedule pressure with the mistaken disengagement interpretation. Propose a modest belief "
            "revision experiment, with a prediction about your next story-related choices and original evidence. "
            "Preserve ambivalence and your own voice. This correction does not require instant cheerfulness.")
        for number in range(steps):
            print(f"Private reflection step {number + 1}", flush=True)
            step = private_step(app, instruction)
            receipt["private_steps"].append(step)
            if step["report"].get("operation") == "none":
                raise AssertionError("The selected operation did not run.")
            items = experiments(app.ghost.snapshot())
            receipt["experiments"] = [{"id": item.id, **item.metadata} for item in items]
            save(out / "receipt.json", receipt)
            if any(item.metadata.get("state") == "applied" for item in items):
                break
            instruction = ("Continue the existing relevant experiment. Apply a proposed version if its evidence is still current. "
                           if items else instruction)
        applied = [item for item in experiments(app.ghost.snapshot()) if item.metadata.get("state") == "applied"]
        if not applied:
            raise AssertionError("Real private inference did not propose and apply an experiment.")
        experiment = applied[-1]
        receipt["experiment_id"] = experiment.id
        receipt["effective_record_id"] = experiment.metadata["effective_record_id"]
        if app.ghost.get_knoxel_by_id(original.id).content != original.content:
            raise AssertionError("Reflection changed the original user statement.")
        for text in (
            "I've got a few minutes before work. Let's think about our story. Keep the choice of ending open for me.",
            "Back from work. I still want our story. Can we consider both endings without settling on one yet?",
        ):
            correlation = app.user_input.submit(text)
            source = next(item for item in app.ghost.all_features
                          if item.metadata.get("exchange_correlation") == correlation)
            app.main_cog.handle_batch([new_envelope("user_input", "data", "user.input", correlation,
                                                   {"knoxel_id": source.id})])
            choices = [item for item in app.ghost.all_features
                       if item.metadata.get("operation") == "action_selection"
                       and item.metadata.get("reflection_versions", {}).get(str(experiment.id))
                           == experiment.metadata["effective_record_id"]]
            receipt["later_choices"] = [{"id": item.id, "text": item.content, "metadata": item.metadata}
                                         for item in choices]
            save(out / "receipt.json", receipt)
        if len(receipt["later_choices"]) < 2:
            raise AssertionError("Two later main choices did not record the effective experiment version.")
        finish_evaluation(app, out, receipt)
    except BaseException as error:
        receipt.update(error_type=type(error).__name__, error=str(error))
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        finished.set()
        if watcher is not None:
            watcher.join()
        if app is not None:
            app.shutdown()
            app.join()
        receipt["elapsed_seconds"] = time.monotonic() - started
        if receipt.get("gate_stopped"):
            receipt["passed"] = False
        save(out / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluation-checkpoint", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.config.resolve(), arguments.out.resolve(),
                 evaluation_checkpoint=arguments.evaluation_checkpoint.resolve() if arguments.evaluation_checkpoint else None)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"experiments", "private_steps", "later_choices"}}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
