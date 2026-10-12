"""Verify A05 classification, PM activation, and compatibility with real services."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from importlib import metadata
import json
from pathlib import Path
import socket
import sys
from threading import Event
from types import SimpleNamespace

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bus.envelopes import Envelope
from main import build_app
from pm_workflows.classification import replay_classification
from pm_workflows.dryrun import check_workflow
from pm_workflows.inputs import digest
from pm_workflows.runtime import WorkflowRuntime
from runtime.model_activation import ModelActivation
from runtime.models import load_config
from runtime.workflow_models import WorkflowModels
from runtime.workflow_runner import WorkflowRequest, WorkflowRunner, WorkflowStatus
from dev.work_items.alpha_classification_gate import inspect_overflow, long_log, make_logs, response_recorder, save, sha
from working_target import FolderEnvironment, GhostBashWrapper

DEMO = ROOT / "workflows/classification_demo/classification_demo.workflow.md"
LONG_DEMO = DEMO.with_name("long_inspection.workflow.md")


def entries(path: Path) -> list[dict]:
    """Read the journal without changing its retained evidence."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def variant(out: Path, name: str, *, binding: dict, limits: dict | None = None) -> Path:
    """Change only a demo input or budget for an explicit boundary case."""
    document = yaml.safe_load(DEMO.read_text(encoding="utf-8").split("---", 2)[1])
    phase = document["phases"][0]
    del phase["input_folder"], phase["data_selection"]
    phase.update(binding)
    if limits:
        phase["classification_limits"] = limits
    path = out / f"{name}.workflow.md"
    path.write_text("---\n" + yaml.safe_dump(document, sort_keys=False) + "---\n", encoding="utf-8")
    return path


def run_case(out: Path, name: str, environment, models, manifest=DEMO, payload=None):
    """Use PM's synchronous runner and retain report, service, and replay data."""
    dryrun = check_workflow(manifest, ROOT, seed=5, monte_carlo_runs=100)
    save(out / f"{name}-dryrun.json", dryrun.to_dict())
    assert not dryrun.errors, dryrun.errors
    runtime = WorkflowRuntime(environment, Event(), {}, propagate_python_errors=True)
    result = WorkflowRunner(manifest, out / "runtime", models=models).invoke(
        WorkflowRequest(name, payload or {"artifact_folder": "logs"}), runtime,
    )
    save(out / f"{name}-result.json", asdict(result) | {"artifact_dir": str(result.artifact_dir)})
    assert result.status is WorkflowStatus.Completed, result.kernel_summary
    classified = [entry for entry in entries(result.artifact_dir / "journal.jsonl") if entry.get("kind") == "classify"]
    for entry in classified:
        replay_classification(entry["result"]["receipt"])
    assert result.report["answer"] == {key: classified[-1]["result"][key] for key in ("status", "reason", "evidence", "candidate_id", "input_sha256")}
    print(f"{name}: {result.report['classification']}", flush=True)
    return result, classified


def stale_producer(out: Path, environment, models):
    """Invalidate a real producer attempt before consuming its named result."""
    path = variant(out, "stale-producer", binding={"output": "test_report"})
    document = yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])
    role = out / "producer.py"
    role.write_text(
        "def run(context):\n"
        "    if context.attempt == 1:\n"
        "        return {'status': 'done', 'report': context.runtime.environment.read_text('logs/current.log')}\n"
        "    return {'status': 'done'}\n", encoding="utf-8")
    document["roles"]["producer"] = {"skill": str(role), "result_contract": {"schema": {"status": {"enum": ["done"]}}}}
    document["phases"].insert(0, {"name": "produce", "kind": "role", "role": "producer",
        "on_status": {"done": "produce"}, "on_invalid": {"action": "route_to", "target": "check"},
        "outputs": {"test_report": {"pointer": "/report", "expected_type": "string"}}})
    path.write_text("---\n" + yaml.safe_dump(document, sort_keys=False) + "---\n", encoding="utf-8")
    return run_case(out, "stale-producer", environment, models, path)


def decision_absent() -> bool:
    """Check the actual optional port without sending a decision request."""
    try:
        with socket.create_connection(("127.0.0.1", 8082), timeout=1):
            return False
    except OSError:
        return True


def main() -> None:
    """Run M1 with the real seed and fresh environments.

    The gate changes no deployment configuration and never starts Laya.
    Stop its optional service before delivery acceptance. Development flags
    retain incomplete receipts. Every source and response stays under .runtime.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.alpha.example.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-overflow", action="store_true")
    parser.add_argument("--allow-laya-present", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Keep complete original-history evidence under .runtime.")
    out.mkdir(parents=True, exist_ok=False)
    seed = ROOT / "test_data/personas/spamuel/canon.yaml"
    seed_before = sha(seed)
    receipt = {"passed": False, "seed_sha256": seed_before, "cases": [], "laya_wiring": "deferred",
               "laya_requests": 0, "human_review": "not rated", "C01_C02": "remain open"}
    try:
        absent = decision_absent()
        receipt["laya_absent_before"] = absent
        assert absent or args.allow_laya_present, "Stop optional Laya before M1 acceptance."
        replies = []
        app = build_app(seed, out / "application", cognition="real", config_path=str(args.config),
                        checkpoint_path=out / "ghost-checkpoint.json", emit_reply=replies.append)
        models = app.workflow_models
        models.live_test = True
        assert models.owner._ready == {}, "Construction contacted chat."
        assert all(module.workflow_models is models for module in app.modules if hasattr(module, "workflow_models"))
        assert app.supervisor.workflow_models is models
        virtual = GhostBashWrapper(app.ghost).open("a05")
        host = FolderEnvironment(out / "host", "bash")
        expected_path, sources = make_logs(out, host, virtual)
        config = load_config(args.config)
        connection = models.connections["Default"]
        origin = connection.base_url.removesuffix("/v1")
        with httpx.Client(timeout=120) as client:
            model_response = client.get(connection.base_url + "/models")
            model_response.raise_for_status()
            props_response = client.get(origin + "/props")
            props_response.raise_for_status()
            served, props = model_response.json(), props_response.json()
        context = props["default_generation_settings"]["n_ctx"]
        save(out / "identity.json", {"models": served, "build": props["build_info"], "context_window": context,
            "seed_sha256": seed_before, "records": len(app.ghost.snapshot().serialize_knoxels()),
            "demo_sources": {str(path.relative_to(ROOT)): sha(path) for path in (DEMO, LONG_DEMO, DEMO.with_name("inspect.md"), DEMO.with_name("inspect_long.md"), DEMO.with_name("steps.py"))},
            "origins": {name: metadata.distribution(name).read_text("direct_url.json") for name in ("pm-workflows", "pm-coder")}})
        original_state = app.ghost.snapshot().serialize_knoxels()
        for name, environment in (("host", host), ("ghost", virtual)):
            result, classified = run_case(out, name, environment, models)
            capture = next(iter(classified[0]["result"]["receipt"]["captures"].values()))
            assert result.report["classification"] == "all_good"
            assert capture["file"]["path"] == expected_path
            assert (capture["file"]["modified_time"] is None) == (name == "ghost")
            receipt["cases"].append({"name": name, "class": "all_good", "digest": result.report["answer"]["input_sha256"]})
        changed_path = ROOT / ".runtime/alpha/a04/foundation-validated-delivery.log"
        changed_bytes = changed_path.read_bytes()
        changed_encoding = "utf-16" if changed_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        raw = changed_bytes.decode(changed_encoding)
        changed_source = {"source": str(changed_path), "source_sha256": sha(changed_path), "encoding": changed_encoding}
        save(out / "changed-source.json", changed_source)
        before = digest(virtual.read_text(expected_path))
        virtual.write_text(expected_path, raw)
        result, _ = run_case(out, "changed-ghost", virtual, models)
        assert result.report["classification"] == "all_good" and result.report["answer"]["input_sha256"] == digest(raw) != before
        receipt["cases"].append({"name": "changed-ghost", "class": "all_good", "source_sha256": changed_source["source_sha256"], "digest": digest(raw)})
        virtual.write_text(expected_path, host.read_text(expected_path))
        virtual.write_text("logs/current.log", host.read_text(expected_path))
        for name, binding, limits, code, expected in (
            ("missing", {"input_file": "logs/missing.log"}, None, "missing_file", "INVALID"),
            ("oversized", {"input_file": expected_path}, {"max_input_tokens": 512}, "oversized_input", "all_good"),
            ("ambiguous", {"input_folder": "logs", "data_selection": "Select the foundation log produced by task a05-unrecorded. Require actual producer metadata. If no producer is recorded, return INVALID."}, None, "uncertain_selection", "INVALID"),
        ):
            result, classified = run_case(out, name, virtual, models, variant(out, name, binding=binding, limits=limits))
            assert classified[0]["result"]["receipt"]["invalid_code"] == code
            assert result.report["classification"] == expected
            assert classified[-1]["result"]["receipt"].get("session_ref"), "Real agent recovery did not run."
            receipt["cases"].append({"name": name, "class": expected, "primary_cause": code})
        result, classified = stale_producer(out, virtual, models)
        assert result.report["classification"] == "INVALID"
        assert classified[0]["result"]["receipt"]["invalid_code"] == "stale_output"
        receipt["cases"].append({"name": "stale-producer", "class": "INVALID", "primary_cause": "stale_output"})
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            unavailable = deepcopy(config)
            unavailable.chat["Default"] = unavailable.chat["Default"].model_copy(update={"base_url": f"http://127.0.0.1:{reserved.getsockname()[1]}/v1", "startup_script": None})
            result, classified = run_case(out, "unavailable", virtual, WorkflowModels(unavailable, ModelActivation(), live_test=True),
                variant(out, "unavailable", binding={"input_file": expected_path}))
        assert result.report["classification"] == "INVALID"
        assert classified[0]["result"]["receipt"]["invalid_code"] == "endpoint_unavailable"
        receipt["cases"].append({"name": "unavailable", "class": "INVALID", "primary_cause": "endpoint_unavailable"})
        # These source facts define expected classes independently of inference.
        for name, text, expected in (
            ("synthetic-mixed", "collected 4 items\ntests/test_capture.py ..FF\n2 failed, 2 passed in 0.42s\n", "some_failed"),
            ("synthetic-failed", "collected 3 items\ntests/test_capture.py FFF\n3 failed in 0.42s\n", "all_failed"),
            ("synthetic-contradictory", "Current summary: collected 3 tests; 3 passed.\nCurrent summary: collected 3 tests; 3 failed.\nNo timestamp distinguishes these summaries.\n", "INVALID"),
        ):
            relative = f"logs/{name}.log"
            virtual.write_text(relative, text)
            result, _ = run_case(out, name, virtual, models, variant(out, name, binding={"input_file": relative}))
            assert result.report["classification"] == expected
            receipt["cases"].append({"name": name, "class": expected, "source": "synthetic boundary input with actual model inference", "digest": digest(text)})
        if not args.skip_overflow:
            original = yaml.load(seed.read_text(encoding="utf-8"), Loader=yaml.CSafeLoader)
            history = "\n".join(entry["data"]["content"] for entry in original["knoxels"] if entry["data"].get("content"))
            with httpx.Client(timeout=120) as client:
                text, tokens = long_log(client, origin, history, host.read_text(expected_path), context)
            virtual.write_text("logs/overflow.log", text)
            save(out / "overflow-input.json", {"tokens": tokens, "bytes": len(text.encode("utf-8")), "context_window": context, "seed_sha256": seed_before})
            # The relay captures real provider failures without changing requests.
            # The explicit context belongs to the verified upstream service.
            with models.session("Default", False, Event(), receipt_path=out / "relay-activation.jsonl") as model:
                with response_recorder(model, out / "overflow-http") as recorded_model:
                    relayed = deepcopy(config)
                    relayed.chat["Default"] = relayed.chat["Default"].model_copy(update={"base_url": recorded_model.base_url, "context_window": context, "startup_script": None})
                    relay_models = WorkflowModels(relayed, ModelActivation(), live_test=True)
                    result, classified = run_case(out, "overflow", virtual, relay_models, LONG_DEMO)
            assert result.report["classification"] == "all_good"
            class RetainedJournal:
                def entries_for_phase(self, name):
                    return [entry for entry in classified if entry["phase"] == name]
            overflow = inspect_overflow(SimpleNamespace(kernel_data=result.artifact_dir, journal=RetainedJournal()), digest(text), out / "overflow-http")
            save(out / "overflow-receipt.json", overflow)
            receipt["overflow"] = {key: value for key, value in overflow.items() if key != "checkpoints"}
        assert app.ghost.snapshot().serialize_knoxels() == original_state, "Classification mutated Ghost."
        # Run the current Story module with its original tools and model path.
        envelope = Envelope("a05-story", "user", "data", "action.request", "a05-story", {
            "content": "Reply briefly to Jesse about inspecting a test report.",
            "stimuli": [{"text": "We are checking the test reports. Say one short thing about that."}],
        }, target="dialogue")
        app.dialogue.mailbox.put(envelope)
        # The first Story call must reuse this already leased alias. A second
        # activation would fail while classification owns the same service.
        with models.session("Default", False, Event(), receipt_path=out / "compatibility-lease.jsonl"):
            assert app.dialogue.step(), "The existing Story module did not consume its action."
        result = app.dialogue.last_result
        assert result.status is WorkflowStatus.Completed and result.report["accepted"] and result.report["candidate"].strip()
        assert not (result.artifact_dir / "model-sessions.jsonl").exists(), "Story used classification activation."
        save(out / "compatibility-story.json", {"status": result.status, "report": result.report, "artifact_dir": str(result.artifact_dir),
             "connection": connection.model, "driver": "python", "classification_sessions": 0,
             "human_immersion": "not rated", "laya_absent": decision_absent(), "shared_active_lease_reused": True})
        receipt["compatibility"] = {"status": "completed", "model": connection.model, "path": "Story Python roles and existing lazy chat tool", "laya_absent": decision_absent()}
        assert decision_absent() or args.allow_laya_present
        assert models.owner._active_calls == 0 and not models.owner._switching
        assert sha(seed) == seed_before
        receipt["seed_unchanged"] = True
        receipt["passed"] = not args.skip_overflow and not args.allow_laya_present
        print(f"A05 acceptance passed={receipt['passed']}", flush=True)
    finally:
        save(out / "receipt.json", receipt)


if __name__ == "__main__":
    main()
