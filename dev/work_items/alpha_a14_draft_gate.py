"""Exercise real native MCP edits and automatic turn validation."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from threading import Event

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pm.host.folder_target import FolderEnvironment
from pm.llm.activation import ModelActivation
from pm.llm.config import load_config
from pm.llm.workflow_sessions import WorkflowModels
from pm.runs.draft_session import run_draft_session
from pm_workflows.runtime import WorkflowRuntime
from workflows.minecraft_project.planning.draft import TaskDraft
from workflows.minecraft_project.tree_store import TreeStore


def run(config_path: Path, out: Path) -> None:
    """Verify the session protocol without claiming any Minecraft world effect."""
    out = out.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Retain live inference evidence under .runtime.")
    out.mkdir(parents=True, exist_ok=False)
    environment = FolderEnvironment(out / "workspace", "bash")
    runtime = WorkflowRuntime(environment, Event(), {}, propagate_python_errors=True)
    store = TreeStore(environment.root)
    changes = []

    def save(value):
        changes.append(value)
        store.write(f"draft-changes/{len(changes):04d}.json", value)

    draft = TaskDraft(store, "Plan a five-by-five wooden floor.", save, runtime.check_cancelled)
    models = WorkflowModels(load_config(config_path), ModelActivation())
    receipt = {"scope": "Real native MCP draft operations and backend checks; no world task execution.",
               "status": "running", "started": datetime.now(timezone.utc).isoformat(), "checks": {}}
    try:
        with models.session("Default", False, runtime.stop_event, receipt_path=out / "models.jsonl") as model:
            result = run_draft_session(replace(runtime, agent_model=replace(model, enable_thinking=True)),
                run_id=out.name, servers={"project_plan": draft.server()}, validate=draft.validate,
                artifact_dir=out / "native", timeout_seconds=900,
                prompt="This is a real test of automatic turn validation. Use only the project_plan MCP tools. "
                "For your FIRST turn, add one builder leaf for a five-by-five birch-plank floor at y=10, "
                "x=0..4 and z=0..4. Use condition_set with kind=block and an inclusive box. "
                "Do not set the root goal's criteria on the first turn. End that turn with a short explanation. "
                "The backend will provide validation feedback. On the follow-up, repair the existing draft, "
                "retain the same leaf, and copy its exact condition into the root using condition_copy. "
                "Then end your turn. Do not write a JSON tree, use shell scripts, or claim world construction.")
        tree = result.value.tree
        turns = [json.loads(line) for line in (out / "native/draft-turns.jsonl").read_text(encoding="utf-8").splitlines()]
        receipt["checks"] = {"automatic_followup": result.validation_attempts >= 2,
            "same_session": len({row["session_ref"] for row in turns}) == 1,
            "first_turn_rejected": not turns[0]["accepted"], "final_turn_accepted": turns[-1]["accepted"],
            "one_retained_leaf": len([node for node in tree.nodes if node.kind == "leaf"]) == 1,
            "backend_expanded_twenty_five_cells": all(len(node.criteria[0].positions) == 25 for node in tree.nodes),
            "no_plan_committed": store.state["revision"] == 0}
        store.check_revision(tree)
        receipt["result"] = result.value.model_dump(mode="json")
        receipt["session_ref"] = result.session_ref
        receipt["validation_attempts"] = result.validation_attempts
        receipt["status"] = "passed" if all(receipt["checks"].values()) else "failed"
        if receipt["status"] != "passed":
            raise AssertionError("The real native draft session missed a protocol check.")
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["finished"] = datetime.now(timezone.utc).isoformat()
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(out / "receipt.json", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.out)
