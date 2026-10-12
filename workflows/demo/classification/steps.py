"""Publish observed classifier answers without another model call."""
from __future__ import annotations

import json

import yaml
from pm_workflows.python_role import RoleContext


def run(context: RoleContext) -> dict:
    """Write the latest classification and its route history into the target.

    The kernel journal is the source of truth. This trusted Python role writes
    only the demonstration report. It does not accept MemoryStore proposals or tasks.
    Cancellation prevents publication at the next cooperative boundary.
    A missing classification raises ValueError rather than inventing an answer.
    """
    context.runtime.check_cancelled()
    entries = [json.loads(line) for line in (context.kernel_data / "journal.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    # RoleContext.run_id names the role session. The journal names its workflow.
    # Select the task and item from retained classification scope instead.
    decisions = [entry for entry in entries if entry.get("kind") == "classify"
                 and entry["result"]["receipt"]["scope"]["task_id"] == context.task_id
                 and entry.get("item") == context.current_item]
    if not decisions:
        raise ValueError("The classification demonstration has no scoped answer.")
    latest = decisions[-1]
    report = {"schema": "pm.classification-demo.v1", "run_id": latest["run_id"],
              "phase": latest["phase"], "classification": latest["result"]["status"],
              "answer": {key: latest["result"][key] for key in ("status", "reason", "evidence", "candidate_id", "input_sha256")},
              "route_history": [{"phase": entry["phase"], "status": entry["status"],
                                 "invalid_code": entry["result"]["receipt"].get("invalid_code")}
                                for entry in decisions]}
    context.runtime.check_cancelled()
    context.runtime.environment.write_text("outbox/report.yaml", yaml.safe_dump(report, sort_keys=False, allow_unicode=True))
    return {"status": "done", "summary": f"Published classification {report['classification']}."}
