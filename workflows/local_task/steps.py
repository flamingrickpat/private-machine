"""Local managed worker role.

One small real workflow used to exercise concurrent execution, status,
instruction updates, blocking questions, and termination. It writes real files
into its task folder and never calls a model. The same role serves the normal
and question manifests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pm_workflows.python_role import RoleContext

from pm.utils import yaml_util

WORKFLOW_DIR = Path(__file__).resolve().parent


def run(context: RoleContext) -> dict[str, Any]:
    """Run the one local worker step and write its report and result files."""
    environment = context.runtime.environment
    request = yaml_util.safe_load(environment.read_text("inbox/request.yaml"))
    instruction = str(request.get("instruction", "")).strip() or "(no instruction)"
    answer = str(context.answer or request.get("answer", "")).strip()
    boundary = context.runtime.tools.get("managed_instructions")
    if boundary is not None:
        instruction = boundary()["instruction"] or instruction

    environment.write_text("scratch/started.md", f"# Started\n\n{instruction}\n")
    environment.write_text(
        "scratch/observations.md",
        f"# Observations\n\ninstruction: {instruction}\n"
        f"answer: {answer or '(none)'}\n",
    )
    if request.get("wait_for_cancel"):
        if boundary is None:
            context.runtime.stop_event.wait()
            context.runtime.check_cancelled()
        while True:
            revision = boundary(wait=True)
            instruction = revision["instruction"]
            environment.write_text(
                "scratch/observations.md",
                f"# Observations\n\ninstruction: {instruction}\nrevision: {revision['revision']}\n",
            )
    if boundary is not None:
        instruction = boundary(wait=bool(request.get("wait_for_update")), close=True)["instruction"] or instruction
        environment.write_text(
            "scratch/observations.md",
            f"# Observations\n\ninstruction: {instruction}\nanswer: {answer or '(none)'}\n",
        )
    if request.get("fail"):
        raise RuntimeError("the local task was asked to fail on purpose")

    summary = f"local task finished: {instruction}"
    environment.write_text("scratch/result.md", f"# Result\n\n{summary}\n")
    environment.write_text(
        "outbox/report.yaml",
        yaml_util.safe_dump({"summary": summary, "instruction": instruction, "answer": answer}),
    )
    return {"status": "done", "summary": summary}


__all__ = ["run"]
