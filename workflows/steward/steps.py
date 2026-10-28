"""Steward role. Deterministic; no prompt and no model call.

The role reads one operation and calls the injected steward tool. A dormant
steward refuses every operation. An operation error becomes a typed rejected
report, never an invented success.
"""
from __future__ import annotations

from typing import Any

from pm_workflows.python_role import RoleContext

from pm.utils import yaml_util

from workflows.steward.schemas import StewardReport, StewardRequest


def run(context: RoleContext) -> dict[str, Any]:
    """Validate one steward operation and return its typed report."""
    environment = context.runtime.environment
    request = StewardRequest.model_validate(
        yaml_util.safe_load(environment.read_text("inbox/request.yaml"))
    )
    result = _dispatch(request, context.runtime.tools)
    report = StewardReport(
        operation=request.operation,
        accepted=bool(result.get("accepted")),
        summary=str(result.get("reason") or result.get("state") or "ok"),
        result=result,
    )
    environment.write_text(
        "scratch/report.yaml",
        yaml_util.safe_dump(report.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
    )
    environment.move("scratch/report.yaml", "outbox/report.yaml")
    return {"status": "done", "summary": report.summary}


def _dispatch(request: StewardRequest, tools: dict[str, Any]) -> dict[str, Any]:
    """Call the one steward tool that the requested operation names."""
    operation = request.operation
    if operation == "status":
        return tools["steward_status"](run_id=request.run_id)
    if operation == "answer":
        return tools["steward_answer"](
            run_id=request.run_id, question_id=request.question_id, answer=request.answer,
        )
    if operation == "update":
        return tools["steward_update"](
            run_id=request.run_id, instructions=request.instructions,
            request_id=request.request_id,
        )
    if operation == "terminate":
        return tools["steward_terminate"](run_id=request.run_id, request_id=request.request_id)
    if operation == "resume":
        return tools["steward_resume"](run_id=request.run_id, request_id=request.request_id)
    return {"accepted": False, "reason": f"Unknown steward operation {operation!r}."}


__all__ = ["run"]
