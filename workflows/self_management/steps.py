"""SelfManagement role.

Entry point: ``run(context)``. The role reads one typed operation and calls the
injected managed-run tools. It contains no prompts and no model calls. A
refused operation becomes a typed rejected report, never an invented success.
"""
from __future__ import annotations

from pm.utils import yaml_util

from pm.runs.run_receipt import ManagedRunError
from workflows.self_management.schemas import SelfManagementReport, SelfManagementRequest


def run(context) -> dict:
    environment = context.runtime.environment
    request = SelfManagementRequest.model_validate(
        yaml_util.safe_load(environment.read_text("inbox/request.yaml"))
    )
    tools = context.runtime.tools
    try:
        report = _dispatch(request, tools)
    except ManagedRunError as error:
        report = SelfManagementReport(
            operation=request.operation,
            accepted=False,
            summary=str(error),
        )
    environment.write_text(
        "scratch/report.yaml",
        yaml_util.safe_dump(report.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
    )
    environment.move("scratch/report.yaml", "outbox/report.yaml")
    return {
        "status": "done",
        "summary": report.summary,
    }


def _dispatch(request: SelfManagementRequest, tools) -> SelfManagementReport:
    operation = request.operation
    if operation == "resume":
        receipt = tools["managed_resume"](run_id=request.run_id, request_id=request.request_id or request.action_id)
        return SelfManagementReport(operation=operation, accepted=True,
                                    summary=f"run {receipt['run_id']} recovery accepted", receipt=receipt)
    if operation == "start":
        receipt = tools["managed_start"](
            action_id=request.action_id,
            workflow_name=request.workflow_name,
            instructions=request.instructions,
            parent_correlation=request.parent_correlation,
            arguments=request.arguments,
        )
        return SelfManagementReport(
            operation=operation,
            accepted=receipt["state"] != "failed",
            summary=f"run {receipt['run_id']} {receipt['state']}",
            receipt=receipt,
        )
    if operation == "status":
        receipt = tools["managed_status"](run_id=request.run_id)
        return SelfManagementReport(
            operation=operation, accepted=True,
            summary=f"run {receipt['run_id']} is {receipt['state']}",
            receipt=receipt,
        )
    if operation == "list":
        receipts = tools["managed_list"]()
        return SelfManagementReport(
            operation=operation, accepted=True,
            summary=f"{len(receipts)} managed runs",
            receipts=receipts,
        )
    if operation == "update":
        receipt = tools["managed_update"](
            run_id=request.run_id, instructions=request.instructions,
            request_id=request.request_id or request.action_id,
        )
        return SelfManagementReport(
            operation=operation, accepted=True,
            summary=f"run {receipt['run_id']} at revision {receipt['revision']}",
            receipt=receipt,
        )
    if operation == "terminate":
        receipt = tools["managed_terminate"](run_id=request.run_id, request_id=request.request_id or request.action_id)
        return SelfManagementReport(
            operation=operation, accepted=True,
            summary=f"run {receipt['run_id']} is {receipt['state']}",
            receipt=receipt,
        )
    if operation == "answer":
        receipt = tools["managed_answer"](
            run_id=request.run_id, question_id=request.question_id, answer=request.answer,
        )
        return SelfManagementReport(
            operation=operation, accepted=True,
            summary=f"answered run {receipt['run_id']}",
            receipt=receipt,
        )
    raise ManagedRunError(f"Unknown self-management operation {operation!r}.")


__all__ = ["run"]
