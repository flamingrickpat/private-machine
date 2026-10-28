"""Memory consolidation role. Deterministic; no model call.

It reads a stable task-artifact inventory and commits evidence-backed memory
through the injected tool. Repeated requests reuse the recorded receipt.
"""
from __future__ import annotations

from typing import Any

from pm_workflows.python_role import RoleContext

from pm.utils import yaml_util

from workflows.memory_consolidation.schemas import ConsolidationReport, ConsolidationRequest


def run(context: RoleContext) -> dict[str, Any]:
    """List consolidation receipts, or commit one consolidation, as asked."""
    environment = context.runtime.environment
    request = ConsolidationRequest.model_validate(
        yaml_util.safe_load(environment.read_text("inbox/request.yaml"))
    )
    tools = context.runtime.tools
    if request.operation == "list":
        receipts = tools["list_consolidations"]()
        report = ConsolidationReport(
            operation="list", accepted=True,
            summary=f"{len(receipts)} consolidations", receipts=receipts,
        )
    else:
        receipt = tools["consolidated_artifacts"](
            run_id=request.run_id,
            artifact_path=request.artifact_path,
            source_event_ids=request.source_event_ids,
        )
        report = ConsolidationReport(
            operation="consolidate", accepted=True,
            summary=f"consolidated {len(receipt['memory_ids'])} memories",
            receipt=receipt,
        )
    environment.write_text(
        "scratch/report.yaml",
        yaml_util.safe_dump(report.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
    )
    environment.move("scratch/report.yaml", "outbox/report.yaml")
    return {"status": "done", "summary": report.summary}


__all__ = ["run"]
