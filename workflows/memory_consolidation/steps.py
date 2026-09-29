"""Memory consolidation role. Deterministic; no model call.

It reads a stable task-artifact inventory and commits evidence-backed memory
through the injected tool. Repeated requests reuse the recorded receipt.
"""
from __future__ import annotations

import yaml

from workflows.memory_consolidation.schemas import ConsolidationReport, ConsolidationRequest


def run(context) -> dict:
    environment = context.runtime.environment
    request = ConsolidationRequest.model_validate(
        yaml.safe_load(environment.read_text("inbox/request.yaml"))
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
        yaml.safe_dump(report.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
    )
    environment.move("scratch/report.yaml", "outbox/report.yaml")
    return {"status": "done", "summary": report.summary}


__all__ = ["run"]
