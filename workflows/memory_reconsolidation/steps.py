"""Role of the memory reconsolidation workflow. It makes no model call itself.

The ``reconsolidate_memory`` tool runs ``pm.memory.reconsolidation`` on the
live store with the run's stop event. The role writes the report to
``outbox/report.yaml``. Cancellation stops the reconsolidation at its next
model call or commit; the groups committed before stay accepted.
"""
from __future__ import annotations

from typing import Any

from pm_workflows.python_role import RoleContext

from pm.memory.reconsolidation import ReconsolidationReport, reconsolidate
from pm.memory.reconsolidation.pipeline import ReconsolidationProgress
from pm.runs.role_runtime import PMWorkflowRuntime
from pm.llm.role_model import role_model
from pm.utils import yaml_util


def run(context: RoleContext) -> dict[str, Any]:
    """Run one reconsolidation and report what it created."""
    runtime = context.runtime
    environment = runtime.environment
    lines: list[str] = []
    if not isinstance(runtime, PMWorkflowRuntime) or runtime.memory_store is None or runtime.embedding_service is None:
        raise TypeError("Reconsolidation requires its live memory store, embeddings, and model runtime.")
    report = reconsolidate(runtime.memory_store, complete=role_model(runtime).model,
        embeddings=runtime.embedding_service, force=True, cancel=runtime.stop_event,
        progress=ReconsolidationProgress(enabled=True))
    runtime.check_cancelled()
    summary = (f"Sorted through memory: {len(report.topical_ids)} topical and {len(report.temporal_ids)} temporal "
               f"summaries, {len(report.fact_ids)} facts, {len(report.narrative_ids)} narrative versions.")
    environment.write_text("scratch/progress.log", "\n".join(lines))
    environment.write_text("scratch/report.yaml", yaml_util.safe_dump(
        {"schema": "pm.memory-reconsolidation.v1", "summary": summary, **report.model_dump(mode="json")},
        sort_keys=False, allow_unicode=True))
    environment.move("scratch/report.yaml", "outbox/report.yaml")
    return {"status": "done", "summary": summary}


__all__ = ["run"]
