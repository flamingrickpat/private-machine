"""Prepare complete units of memory on the managed worker thread."""
from __future__ import annotations

from asyncio import CancelledError
from typing import Literal, TypedDict

from pm_workflows.python_role import RoleContext
from pydantic import BaseModel

from pm.llm.role_model import role_model
from pm.memory.reconsolidation.coordination import MemoryWorkProgress, MemoryWorkRequest
from pm.memory.snapshot import GhostCancelled
from pm.runs.role_io import read_request, write_report
from pm.runs.role_runtime import PMWorkflowRuntime


class MemoryWorkReport(BaseModel):
    schema_id: Literal["pm.memory-reconsolidation.v2"] = "pm.memory-reconsolidation.v2"
    summary: str
    progress: MemoryWorkProgress


class MemoryRoleResult(TypedDict):
    status: Literal["done"]
    summary: str


def run(context: RoleContext) -> MemoryRoleResult:
    """Read the selected mode and report preparation independently of publication."""
    runtime = context.runtime
    if not isinstance(runtime, PMWorkflowRuntime) or runtime.reconsolidation is None or runtime.embedding_service is None:
        raise TypeError("Memory reconsolidation requires its character's coordinator and embedding service.")
    request = read_request(runtime.environment, MemoryWorkRequest)
    try:
        progress = runtime.reconsolidation.prepare(request, role_model(runtime).model,
            runtime.embedding_service, runtime.stop_event)
    except (GhostCancelled, CancelledError):
        # Translate a cancelled PM operation to the external kernel's termination
        # contract only when the supervisor or a wake actually requested it.
        runtime.check_cancelled()
        raise
    runtime.check_cancelled()
    summary = (f"Prepared {progress.prepared} memory units. {progress.published} were published, "
               f"{progress.awaiting_publication} await publication, and {progress.deferred} need fresh evidence.")
    if progress.awaiting_prior_publication:
        summary = f"A previous run has {progress.awaiting_prior_publication} memory units awaiting the next publication boundary."
    write_report(runtime.environment, MemoryWorkReport(summary=summary, progress=progress))
    return MemoryRoleResult(status="done", summary=summary)
