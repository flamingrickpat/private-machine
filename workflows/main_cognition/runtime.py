from __future__ import annotations

from dataclasses import dataclass
from pm_workflows.runtime import WorkflowRuntime
from pm.runs.role_runtime import PMWorkflowRuntime
from workflows.main_cognition.codelet_agent import CodeletAgent


@dataclass(kw_only=True)
class CognitionRuntime(PMWorkflowRuntime):
    codelet_agent: CodeletAgent


def cognition_runtime(runtime: WorkflowRuntime | None) -> CognitionRuntime:
    if not isinstance(runtime, CognitionRuntime):
        raise TypeError("Main cognition requires its concrete runtime and codelet owner.")
    return runtime
