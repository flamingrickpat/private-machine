from __future__ import annotations

from dataclasses import dataclass
from pm_workflows.runtime import WorkflowRuntime
from pm.runs.role_runtime import PMWorkflowRuntime
from workflows.dialogue_assistant.schemas import McpEndpoint


@dataclass(kw_only=True)
class AssistantRuntime(PMWorkflowRuntime):
    mcp_endpoints: tuple[McpEndpoint, ...]


def assistant_runtime(runtime: WorkflowRuntime | None) -> AssistantRuntime:
    if not isinstance(runtime, AssistantRuntime):
        raise TypeError("Assistant dialogue requires its concrete runtime with scoped MCP endpoints.")
    return runtime
