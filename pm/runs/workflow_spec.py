"""How the supervisor runs one managed workflow: manifest, tools, agent model, and declared behavior."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, TYPE_CHECKING
from pydantic import BaseModel, JsonValue
from pm.llm.mcp_config import ToolCallConfig
from pm.runs.role_runtime import PMWorkflowRuntime
from pm_workflows.working_target import WorkingEnvironment
from pm.runs.memory_context import MemoryConfigOwner

if TYPE_CHECKING:
    from pm.memory.store import MemoryStore

from pm_workflows.runtime import WorkflowRuntime

from pm.host.folder_target import FolderEnvironment
from pm.memory.agent_tools import MemoryToolConfig


@dataclass(frozen=True, slots=True)
class ChildRuntimeOwner:
    runtime: WorkflowRuntime
    root: Path

    def child(self, scope: str) -> WorkflowRuntime:
        from pm_workflows.inputs import relative_path
        path = self.root / "children" / relative_path(scope)
        child = replace(self.runtime, environment=FolderEnvironment(path, "bash"))
        return scoped_children(child, path)


def scoped_children(runtime: WorkflowRuntime, root: Path) -> WorkflowRuntime:
    runtime.child_factory = ChildRuntimeOwner(runtime, root).child
    return runtime


class DispatchContextOwner(Protocol):
    def capture(self, store: MemoryStore, arguments: Mapping[str, JsonValue]) -> BaseModel: ...


class WorkflowRuntimeOwner(Protocol):
    def build_runtime(self, runtime: PMWorkflowRuntime, run_id: str) -> PMWorkflowRuntime: ...


class RequestContextOwner(Protocol):
    def capture(self, store: MemoryStore) -> BaseModel: ...


@dataclass(frozen=True)
class WorkflowSpec:
    """One workflow with explicit agent, MCP, and child runtime configuration.

    Children retain cancellation and tool ownership. Each child gets its own
    directory. Trusted Python roles keep their existing MCP configuration.
    """

    manifest: Path
    environment_factory: Callable[[str], Any] | None = None
    namespace: str = ""
    tools_factory: Callable[[str, Any], Mapping[str, Callable]] | None = None
    runtime_owner: WorkflowRuntimeOwner | None = None
    request_context_owner: RequestContextOwner | None = None
    memory_config_owner: MemoryConfigOwner | None = None
    agent_model: Any = None
    mcp_servers: Mapping[str, Mapping[str, Any]] | None = None
    #: An interrupted, cancelled, or failed run may resume with a new epoch.
    resumable: bool = False
    #: Extra request fields from the memory store and the accepted run arguments.
    #: The supervisor calls it under the store lock at each launch, after
    #: ``request_context``. The arguments hold the dispatch identity that the
    #: start pinned (for example ``_public_action_schema_id``), so a workflow
    #: can read the context of the cycle that chose the action. Launch-time
    #: context alone cannot name that cycle.
    dispatch_context_owner: DispatchContextOwner | None = None
    tool_call_config: ToolCallConfig | None = None
    #: Called once for each run of this workflow that a restart found
    #: interrupted, before its terminal hook. It receives the receipt and can
    #: retain a terminal report in the artifact folder. It must not repeat an
    #: external effect, and a repeated call must change nothing.
    on_interrupted: Callable[[Mapping[str, Any]], None] | None = None
    #: Admit one active run. A cancelled worker retains its slot through cleanup.
    singleton: bool = False
    #: Share admission across workflow names. An empty key uses the workflow name.
    singleton_key: str = ""
    #: Allow main cognition to accept an explicit partner request during ordinary interaction.
    routine_start: bool = False
    #: Describe the tasks this routine workflow can start from original partner text alone.
    direct_request: str = ""

