"""How the supervisor runs one managed workflow: manifest, tools, agent model, and declared behavior."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

from pm_workflows.runtime import WorkflowRuntime

from pm.host.folder_target import FolderEnvironment
from pm.memory.agent_tools import MemoryToolConfig


def scoped_children(runtime: WorkflowRuntime, root: Path) -> WorkflowRuntime:
    """Give each child a folder while retaining model, tools, and cancellation.

    Nested children remain below their parent. This is runtime configuration,
    not a durable task tree or an independent body owner.
    """
    from pm_workflows.inputs import relative_path

    def child(scope: str) -> WorkflowRuntime:
        path = root / "children" / relative_path(scope)
        child_runtime = replace(runtime, environment=FolderEnvironment(path, "bash"))
        return scoped_children(child_runtime, path)
    runtime.child_factory = child
    return runtime


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
    agent_model: Any = None
    mcp_servers: Mapping[str, Mapping[str, Any]] | None = None
    #: An interrupted, cancelled, or failed run may resume with a new epoch.
    resumable: bool = False
    #: Extra request fields, read from the memory store under its lock at each launch.
    request_context: Callable[[Any], Mapping[str, Any]] | None = None
    #: The workflow selects an accepted memory view at each native decision group.
    memory_config_factory: Callable[[Any, Any], MemoryToolConfig] | None = None
    #: Admit one active run. A cancelled worker retains its slot through cleanup.
    singleton: bool = False
    #: Share admission across workflow names. An empty key uses the workflow name.
    singleton_key: str = ""

