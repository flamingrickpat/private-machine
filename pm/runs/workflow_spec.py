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
    #: Extra request fields from the memory store and the accepted run arguments.
    #: The supervisor calls it under the store lock at each launch, after
    #: ``request_context``. The arguments hold the dispatch identity that the
    #: start pinned (for example ``_public_action_schema_id``), so a workflow
    #: can read the context of the cycle that chose the action. Launch-time
    #: context alone cannot name that cycle.
    dispatch_context: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]] | None = None
    #: Called once for each run of this workflow that a restart found
    #: interrupted, before its terminal hook. It receives the receipt and can
    #: retain a terminal report in the artifact folder. It must not repeat an
    #: external effect, and a repeated call must change nothing.
    on_interrupted: Callable[[Mapping[str, Any]], None] | None = None
    #: Re-reads retained evidence of a terminal run before a public reply. It
    #: receives the receipt and returns fields that the reply evidence adds.
    reply_evidence: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None
    #: The workflow selects an accepted memory view at each native decision group.
    memory_config_factory: Callable[[Any, Any], MemoryToolConfig] | None = None
    #: Admit one active run. A cancelled worker retains its slot through cleanup.
    singleton: bool = False
    #: Share admission across workflow names. An empty key uses the workflow name.
    singleton_key: str = ""

