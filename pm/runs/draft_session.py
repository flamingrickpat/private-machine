"""Continue one native agent session until its workflow accepts the draft.

A workflow that needs a native agent keeps one conversation open: the agent
writes a draft, the workflow validates it, and a rejection returns repair
feedback into the same conversation. Tools, draft state, and the eventual
commit stay with the workflow; this module only owns the turn loop.

Each turn runs against a real MCP descriptor file, so the agent sees exactly
the scoped services of this run. Only a validated draft ends the loop, and a
cancelled run stops inference and closes the listeners before it returns.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Callable, Generic, Mapping, TypeVar

from fastmcp import FastMCP
from pm_workflows.runtime import WorkflowRuntime
from pm_workflows.working_target import AgentTarget

from pm.host.mcp_listeners import serve_services
from pm.llm.prompt_log import log_directory
from pm.memory.agent_tools import MemoryReadSession
from pm.memory.snapshot import GhostConflict

T = TypeVar("T")


@dataclass(frozen=True)
class DraftValidation(Generic[T]):
    """A workflow-owned turn decision and its precise repair feedback."""

    accepted: bool
    value: T | None = None
    feedback: str = ""


@dataclass(frozen=True)
class DraftSessionResult(Generic[T]):
    """The validated value and the native conversation that produced it."""

    value: T | None
    session_ref: str
    validation_attempts: int


def run_draft_session(
    runtime: WorkflowRuntime,
    *,
    run_id: str,
    prompt: str,
    servers: Mapping[str, FastMCP],
    validate: Callable[[str], DraftValidation[T]],
    artifact_dir: Path,
    timeout_seconds: float = 7200,
    enable_write: bool = False,
    memory: MemoryReadSession | None = None,
) -> DraftSessionResult[T]:
    """Run scoped MCP tools and validate each completed native turn.

    The workflow owns tools, draft state, validation, and its eventual commit.
    Rejection adds feedback to the same conversation. It does not reset tools
    or replay edits. Native inference retains its normal retry and compaction.

    Optional ``memory`` supplies a stable workflow grant on another listener.
    The caller owns that read session and closes it after this call. Inspected
    source conflicts request repair in the same conversation. Validators must
    return decisions without commits. The owner guards sources again at commit.

    The caller must supply an activated model connection. Configuration and
    provider adaptation remain in ``pm.llm``. Cancellation stops inference and
    awaits cleanup before listeners close. Synchronous validators must respect
    cancellation beside their own effects. Logic errors reach the caller.
    """
    runtime.check_cancelled()
    if runtime.agent_model is None:
        raise ValueError("A draft session requires an activated model connection.")
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = runtime.environment.agent_target(artifact_dir)
    scoped_servers = dict(servers)
    if memory is not None:
        if "memory" in scoped_servers:
            raise ValueError("The memory service name is reserved for scoped read tools.")
        scoped_servers["memory"] = memory.server()
    original_validate = validate

    def validate_with_memory(response: str) -> DraftValidation[T]:
        """Reject a draft whose inspected memory sources changed before commit."""
        decision = original_validate(response)
        if decision.accepted and memory is not None:
            try:
                with memory.guard():
                    pass
            except GhostConflict as error:
                return DraftValidation(False, feedback=str(error))
        return decision

    with serve_services(scoped_servers) as descriptors:
        config = artifact_dir / "draft-mcp.json"
        config.write_text(json.dumps({"mcpServers": descriptors}, indent=2), encoding="utf-8")
        return asyncio.run(_run(runtime, target, run_id, prompt, validate_with_memory, config,
                                artifact_dir, timeout_seconds, enable_write))


async def _run(runtime: WorkflowRuntime, target: AgentTarget, run_id: str, prompt: str,
               validate: Callable[[str], DraftValidation[T]], config: Path, artifact_dir: Path,
               timeout_seconds: float, enable_write: bool) -> DraftSessionResult[T]:
    """Keep the entered agent and its MCP connections through repair turns."""
    from pm_coder import build_agent, build_settings, discover_workspace, SessionStore, run_turn

    connection = runtime.agent_model
    settings = build_settings(cwd=target.cwd, base_url=connection.base_url,
        api_key=connection.api_key, model=connection.model, context_window=connection.context_window,
        enable_thinking=connection.enable_thinking, live_test=connection.live_test,
        enable_write=enable_write, workspace_discovery=False, mcp_config=config)
    discovery = discover_workspace(settings)
    session = SessionStore.open(settings.cwd, run_id, log_root=log_directory() / "agents")
    session.context_window = settings.context_window
    agent = build_agent(settings, discovery, session, bash_machine=target.bash_machine,
                        bash_machine_user=target.user, with_subagent_tool=False, capture_stream=True)
    deadline, attempts = time.monotonic() + timeout_seconds, 0
    async with agent:
        while True:
            runtime.check_cancelled()
            if time.monotonic() >= deadline:
                raise TimeoutError("The native draft session exceeded its observation deadline.")
            task = asyncio.create_task(run_turn(agent, settings, session, prompt))
            try:
                while not task.done():
                    runtime.check_cancelled()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("The native draft session exceeded its observation deadline.")
                    await asyncio.wait({task}, timeout=0.05)
                turn = task.result()
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            runtime.check_cancelled()
            decision = validate(turn.response)
            runtime.check_cancelled()
            attempts += 1
            with (artifact_dir / "draft-turns.jsonl").open("a", encoding="utf-8") as journal:
                journal.write(json.dumps({"attempt": attempts, "session_ref": session.run_id,
                    "response": turn.response, "accepted": decision.accepted,
                    "feedback": decision.feedback, "duration_seconds": turn.duration_seconds,
                    "tokens_used": turn.tokens_used}, ensure_ascii=False) + "\n")
            if decision.accepted:
                return DraftSessionResult(decision.value, session.run_id, attempts)
            if not decision.feedback:
                raise ValueError("A rejected draft needs workflow validation feedback.")
            prompt = "The backend did not accept the draft. Retain your edits and repair these errors:\n" + decision.feedback


__all__ = ["DraftValidation", "DraftSessionResult", "run_draft_session"]
