"""Non-kernel orchestration for the main cognitive cycle.

Builds a synchronous ``llm_complete`` tool from the async client, a
``run_stage`` tool that invokes each child manifest through the shared runner,
and the parent invocation. Standalone tests and the live smoke test use this
same path.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Callable
from pm.llm.completion import TextCompletion
from pm.context_render.turns import Role, Turn
from pm.llm.prompt_encoding import TurnInput, normalize_turns
from collections.abc import Sequence
from collections.abc import Mapping
from pm.context_render import ContextKind, render_context
from uuid import uuid4

from pm_workflows.runtime import WorkflowRuntime

from pm.llm.stage_model import complete_with_retry
from pm.llm.activation import ChatClient
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.main_cognition.pipeline import STAGE_ORDER
from workflows.main_cognition.schemas import CognitiveReport, CognitiveRequest
from workflows.inference_settings import REASONING
from pm.memory.bash_view import MemoryBashEnvironment

PACKAGE_DIR = Path(__file__).resolve().parent
PARENT_MANIFEST = PACKAGE_DIR / "main_cognition.workflow.md"
STAGES_DIR = PACKAGE_DIR / "stages"


def stage_manifest(name: str) -> Path:
    if name not in STAGE_ORDER:
        raise KeyError(f"unknown cognitive stage: {name}")
    return STAGES_DIR / name / f"{name}.workflow.md"


def make_environment(ghost) -> MemoryBashEnvironment:
    view = ghost.snapshot()
    return MemoryBashEnvironment(view.serialize_knoxels(), view.serialize_gwt(), "read")


def build_llm_tool(client: ChatClient) -> TextCompletion:
    """Wrap the async client as the synchronous tool roles call."""

    def llm_complete(messages: Sequence[TurnInput], **generation: object) -> str:
        # The companion backend is a reasoning model. Reserve enough budget
        # that reasoning cannot consume the whole completion and leave empty
        # content. No generation deadline is imposed. An empty response is
        # retried at most once; it never becomes an endless reprompt loop.
        return asyncio.run(complete_with_retry(client, messages, generation))

    return llm_complete


def build_stage_runner(
    ghost,
    llm_complete: TextCompletion,
    runtime_root: Path,
    run_id: str | None = None,
    read_environment=None,
    generation_overrides: Mapping[str, Mapping[str, object]] | None = None,
    cancel: threading.Event | None = None,
) -> Callable[[str, dict], dict]:
    """Return ``run_stage(name, payload)`` bound to one run's model tool.

    ``run_id`` is optional. The live smoke test supplies one so its artifacts
    carry a readable prefix; the application omits it and each stage gets a
    unique child run ID under the shared runtime root.
    Generation overrides select per-stage options, such as enable_thinking.
    They cannot change routing or the result schema. Current stage affect
    takes precedence over the earlier public-context snapshot.
    Child calls share the invocation's cancellation signal. A failed parallel
    call stops siblings before the parent can accept their results.
    """

    stop = cancel if cancel is not None else threading.Event()

    def run_stage(name: str, payload: dict) -> dict:
        stage_run_id = f"{run_id}-{name}" if run_id else f"{name}-{uuid4().hex[:12]}"
        environment = (MemoryBashEnvironment(read_environment.visible_knoxels, read_environment.visible_gwt, "read")
                       if read_environment is not None else make_environment(ghost))
        def grounded_complete(messages: Sequence[TurnInput], **generation: object) -> str:
            overrides = dict((generation_overrides or {}).get(name, {}))
            if overrides.keys() & {"model", "messages", "stream", "response_format"}:
                raise ValueError("Stage generation overrides cannot change routing or the result contract.")
            generation.update(overrides)
            if not overrides.keys() & {"enable_thinking", "reasoning", "reasoning_budget",
                                       "reasoning_tokens", "reasoning_effort", "reasoning_budget_tokens", "chat_template_kwargs"}:
                generation["reasoning_budget"] = REASONING["off"]
            turns = normalize_turns(messages)
            boundary = payload.get("public_context")
            if boundary:
                from workflows.public_speech.public_context import PublicContext
                kinds = ContextKind.All
                if payload.get("affect"):
                    kinds &= ~ContextKind.Emotion
                selected = PublicContext.model_validate(boundary)
                stimuli = {row.get("knoxel_id"): row.get("text") for row in payload.get("stimuli", [])}
                selected = selected.model_copy(update={"evidence": [row for row in selected.evidence
                    if row.get("knoxel_id") not in stimuli or row.get("text") != stimuli[row.get("knoxel_id")]],
                    "memories": [] if selected.evidence else selected.memories})
                grounding = "Private context. Respect domain capabilities and source attribution. "
                grounding += "Private thoughts remain private. An emotion word cannot create an unimplemented state or sense.\n"
                fragment = grounding + render_context(selected, kinds)
                # Context is data. Keep it outside the instruction turn, and
                # place current stage input last so it remains authoritative.
                turns.insert(1, Turn(Role.User, fragment))
            return llm_complete(turns, cancel=stop, **generation)
        runtime = WorkflowRuntime(
            environment=environment,
            stop_event=stop,
            tools={"llm_complete": grounded_complete},
            propagate_python_errors=True,
        )
        runner = WorkflowRunner(
            stage_manifest(name),
            Path(runtime_root) / name,
            namespace=name,
            base_dir=PROJECT_ROOT,
        )
        result = runner.invoke(WorkflowRequest(stage_run_id, payload), runtime)
        if result.status is not WorkflowStatus.Completed:
            raise RuntimeError(f"stage {name} did not complete: {result.kernel_summary}")
        return result.report

    return run_stage


def run_cycle(
    ghost,
    client: ChatClient,
    request: CognitiveRequest,
    runtime_root: Path,
    run_id: str,
) -> tuple[object, CognitiveReport]:
    """Run the parent cycle with real children and a real model client."""
    llm_complete = build_llm_tool(client)
    run_stage = build_stage_runner(ghost, llm_complete, runtime_root, run_id)
    environment = make_environment(ghost)
    runtime = WorkflowRuntime(
        environment=environment,
        stop_event=threading.Event(),
        tools={"run_stage": run_stage},
        propagate_python_errors=True,
    )
    runner = WorkflowRunner(
        PARENT_MANIFEST,
        Path(runtime_root) / "parent",
        namespace="main_cognition",
        base_dir=PROJECT_ROOT,
    )
    result = runner.invoke(
        WorkflowRequest(run_id, request.model_dump(mode="json")), runtime,
    )
    if result.status is not WorkflowStatus.Completed:
        raise RuntimeError(f"main cognition did not complete: {result.kernel_summary}")
    return result, CognitiveReport.model_validate(result.report)


__all__ = [
    "PARENT_MANIFEST",
    "STAGES_DIR",
    "build_llm_tool",
    "build_stage_runner",
    "make_environment",
    "run_cycle",
    "stage_manifest",
]
