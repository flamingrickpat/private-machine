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

from pm_workflows.runtime import WorkflowRuntime

from runtime.model_activation import ChatClient
from runtime.model_io import ModelError
from runtime.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.main_cognition.pipeline import STAGE_ORDER
from workflows.main_cognition.schemas import CognitiveReport, CognitiveRequest
from working_target import GhostBashEnvironment

PACKAGE_DIR = Path(__file__).resolve().parent
PARENT_MANIFEST = PACKAGE_DIR / "main_cognition.workflow.md"
STAGES_DIR = PACKAGE_DIR / "stages"


def stage_manifest(name: str) -> Path:
    if name not in STAGE_ORDER:
        raise KeyError(f"unknown cognitive stage: {name}")
    return STAGES_DIR / name / f"{name}.workflow.md"


def make_environment(ghost) -> GhostBashEnvironment:
    view = ghost.snapshot()
    return GhostBashEnvironment(view.serialize_knoxels(), view.serialize_gwt(), "read")


def build_llm_tool(client: ChatClient) -> Callable[..., str]:
    """Wrap the async client as the synchronous tool roles call."""

    def llm_complete(messages: list[dict], **generation) -> str:
        # The companion backend is a reasoning model. Reserve enough budget
        # that reasoning cannot consume the whole completion and leave empty
        # content. No generation deadline is imposed. An empty response is
        # retried at most once; it never becomes an endless reprompt loop.
        generation.setdefault("max_tokens", 8192)
        last: ModelError | None = None
        for attempt in range(2):
            try:
                result = asyncio.run(client.complete(messages, **generation))
            except ModelError as error:
                if "no content" not in str(error) or attempt == 1:
                    raise
                last = error
                continue
            return result.choices[0].message.content or ""
        raise last if last is not None else ModelError("The stage model returned no content.")

    return llm_complete


def build_stage_runner(
    ghost,
    llm_complete: Callable[..., str],
    runtime_root: Path,
    run_id: str,
) -> Callable[[str, dict], dict]:
    """Return ``run_stage(name, payload)`` bound to one run's model tool."""

    def run_stage(name: str, payload: dict) -> dict:
        environment = make_environment(ghost)
        runtime = WorkflowRuntime(
            environment=environment,
            stop_event=threading.Event(),
            tools={"llm_complete": llm_complete},
            propagate_python_errors=True,
        )
        runner = WorkflowRunner(
            stage_manifest(name),
            Path(runtime_root) / name,
            namespace=name,
            base_dir=PROJECT_ROOT,
        )
        result = runner.invoke(WorkflowRequest(f"{run_id}-{name}", payload), runtime)
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
