"""The ``run_stage`` tool for the private thought loop.

Each private stage runs through the shared kernel as a child workflow of the
parent manifest. The environment is a frozen read-only snapshot of the
MemoryStore, exactly as in the main cognitive loop. Private stages never
receive the public grounding block: their material stays private until the
host publishes an accepted self-origin event.
"""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from pm_workflows.runtime import WorkflowRuntime
from pm.memory.bash_view import MemoryBashEnvironment
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus

from workflows.main_cognition.stage_runner import make_environment
from workflows.inner_thought.pipeline import STAGE_ORDER
from workflows.inference_settings import REASONING

PACKAGE_DIR = Path(__file__).resolve().parent
PARENT_MANIFEST = PACKAGE_DIR / "inner_thought.workflow.md"
STAGES_DIR = PACKAGE_DIR / "stages"


def stage_manifest(name: str) -> Path:
    if name not in STAGE_ORDER:
        raise KeyError(f"unknown private thought stage: {name}")
    return STAGES_DIR / name / f"{name}.workflow.md"


def build_inner_stage_runner(ghost, llm_complete: Callable, runtime_root: Path,
                             read_environment=None, *, cancel: threading.Event | None = None) -> Callable[[str, dict], dict]:
    """Bind private child stages to one model and cancellation signal.

    Requested child cancellation raises asyncio.CancelledError so the parent
    can discard its proposal. Unexpected termination still raises RuntimeError.
    Kernel artifacts remain under runtime_root; cancellation returns no report.
    """

    stop = cancel if cancel is not None else threading.Event()

    def run_stage(name: str, payload: dict) -> dict:
        stage_run_id = f"{name}-{uuid4().hex[:12]}"
        # The invocation's frozen view wins when one exists, so a private
        # stage cannot read records the main loop accepted meanwhile.
        environment = (MemoryBashEnvironment(read_environment.visible_knoxels, read_environment.visible_gwt, "read")
                       if read_environment is not None else make_environment(ghost))
        def complete(messages, **generation):
            profile = payload["operation"]["group"] if name == "thought" else "off"
            return llm_complete(messages, cancel=stop, reasoning_budget=REASONING[profile], **generation)

        runtime = WorkflowRuntime(
            environment=environment,
            stop_event=stop,
            tools={"llm_complete": complete},
            propagate_python_errors=True,
        )
        runner = WorkflowRunner(stage_manifest(name), Path(runtime_root) / name,
                                namespace=name, base_dir=PROJECT_ROOT)
        result = runner.invoke(WorkflowRequest(stage_run_id, payload), runtime)
        # A requested stop is cancellation, not a failed private stage.
        if result.status is WorkflowStatus.Cancelled and stop.is_set():
            raise asyncio.CancelledError
        if result.status is not WorkflowStatus.Completed:
            raise RuntimeError(f"stage {name} did not complete: {result.kernel_summary}")
        return result.report

    return run_stage


__all__ = ["PARENT_MANIFEST", "build_inner_stage_runner", "stage_manifest"]
