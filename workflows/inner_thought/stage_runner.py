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
from pathlib import Path
from typing import TYPE_CHECKING
from enum import StrEnum
from uuid import uuid4

from pm.llm.stage_model import LazyStageModel
from pm.memory.bash_view import MemoryBashEnvironment
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus, WorkflowResult

from workflows.inner_thought.schemas import InnerRequest, TriggerChoice, ThoughtAttentionDecision, ThoughtResult, InquiryStep, IntegrationResult, NominationBatch
from workflows.reflection.schemas import ReflectionDecision

if TYPE_CHECKING:
    from workflows.inner_thought.stages.trigger.rendering import AttentionStageRequest, TriggerEnvelope
    from workflows.inner_thought.stages.thought.steps import ThoughtEnvelope
    from workflows.inner_thought.stages.integrate.steps import IntegrationEnvelope
    from workflows.inner_thought.graph_execution import GraphThoughtRequest

PACKAGE_DIR = Path(__file__).resolve().parent
PARENT_MANIFEST = PACKAGE_DIR / "inner_thought.workflow.md"
STAGES_DIR = PACKAGE_DIR / "stages"


class PrivateStage(StrEnum):
    Trigger = "trigger"
    Nominate = "nominate"
    Thought = "thought"
    Integrate = "integrate"
    Reflection = "reflection"


class PrivateStageRunner:
    def __init__(self, model: LazyStageModel, runtime_root: Path,
                 read_environment: MemoryBashEnvironment, cancel: threading.Event,
                 yield_requested: threading.Event) -> None:
        self.model = model
        self.yield_requested = yield_requested
        self.runtime_root = runtime_root
        self.read_environment = read_environment
        self.stop = cancel

    def trigger(self, request: InnerRequest) -> TriggerChoice:
        from workflows.inner_thought.stages.trigger.rendering import TriggerEnvelope
        return TriggerChoice.model_validate(self.invoke_stage(PrivateStage.Trigger, TriggerEnvelope(request=request)).report)

    def nominate(self, request: InnerRequest) -> NominationBatch:
        return NominationBatch.model_validate(self.invoke_stage(PrivateStage.Nominate, request).report)

    def attention(self, request: AttentionStageRequest) -> ThoughtAttentionDecision:
        return ThoughtAttentionDecision.model_validate(self.invoke_stage(PrivateStage.Trigger, request).report)

    def thought(self, request: ThoughtEnvelope) -> ThoughtResult:
        return ThoughtResult.model_validate(self.invoke_stage(PrivateStage.Thought, request).report)

    def inquiry(self, request: GraphThoughtRequest) -> InquiryStep:
        return InquiryStep.model_validate(self.invoke_stage(PrivateStage.Thought, request).report)

    def integrate(self, request: IntegrationEnvelope) -> IntegrationResult:
        return IntegrationResult.model_validate(self.invoke_stage(PrivateStage.Integrate, request).report)

    def reflect(self, request: InnerRequest) -> ReflectionDecision:
        return ReflectionDecision.model_validate(self.invoke_stage(PrivateStage.Reflection, request).report)

    def invoke_stage(self, stage: PrivateStage, request: InnerRequest | AttentionStageRequest | TriggerEnvelope | ThoughtEnvelope | GraphThoughtRequest | IntegrationEnvelope) -> WorkflowResult:
        from pm.runs.role_runtime import PMWorkflowRuntime
        from pm.llm.session import ModelSession
        stage_run_id = f"{stage.value}-{uuid4().hex[:12]}"
        environment = MemoryBashEnvironment(self.read_environment.visible_knoxels, self.read_environment.turns, "read")
        runtime = PMWorkflowRuntime(environment=environment, tools={}, stop_event=self.stop,
            propagate_python_errors=True, model_session=ModelSession(self.model, self.stop, yield_requested=self.yield_requested))
        manifest = STAGES_DIR / stage.value / f"{stage.value}.workflow.md"
        runner = WorkflowRunner(manifest, self.runtime_root / stage.value, namespace=stage.value, base_dir=PROJECT_ROOT)
        result = runner.invoke(WorkflowRequest(stage_run_id, request.model_dump(mode="json")), runtime)
        if result.status is WorkflowStatus.Cancelled and self.stop.is_set():
            raise asyncio.CancelledError
        if result.status is not WorkflowStatus.Completed:
            raise RuntimeError(f"stage {stage.value} did not complete: {result.kernel_summary}")
        assert result.report is not None
        return result
