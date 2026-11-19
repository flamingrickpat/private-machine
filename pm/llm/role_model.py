from __future__ import annotations

from pm_workflows.runtime import WorkflowRuntime
from pm.runs.role_runtime import PMWorkflowRuntime
from pm.llm.session import ModelSession
from pm.memory.embeddings import EmbeddingService


def role_model(runtime: WorkflowRuntime | None) -> ModelSession:
    if not isinstance(runtime, PMWorkflowRuntime):
        raise TypeError("PM Python roles require PMWorkflowRuntime with a concrete model session.")
    if runtime.model_session is None:
        raise RuntimeError("The workflow owner did not supply its model session.")
    return runtime.model_session


def role_embeddings(runtime: WorkflowRuntime | None) -> EmbeddingService:
    if not isinstance(runtime, PMWorkflowRuntime):
        raise TypeError("PM embedding operations require PMWorkflowRuntime.")
    if runtime.embedding_service is None:
        raise RuntimeError("The workflow owner did not supply its embedding service.")
    return runtime.embedding_service
