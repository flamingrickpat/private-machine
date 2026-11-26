"""Bind the managed worker to its character's memory publication queue."""
from __future__ import annotations

from dataclasses import dataclass, replace

from pm.memory.embeddings import EmbeddingService
from pm.memory.reconsolidation.coordination import ReconsolidationCoordinator
from pm.runs.role_runtime import PMWorkflowRuntime


@dataclass(frozen=True, slots=True)
class ReconsolidationRuntimeOwner:
    coordinator: ReconsolidationCoordinator
    embeddings: EmbeddingService

    def build_runtime(self, runtime: PMWorkflowRuntime, run_id: str) -> PMWorkflowRuntime:
        """Supply live memory services to the supervisor's invocation runtime."""
        return replace(runtime, reconsolidation=self.coordinator, embedding_service=self.embeddings)
