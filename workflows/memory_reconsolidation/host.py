"""Tool factory of the memory reconsolidation workflow.

The tool closes over the live memory store, the application's stage model,
and its embedding service. The supervisor calls the factory once for each
run. The tool itself runs the whole reconsolidation in the run's worker
thread; its commits use the normal store path.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pm.runs.role_runtime import PMWorkflowRuntime
from threading import Event
from typing import TYPE_CHECKING
from pm.memory.reconsolidation import reconsolidate, ReconsolidationReport
from pm.memory.reconsolidation.pipeline import ReconsolidationProgress

if TYPE_CHECKING:
    from pm.llm.stage_model import LazyStageModel
    from pm.memory.embeddings import EmbeddingService
    from pm.memory.store import MemoryStore


@dataclass(frozen=True, slots=True)
class ReconsolidationToolsOwner:
    store: MemoryStore
    model: LazyStageModel | None
    embeddings: EmbeddingService | None

    def build_runtime(self, runtime: PMWorkflowRuntime, run_id: str) -> PMWorkflowRuntime:
        return replace(runtime, embedding_service=self.embeddings)

    def reconsolidate_memory(self, cancel: Event | None = None) -> ReconsolidationReport:
        if self.model is None or self.embeddings is None:
            raise ValueError("Memory reconsolidation needs the real stage model and embeddings.")
        return reconsolidate(self.store, complete=self.model, embeddings=self.embeddings, force=True,
            cancel=cancel, progress=ReconsolidationProgress(enabled=True))


def build_reconsolidation_tools(store: MemoryStore, stage_model: LazyStageModel | None,
                                embeddings: EmbeddingService | None) -> ReconsolidationToolsOwner:
    return ReconsolidationToolsOwner(store, stage_model, embeddings)
