from __future__ import annotations

from dataclasses import dataclass
from pm_workflows.runtime import WorkflowRuntime
from pm.llm.session import ModelSession
from pm.memory.embeddings import EmbeddingService
from pm.runs.managed_boundary import ManagedRunBoundary
from pm.llm.mcp_config import ToolCallConfig
from pm.memory.store import MemoryStore
from pm.runs.memory_context import ManagedMemoryContext
from pm.host.cognition_profiler import CognitionProfiler


@dataclass
class PMWorkflowRuntime(WorkflowRuntime):
    profiler: CognitionProfiler | None = None
    workflow_span_id: str = ""
    phase_span_id: str = ""
    model_session: ModelSession | None = None
    embedding_service: EmbeddingService | None = None
    managed_boundary: ManagedRunBoundary | None = None
    tool_call_config: ToolCallConfig | None = None
    memory_store: MemoryStore | None = None
    memory_context: ManagedMemoryContext | None = None
