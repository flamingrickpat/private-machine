from __future__ import annotations

import asyncio
from threading import Event
from pydantic import BaseModel, ConfigDict
from pm.memory.dependencies import SourceDependency, capture_source
from pm.memory.embeddings import EmbeddingService
from pm.memory.snapshot import Snapshot
from pm.memory.store import MemoryStore
from pm.memory.evidence import EvidenceItem
from pm.memory.retrieval import RetrievalTrace as RecallTrace
from workflows.reflection.schemas import ReflectionSource
from workflows.reflection.experiments import source_evidence


class PrivateRecall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    matches: tuple[EvidenceItem, ...]
    sources: tuple[SourceDependency, ...]
    reflection_sources: tuple[ReflectionSource, ...]
    trace: RecallTrace


class PrivateMemoryAccess:
    def __init__(self, store: MemoryStore, view: Snapshot, embeddings: EmbeddingService | None,
                 scopes: frozenset[tuple[int, str]], cancel: Event) -> None:
        self.store = store
        self.view = view
        self.embeddings = embeddings
        self.scopes = scopes
        self.cancel = cancel

    def recall(self, query: str) -> PrivateRecall:
        if not query.strip():
            raise ValueError("Private recall requires a nonempty query.")
        result = asyncio.run(self.store.retrieve_experiences(query, self.embeddings,
            view=self.view, scopes=set(self.scopes), limit=8, cancel=self.cancel))
        matches = result.matches
        source_ids = {key for item in matches for key in [item.knoxel_id, *item.source_ids]}
        return PrivateRecall(matches=matches, trace=result.trace,
            sources=tuple(capture_source(self.view, key, current_head=False) for key in sorted(source_ids)),
            reflection_sources=tuple(source_evidence(self.view, self.view.get(key)) for key in sorted(source_ids)))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        assert self.embeddings is not None
        return asyncio.run(self.embeddings.embed_many(texts, cancel=self.cancel))
