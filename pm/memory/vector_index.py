"""An in-memory cosine index over vectors owned by the central records."""
from __future__ import annotations

from collections.abc import Iterable
from threading import RLock

import faiss
import numpy as np

from pm.records.base import RecordBase


class VectorIndex:
    """Index record IDs with FAISS. The lock protects native reads and writes."""

    def __init__(self) -> None:
        self.lock = RLock()
        self.index: faiss.IndexIDMap2 | None = None
        self.ids: set[int] = set()

    def rebuild(self, records: Iterable[RecordBase]) -> None:
        """Build one index from loaded records without changing their vectors."""
        records = [record for record in records if record.embedding]
        with self.lock:
            self.index = None
            self.ids.clear()
            if records:
                vectors = np.asarray([record.embedding for record in records], dtype=np.float32)
                faiss.normalize_L2(vectors)
                self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
                self.index.add_with_ids(vectors, np.asarray([record.id for record in records], dtype=np.int64))
                self.ids.update(record.id for record in records)

    def update(self, record: RecordBase) -> None:
        """Replace one indexed vector. An empty vector removes the record ID."""
        with self.lock:
            self.remove(record.id)
            if record.embedding:
                vector = np.asarray([record.embedding], dtype=np.float32)
                faiss.normalize_L2(vector)
                if self.index is None:
                    self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(vector.shape[1]))
                self.index.add_with_ids(vector, np.asarray([record.id], dtype=np.int64))
                self.ids.add(record.id)

    def remove(self, record_id: int) -> None:
        """Remove a superseded query head without changing its stored vector."""
        with self.lock:
            if record_id in self.ids:
                self.index.remove_ids(np.asarray([record_id], dtype=np.int64))
                self.ids.remove(record_id)
                if not self.ids:
                    self.index = None

    def search(self, vector: list[float], visible_ids: Iterable[int], *, limit: int | None = None) -> list[tuple[int, float]]:
        """Rank visible IDs by cosine similarity. Hidden IDs never enter results."""
        visible = set(visible_ids)
        with self.lock:
            if self.index is None or not self.ids or not visible:
                return []
            query = np.asarray([vector], dtype=np.float32)
            faiss.normalize_L2(query)
            scores, ids = self.index.search(query, self.index.ntotal)
            results = [(int(key), float(score)) for key, score in zip(ids[0], scores[0]) if key in visible]
        results.sort(key=lambda item: (-item[1], item[0]))
        return results if limit is None else results[:limit]
