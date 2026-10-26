"""Check the embedding quality probe against known-good and known-bad vector sets.

The probe is the guard that stopped a degenerate encoder from filling a live
memory store on 2026-10-24. These tests keep both directions: a healthy set
must pass, and a degenerate or a meaningless set must stop the run before any
vector is written.
"""
from __future__ import annotations

import asyncio
import unittest

from pm.llm.config import EmbeddingConfig
from pm.memory.embeddings import (EmbeddingQualityError, EmbeddingService, check_embedding_quality,
                                  embedding_quality_report)
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature

# A working encoder: sentences 0 and 1 overlap strongly, the rest share little.
HEALTHY = [
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0.1, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 0, 1, 1, 1],
]
# The 2026-10-24 failure: one strong component in every vector, with small
# noise on top. Pairs land near 0.99, so all memories look alike.
DEGENERATE = [[1, 1, 0.02 * index, 0.01, 0, 0, 0, 0] for index in range(5)]
# No meaning at all: five orthogonal directions.
MEANINGLESS = [[1 if column == row else 0 for column in range(5)] for row in range(5)]


class QualityReportTests(unittest.TestCase):
    def test_healthy_vectors_pass_and_keep_the_paraphrase_apart(self):
        report = check_embedding_quality(HEALTHY, model="healthy-test", dimensions=8)
        self.assertLess(report["unrelated_mean"], 0.6)
        self.assertGreater(report["margin"], 0.4)

    def test_report_scales_with_direction_not_length(self):
        scaled = [[value * 40 for value in vector] for vector in HEALTHY]
        for key, value in embedding_quality_report(scaled).items():
            self.assertAlmostEqual(value, embedding_quality_report(HEALTHY)[key], places=9)

    def test_degenerate_vectors_are_refused_with_numbers(self):
        with self.assertRaises(EmbeddingQualityError) as caught:
            check_embedding_quality(DEGENERATE, model="broken-test", dimensions=8)
        message = str(caught.exception)
        self.assertIn("broken-test", message)
        self.assertIn("unrelated sentences average", message)
        self.assertIn("one shared direction dominates", message)

    def test_orthogonal_vectors_carry_no_meaning(self):
        with self.assertRaises(EmbeddingQualityError) as caught:
            check_embedding_quality(MEANINGLESS, model="random-test", dimensions=5)
        self.assertIn("carry no meaning", str(caught.exception))

    def test_wrong_probe_size_is_a_programming_error(self):
        with self.assertRaises(ValueError):
            embedding_quality_report([vector for vector in HEALTHY[:4]])


class StubEmbeddingService(EmbeddingService):
    """An embedding service that returns a fixed set instead of calling a server."""

    def __init__(self, vectors: list[list[float]]):
        super().__init__(EmbeddingConfig(base_url="http://127.0.0.1:1/v1", model="stub-vectors", dimensions=8))
        self.vectors = [[float(value) for value in vector] for vector in vectors]
        self.prepared = 0

    async def embed_many(self, texts, *, cancel=None):
        return [list(vector) for vector in self.vectors]

    async def prepare_index(self, records, *, cancel=None, progress=None):
        """Fill missing vectors in input order, as the real service does."""
        missing = [key for key, record in records.items() if not record.embedding]
        self.prepared = len(missing)
        for offset, key in enumerate(missing):
            records[key].embedding = list(self.vectors[offset % len(self.vectors)])
        if progress is not None:
            progress(len(missing), len(missing))
        return None


class ServiceProbeTests(unittest.TestCase):
    def test_smoke_test_reports_the_endpoint(self):
        service = StubEmbeddingService(HEALTHY)
        report = asyncio.run(service.smoke_test())
        self.assertGreater(report["paraphrase"], report["unrelated_mean"])
        self.assertIsNone(service.last_response_model)

    def test_smoke_test_refuses_a_degenerate_endpoint(self):
        with self.assertRaises(EmbeddingQualityError):
            asyncio.run(StubEmbeddingService(DEGENERATE).smoke_test())


class StoreGuardTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        for index in range(3):
            self.store.store_knoxel(Feature(content=f"memory {index}", feature_type=FeatureType.Dialogue,
                                            embedding=[float(value) for value in HEALTHY[index]]))

    def test_store_writes_nothing_when_the_probe_fails(self):
        service = StubEmbeddingService(DEGENERATE)
        with self.assertRaises(EmbeddingQualityError):
            self.store.check_embeddings(service)
        self.assertEqual(service.prepared, 0)

    def test_rebuild_replaces_every_vector_and_records_the_model(self):
        service = StubEmbeddingService(HEALTHY)
        embedded = self.store.rebuild_embeddings(service)
        self.assertEqual(embedded, 3)
        self.assertEqual(service.prepared, 3)
        self.assertEqual(self.store.previous_emb_model, "stub-vectors")
        stored = {tuple(record.embedding) for record in self.store.all_knoxels.values()}
        self.assertEqual(len(stored), 3)
        for vector in stored:
            self.assertIn(vector, [tuple(row) for row in HEALTHY])

    def test_initialize_embeddings_stops_before_touching_records(self):
        service = StubEmbeddingService(DEGENERATE)
        before = {key: list(record.embedding) for key, record in self.store.all_knoxels.items()}
        with self.assertRaises(EmbeddingQualityError):
            self.store.initialize_embeddings(service)
        self.assertEqual({key: list(record.embedding) for key, record in self.store.all_knoxels.items()}, before)
        self.assertEqual(service.prepared, 0)


if __name__ == "__main__":
    unittest.main()
