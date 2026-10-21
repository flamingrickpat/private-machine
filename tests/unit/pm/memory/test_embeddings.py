"""Exercise real FAISS indexes and vectors stored on typed memory records."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import unittest

from pm.llm.config import EmbeddingConfig
from pm.llm.http_client import ModelError
from pm.memory.embeddings import EmbeddingService, cosine_similarity, parse_embedding_response
from pm.memory.store import MemoryStore
from pm.memory.vector_index import VectorIndex
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature


class EmbeddingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = EmbeddingConfig(base_url="http://127.0.0.1:1/v1", model="stored-vectors", dimensions=2)
        self.service = EmbeddingService(self.config, live_test=True)
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))

    def record(self, content, vector):
        return self.store.store_knoxel(Feature(content=content, feature_type=FeatureType.Dialogue, embedding=vector))

    def test_cosine_uses_both_norms(self):
        self.assertAlmostEqual(cosine_similarity([3, 4], [30, 40]), 1)
        self.assertAlmostEqual(cosine_similarity([3, 4], [-4, 3]), 0)
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [1, 0])

    def test_provider_alias_and_invalid_vectors(self):
        config = self.config.model_copy(update={"response_model": "provider-alias"})
        response = {"model": "provider-alias", "data": [{"index": 0, "embedding": [1.0, 0.5]}]}
        self.assertEqual(parse_embedding_response(response, config), [1.0, 0.5])
        for invalid in ({**response, "model": "wrong"}, {**response, "data": [{"index": 0, "embedding": [0.0, 0.0]}]},
                        {**response, "data": [{"index": 0, "embedding": [1.0]}]}):
            with self.assertRaises(ModelError):
                parse_embedding_response(invalid, config)

    async def test_record_vectors_are_reused_and_search_respects_membership(self):
        first = self.record("known query", [3.0, 4.0])
        second = self.record("orthogonal", [-4.0, 3.0])
        hidden = self.record("hidden", [3.0, 4.0])
        records = {first.id: first, second.id: second}
        plan = await self.service.prepare_index(records)
        self.assertEqual(plan.missing, [])
        self.assertIs(await self.service.record_embedding(first), first.embedding)
        ranked = await self.service.search(first.content, records, index=self.store.embedding_index)
        self.assertEqual([key for key, _ in ranked], [first.id, second.id])
        self.assertNotIn(hidden.id, dict(ranked))
        self.assertAlmostEqual(ranked[0][1], 1.0, places=6)
        self.assertEqual(self.service.client.request_count, 0)

    def test_assignment_and_explicit_replacement_update_faiss(self):
        record = self.record("one", [1.0, 0.0])
        record.embedding = [0.0, 1.0]
        self.assertAlmostEqual(self.store.embedding_index.search([0, 1], [record.id])[0][1], 1)
        record.content = "changed text"
        self.assertEqual(record.embedding, [])
        self.assertEqual(self.store.embedding_index.search([0, 1], [record.id]), [])
        batch = self.store.begin_batch("edit")
        edited = record.copy_for_edit()
        edited.embedding = [1.0, 0.0]
        batch.replace(edited)
        self.store.commit(batch)
        self.assertIs(self.store.get_knoxel_by_id(record.id), edited)
        self.assertIs(edited._owner, self.store)
        self.assertAlmostEqual(self.store.embedding_index.search([1, 0], [record.id])[0][1], 1)

    def test_pickle_restart_retains_every_vector_and_rebuilds_index(self):
        record = self.record("durable", [1.0, 0.5])
        with tempfile.TemporaryDirectory() as directory:
            self.store.checkpoint = Path(directory) / "ghost.pickle"
            self.store.save()
            restored = MemoryStore.from_checkpoint(self.store.checkpoint)
            loaded = restored.get_knoxel_by_id(record.id)
            self.assertEqual(loaded.embedding, record.embedding)
            self.assertIs(restored.snapshot().get(record.id), loaded)
            self.assertIs(loaded._owner, restored)
            self.assertAlmostEqual(restored.embedding_index.search(record.embedding, [record.id])[0][1], 1, places=6)

    def test_native_index_reads_and_writes_share_a_lock(self):
        index = VectorIndex()
        records = [Feature(id=key, content=str(key), feature_type=FeatureType.Dialogue, embedding=[1.0, float(key)])
                   for key in range(40)]
        index.rebuild(records)
        def change(record):
            index.update(record)
            return len(index.search([1, 0], range(40)))
        with ThreadPoolExecutor(max_workers=4) as workers:
            self.assertEqual(list(workers.map(change, records)), [40] * 40)

    def test_empty_index_accepts_the_next_models_dimension(self):
        record = self.record("changing model", [1.0, 0.0])
        record.embedding = []
        record.embedding = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(self.store.embedding_index.search(record.embedding, [record.id])[0][1], 1.0)

    async def test_unavailable_provider_keeps_the_record_unembedded(self):
        record = self.record("needs inference", [])
        with self.assertRaises(ModelError):
            await self.service.prepare_index({record.id: record})
        self.assertEqual(record.embedding, [])
        self.assertEqual(self.service.inspect_index({record.id: record}).missing, [record.id])
