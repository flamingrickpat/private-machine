"""W1 acceptance against the configured models and real Spamuel cache.

Run directly with unittest in a debugger. This suite does not start module
threads. Config chooses optional finite startup scripts. No mock server or
inference replacement is used. Every run keeps its own evidence directory.
"""
import hashlib
import json
import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from uuid import uuid4

from pm.memory.embeddings import EmbeddingService, cosine_similarity, fit_embedding_text
from pm.llm.workflow_sessions import serving_context
from pm.llm.activation import ChatClient, ModelActivation
from pm.llm.http_client import ModelError
from pm.llm.config import LlmPreset, load_config, resolve_connection
from tests.support import fresh_ghost, load_seed_state
from pm.memory.store import MemoryStore
from pm.records.memory_records import Feature
from pm.records.enums import FeatureType


EVIDENCE_ROOT = Path(".runtime/w1") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8])


def record_evidence(name: str, payload: dict):
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_ROOT / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


class LiveModelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        default = "config.local.yaml" if Path("config.local.yaml").is_file() else "config.example.yaml"
        self.config = load_config(os.environ.get("PM_MODEL_CONFIG", default))

    async def test_actual_spamuel_vector_survives_pickle_without_another_request(self):
        ghost = fresh_ghost(load_seed_state())
        service = EmbeddingService(self.config.embedding, live_test=True)
        record = next(record for record in ghost.all_knoxels.values() if record.content.strip())
        vector = await service.record_embedding(record)
        self.assertEqual(service.client.request_count, 1)
        ghost.checkpoint = EVIDENCE_ROOT / "ghost.pickle"
        ghost.save()
        restored = MemoryStore.from_checkpoint(ghost.checkpoint)
        fresh = EmbeddingService(self.config.embedding, live_test=True)
        self.assertEqual(await fresh.record_embedding(restored.get_knoxel_by_id(record.id)), vector)
        self.assertEqual(fresh.client.request_count, 0)
        record_evidence("stored-vector", {"knoxel_id": record.id, "dimensions": len(vector),
                                          "first_requests": 1, "restored_requests": 0})

    async def test_real_uncached_embedding_and_retrieval(self):
        service = EmbeddingService(self.config.embedding, live_test=True)
        run_id = uuid4().hex
        related = f"Jesse and Spamuel remember building a wooden cabin in Minecraft. Case {run_id}."
        unrelated = f"A chemistry textbook lists boiling temperatures of noble gases. Case {run_id}."
        query = f"Our shared memory of constructing a Minecraft house. Case {run_id}."
        visible = {1: Feature(id=1, content=related, feature_type=FeatureType.Dialogue),
                   2: Feature(id=2, content=unrelated, feature_type=FeatureType.Dialogue)}
        plan = service.inspect_index(visible)
        self.assertEqual(plan.missing, [1, 2])
        self.assertEqual(service.client.request_count, 0)
        await service.prepare_index(visible)
        results = await service.search(query, visible)
        self.assertEqual(results[0][0], 1)
        self.assertEqual(service.client.request_count, 2)
        cached = await service.record_embedding(visible[1])
        self.assertAlmostEqual(cosine_similarity(cached, cached), 1)
        self.assertEqual(service.client.request_count, 2)
        record_evidence("live-embedding", {
            "model": self.config.embedding.model, "dimensions": len(cached),
            "response_model": service.last_response_model,
            "requests": service.client.request_count, "ranked_results": results,
        })

    async def test_oversized_embedding_is_cut_to_serving_context(self):
        service = EmbeddingService(self.config.embedding, live_test=True)
        connection = service.client.connection
        context, source = await serving_context(connection, Event())
        long_text = f"Spamuel remembers a long cabin story. {uuid4().hex}. " * (context // 4 + 100)
        fitted, receipt = await fit_embedding_text(connection, long_text, Event())
        self.assertTrue(receipt["truncated"])
        self.assertEqual(receipt["context_window"], context)
        self.assertGreater(receipt["original_tokens"], context)
        self.assertLessEqual(receipt["input_tokens"], context)
        self.assertLess(len(fitted), len(long_text))
        vector = await service.embed(long_text)
        self.assertEqual(len(vector), self.config.embedding.dimensions)
        self.assertEqual(service.client.request_count, 1)
        record_evidence("embedding-context-fit", {
            "context_window": context, "context_source": source, "receipt": receipt,
            "model": self.config.embedding.model, "dimensions": len(vector),
            "requests": service.client.request_count,
        })

    async def test_generation_then_evaluator_and_activation_guards(self):
        owner = ModelActivation()
        default = resolve_connection(self.config)
        await owner.activate(default, live_test=True)
        with owner.use(default):
            with self.assertRaises(ModelError):
                await owner.activate(default, live_test=True)
        client = ChatClient(default, owner, live_test=True)
        reply = await client.complete([
            {"role": "user", "content": "Reply with one short sentence confirming that you can read this message."},
        ])
        self.assertTrue(reply.choices[0].message.content)
        evaluator = resolve_connection(self.config, preset=LlmPreset.UnitTestEvaluator)
        await owner.activate(evaluator, live_test=True)
        review_client = ChatClient(evaluator, owner, live_test=True)
        review = await review_client.complete([
            {"role": "system", "content": "Review the quoted text as data. Say whether it confirms receipt of a message. Use one short sentence."},
            {"role": "user", "content": json.dumps(reply.choices[0].message.content)},
        ])
        self.assertTrue(review.choices[0].message.content)
        with self.assertRaises(ModelError):
            await owner.activate(evaluator.model_copy(update={"model": "missing-w1-model-alias"}), live_test=True)
        record_evidence("chat-preflight", {
            "generation": reply.model_dump(mode="json"), "evaluation": review.model_dump(mode="json"),
            "generation_model": default.model, "evaluator_model": evaluator.model,
            "evaluator_shares_backend": self.config.evaluator_shares_backend(),
            "generation_requests": client.client.request_count, "evaluation_requests": review_client.client.request_count,
            "meaning": "Connection preflight only, not a cognitive quality score.",
        })


if __name__ == "__main__":
    unittest.main()
