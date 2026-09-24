"""Real cache files and scoped retrieval inputs. No fake model endpoint."""
import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from pm_core.embeddings import EmbeddingCache, EmbeddingService, cosine_similarity, parse_embedding_response
from runtime.model_io import ModelError
from runtime.models import EmbeddingConfig


class EmbeddingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.legacy = self.root / "legacy"
        self.legacy.mkdir()
        self.config = EmbeddingConfig(
            base_url="http://127.0.0.1:1/v1", model="perplexity/pplx-embed-v1-0.6b",
            api_key="unused-marker", dimensions=1024,
            legacy_cache_dir=self.legacy, cache_dir=self.root / "new",
        )
        self.service = EmbeddingService(self.config, live_test=True)

    def write_legacy(self, text="Spamuel remembers Jesse.", **overrides):
        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        vector = [3.0, 4.0] + [0.0] * 1022
        meta = {"input": text, "model": self.config.model, "dimensions": 1024, "emb_hash": key}
        meta.update(overrides)
        binary = self.legacy / f"emb_hash_{key}.bin"
        binary.write_bytes(struct.pack("<Q1024d", 1024, *vector))
        (self.legacy / f"emb_hash_{key}.json").write_text(json.dumps(meta), encoding="utf-8")
        return text, vector, binary

    async def test_cache_hit_preserves_raw_text_and_sends_zero_requests(self):
        text, vector, binary = self.write_legacy("  Spamuel remembers Jesse.\n")
        before = binary.read_bytes()
        self.assertEqual(await self.service.embed(text), vector)
        self.assertEqual(self.service.client.request_count, 0)
        self.assertEqual(binary.read_bytes(), before)
        self.assertIsNone(self.service.cache.get(text.strip()))
        self.assertFalse(self.config.cache_dir.exists())

    def test_model_input_and_dimension_mismatch_are_misses(self):
        for values in ({"model": "wrong"}, {"input": "different"}, {"dimensions": 3}):
            with self.subTest(values=values):
                text, _, _ = self.write_legacy(**values)
                self.assertIsNone(self.service.cache.get(text))

    def test_corrupt_and_nonfinite_binary_are_misses(self):
        text, _, binary = self.write_legacy()
        binary.write_bytes(b"broken")
        self.assertIsNone(self.service.cache.get(text))
        binary.write_bytes(struct.pack("<Q1024d", 1024, math.nan, *([0.0] * 1023)))
        self.assertIsNone(self.service.cache.get(text))

    def test_new_cache_round_trip_never_changes_legacy(self):
        text, vector, binary = self.write_legacy()
        before = binary.read_bytes()
        self.service.cache.put(text + " New memory.", vector)
        reopened = EmbeddingCache(self.config)
        self.assertEqual(reopened.get(text + " New memory."), vector)
        self.assertEqual(binary.read_bytes(), before)
        self.assertEqual(len(list(self.legacy.glob("*.bin"))), 1)

    def test_cache_cannot_write_inside_legacy_directory(self):
        with self.assertRaises(ValueError):
            EmbeddingCache(self.config.model_copy(update={"cache_dir": self.legacy / "new"}))

    def test_cosine_uses_both_norms(self):
        self.assertAlmostEqual(cosine_similarity([3, 4], [30, 40]), 1)
        self.assertAlmostEqual(cosine_similarity([3, 4], [-4, 3]), 0)
        with self.assertRaises(ValueError):
            cosine_similarity([0, 0], [1, 0])

    def test_provider_alias_is_explicit_and_wrong_vectors_fail(self):
        config = self.config.model_copy(update={"response_model": "pplx-embed-v1-0.6b"})
        vector = [1.0] + [0.0] * 1023
        response = {"model": "pplx-embed-v1-0.6b", "data": [{"index": 0, "embedding": vector}]}
        self.assertEqual(parse_embedding_response(response, config), vector)
        with self.assertRaises(ModelError):
            parse_embedding_response(response, self.config)
        with self.assertRaises(ModelError):
            parse_embedding_response({**response, "model": "unrelated-model"}, config)
        with self.assertRaises(ModelError):
            parse_embedding_response({**response, "data": [{"index": 0, "embedding": [1.0]}]}, config)

    async def test_explicit_index_plan_and_scoped_search(self):
        text, vector, _ = self.write_legacy()
        self.service.cache.put("related query", vector)
        self.service.cache.put("unrelated", [4.0, -3.0] + [0.0] * 1022)
        self.service.cache.put("hidden memory", vector)
        visible = {1: text, 2: "unrelated"}
        plan = self.service.inspect_index({**visible, 3: "new memory"})
        self.assertEqual(plan.hits, [1, 2])
        self.assertEqual(plan.missing, [3])
        results = await self.service.search("related query", visible)
        self.assertEqual([record_id for record_id, _ in results], [1, 2])
        self.assertEqual(self.service.client.request_count, 0)
        with self.assertRaises(ValueError):
            await self.service.search("related query", {3: "new memory"})


if __name__ == "__main__":
    unittest.main()
