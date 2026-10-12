"""Public dialogue inspection and scoped semantic search through Ghost."""
import asyncio
from pathlib import Path
import tempfile
import unittest

from pm.memory.store import Ghost
from pm.memory.embeddings import EmbeddingService
from pm.records.cycle_state import GhostConfig
from pm.records.memory_records import Feature
from pm.records.enums import FeatureType, InterlocusType
from pm.llm.config import EmbeddingConfig


class GhostInspectionTests(unittest.TestCase):
    def setUp(self):
        self.ghost = Ghost(GhostConfig(user_name="Jesse", companion_name="Spamuel"))
        self.jesse = self.ghost.store_knoxel(Feature(content="hello", feature_type=FeatureType.Dialogue,
                                                  interlocus=InterlocusType.Public, override_source="Jesse"))
        self.spamuel = self.ghost.store_knoxel(Feature(content="world", feature_type=FeatureType.Dialogue,
                                                    interlocus=InterlocusType.Public, override_source="Spamuel"))

    def test_history_reads_accepted_memory_and_respects_scope(self):
        rows = self.ghost.dialogue_history(user="Jesse")
        self.assertEqual([row["speaker"] for row in rows], ["Jesse", "Spamuel"])
        self.assertEqual(len(self.ghost.dialogue_history(visible_ids=[self.jesse.id])), 1)
        self.assertEqual(self.ghost.dialogue_history(visible_ids=[]), [])

    def test_search_uses_real_cache_and_reports_incomplete_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            embeddings = EmbeddingService(EmbeddingConfig(base_url="http://127.0.0.1:1/v1", model="fixture",
                dimensions=2, legacy_cache_dir=Path(directory)/"legacy", cache_dir=Path(directory)/"cache"))
            embeddings.cache.put("hello", [1., 0.])
            embeddings.cache.put("query", [1., 0.])
            result = asyncio.run(self.ghost.search_memory("query", embeddings))
            self.assertEqual(result["indexed"], 1)
            self.assertEqual(result["missing"], 1)
            self.assertEqual(result["matches"][0]["id"], self.jesse.id)
            hidden = asyncio.run(self.ghost.search_memory("query", embeddings, visible_ids=[self.spamuel.id]))
            self.assertEqual(hidden["matches"], [])
            filtered = asyncio.run(self.ghost.search_memory("query", embeddings,
                record_type="Feature", subtype="feature_type: Dialogue"))
            self.assertEqual(filtered["total"], 2)
            excluded = asyncio.run(self.ghost.search_memory("query", embeddings,
                record_type="Action"))
            self.assertEqual(excluded["total"], 0)
            excluded = asyncio.run(self.ghost.search_memory("query", embeddings,
                subtype="feature_type: WorldEvent", prepare=True))
            self.assertEqual(excluded["matches"], [])
            self.assertEqual(embeddings.client.request_count, 0)
