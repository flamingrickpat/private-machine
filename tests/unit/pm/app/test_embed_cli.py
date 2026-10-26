"""Run the ``--embed`` rebuild path against a temporary checkpoint."""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pm.app.cli import launch_plan
from pm.app.embed_cli import rebuild_store_embeddings, run_embedding_rebuild, vector_sample_stats
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature
from tests.unit.pm.app.test_cli_home import SEED, rejected
from tests.unit.pm.memory.test_embedding_quality import DEGENERATE, HEALTHY, StubEmbeddingService


class EmbedFlagTests(unittest.TestCase):
    def test_home_mode_carries_the_embed_flag_and_its_checkpoint(self):
        home = Path(tempfile.gettempdir()) / "pm-embed-home"
        plan = launch_plan(["--home", str(home), "--config", "models.yaml", "--embed"], SEED)
        self.assertTrue(plan.embed)
        self.assertEqual(plan.checkpoint, home.resolve() / "ghost.pickle")
        self.assertFalse(launch_plan(["--home", str(home)], SEED).embed)

    def test_embed_needs_a_persisted_memory_and_excludes_the_inspector(self):
        for extra in (["--ephemeral"], ["--inspector"]):
            with self.subTest(extra=extra):
                self.assertTrue(rejected(["--embed", *extra]), extra)


class VectorSampleTests(unittest.TestCase):
    def store(self, vectors: list[list[float]]) -> MemoryStore:
        store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        for index, vector in enumerate(vectors):
            store.store_knoxel(Feature(content=f"memory {index}", feature_type=FeatureType.Dialogue,
                                       embedding=[float(value) for value in vector]))
        return store

    def test_degenerate_store_saturates_the_merge_threshold(self):
        stats = vector_sample_stats(self.store(DEGENERATE))
        self.assertEqual(stats["vectors"], 5)
        self.assertEqual(stats["merge_share"], 1.0)
        self.assertGreater(stats["mean"], 0.86)

    def test_healthy_store_stays_below_the_merge_threshold(self):
        stats = vector_sample_stats(self.store(HEALTHY))
        # Only the deliberate paraphrase pair of HEALTHY reaches the merge
        # threshold, so one pair of ten. A degenerate store scores every pair.
        self.assertEqual(stats["merge_share"], 0.1)
        self.assertEqual(stats["dimensions"], 8)

    def test_empty_store_reports_no_sample(self):
        stats = vector_sample_stats(MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel")))
        self.assertEqual(stats["vectors"], 0)
        self.assertEqual(stats["merge_share"], 0.0)


class RebuildTests(unittest.TestCase):
    def test_rebuild_saves_new_vectors_and_keeps_a_backup_of_the_old_ones(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "ghost.pickle"
            store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=checkpoint)
            for index, vector in enumerate(DEGENERATE):
                store.store_knoxel(Feature(content=f"memory {index}", feature_type=FeatureType.Dialogue,
                                           embedding=[float(value) for value in vector]))
            store.previous_emb_model = "broken-encoder"
            store.save()
            old_bytes = checkpoint.read_bytes()

            report = rebuild_store_embeddings(store, StubEmbeddingService(HEALTHY))

            self.assertEqual(report["model_before"], "broken-encoder")
            self.assertEqual(report["model_after"], "stub-vectors")
            self.assertEqual(report["embedded"], 5)
            self.assertEqual(report["before"]["merge_share"], 1.0)
            self.assertLessEqual(report["after"]["merge_share"], 0.1)
            self.assertTrue(checkpoint.exists())
            self.assertNotEqual(checkpoint.read_bytes(), old_bytes)
            backup = Path(report["backup"])
            self.assertTrue(backup.exists())
            self.assertEqual(backup.read_bytes(), old_bytes)
            reloaded = MemoryStore.from_checkpoint(checkpoint)
            self.assertEqual(reloaded.previous_emb_model, "stub-vectors")
            stored = {tuple(record.embedding) for record in reloaded.all_knoxels.values()}
            self.assertEqual(len(stored), 5)
            for vector in stored:
                self.assertIn(vector, [tuple(float(value) for value in row) for row in HEALTHY])


class RunRebuildTests(unittest.TestCase):
    """Cover the ``--embed`` entry point, including the lines it prints."""

    def test_run_reads_the_embedding_section_and_saves_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "ghost.pickle"
            config_path = root / "model.yaml"
            config_path.write_text("embedding: {}\n", encoding="utf-8")
            store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=checkpoint)
            for index, vector in enumerate(DEGENERATE):
                store.store_knoxel(Feature(content=f"memory {index}", feature_type=FeatureType.Dialogue,
                                           embedding=[float(value) for value in vector]))
            store.previous_emb_model = "broken-encoder"
            store.save()
            plan = launch_plan(["--checkpoint", str(checkpoint), "--config", str(config_path), "--embed"], SEED)
            config = SimpleNamespace(embedding=SimpleNamespace(base_url="http://127.0.0.1:1/v1",
                                                              model="stub-vectors", dimensions=1024))
            output = io.StringIO()
            with patch("pm.llm.config.load_config", return_value=config), \
                 patch("pm.app.embed_cli.EmbeddingService", return_value=StubEmbeddingService(HEALTHY)), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(run_embedding_rebuild(plan), 0)
            printed = output.getvalue()
            # The header reads the embedding section, not the whole model config.
            self.assertIn("'stub-vectors' at 1024 dimensions", printed)
            self.assertIn("probe:", printed)
            self.assertIn("before: 5 of 5 vectors", printed)
            self.assertIn("model 'broken-encoder' -> 'stub-vectors'", printed)
            self.assertIn("backup with the old vectors", printed)
            reloaded = MemoryStore.from_checkpoint(checkpoint)
            self.assertEqual(reloaded.previous_emb_model, "stub-vectors")
            self.assertTrue(all(record.embedding for record in reloaded.all_knoxels.values()))

    def test_run_refuses_a_missing_checkpoint_without_printing_a_traceback(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config_path = root / "model.yaml"
            config_path.write_text("embedding: {}\n", encoding="utf-8")
            plan = launch_plan(["--checkpoint", str(root / "absent.pickle"), "--config", str(config_path),
                                "--embed"], SEED)
            config = SimpleNamespace(embedding=SimpleNamespace(base_url="http://127.0.0.1:1/v1",
                                                              model="stub-vectors", dimensions=1024))
            output = io.StringIO()
            with patch("pm.llm.config.load_config", return_value=config), \
                 patch("pm.app.embed_cli.EmbeddingService", return_value=StubEmbeddingService(HEALTHY)), \
                 contextlib.redirect_stdout(output):
                self.assertEqual(run_embedding_rebuild(plan), 1)
            self.assertIn("does not exist", output.getvalue())


if __name__ == "__main__":
    unittest.main()
