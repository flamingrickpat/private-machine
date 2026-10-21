"""Keep readable cache transcripts without changing stored response bytes."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pm.llm.cache import publish
from pm.llm.config import Connection, load_config
from pm.llm.prompt_log import backfill_cache_logs, finish_prompt_log, start_prompt_log
from pm.memory.embeddings import EmbeddingCache


class CacheLogTests(unittest.TestCase):
    def test_transcript_preserves_lines_and_indents_response_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = start_prompt_log(connection=Connection(base_url="http://local/v1", model="example"),
                payload={"messages": [{"role": "user", "content": "first\n  second\n\nlast\n"}]},
                route="chat/completions", request_count=1, directory=Path(directory))
            self.assertIn("  first\n    second\n\n  last\n", path.read_text(encoding="utf-8"))
            finish_prompt_log(path, response={"choices": [{"message": {
                "content": json.dumps({"a": {"b": 1}, "long": "x" * 300})}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 20}}, elapsed=2)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("status: pending", text)
            self.assertIn('      "b": 1', text)
            self.assertIn("tokens_per_second: 10.0", text)
            self.assertTrue(path.with_suffix(".response.md").exists())
            self.assertTrue(all(len(line) <= 120 for line in text.splitlines()))

    def test_replay_records_no_fresh_generation_speed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = start_prompt_log(connection=Connection(base_url="http://local/v1", model="example"),
                payload={"messages": []}, route="chat/completions", request_count=0, directory=Path(directory))
            finish_prompt_log(path, response={"usage": {"completion_tokens": 100}}, elapsed=.001, replayed=True)
            receipt = json.loads(path.with_suffix(".response.json").read_text(encoding="utf-8"))
            self.assertTrue(receipt["cache_hit"])
            self.assertIsNone(receipt["tokens_per_second"])

    def test_backfill_preserves_exact_response_bytes(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"PM_CACHE_DIR": directory}):
            path = Path(directory) / "model" / "cache_comp_old_1.json"
            encoded = b'{"choices":[{"message":{"content":"{\\"ok\\":true}"}}]}'
            publish(path, encoded)
            backfill_cache_logs()
            self.assertEqual(path.read_bytes(), encoded)
            self.assertIn('"ok": true', path.with_suffix(".md").read_text(encoding="utf-8"))
            self.assertIn("unknown", path.with_suffix(".md").read_text(encoding="utf-8"))

    def test_old_config_cache_promotes_valid_vectors_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"PM_CACHE_DIR": str(Path(directory) / "shared")}):
            root = Path(directory)
            config_path = root / "models.yaml"
            config_path.write_text(Path("config.example.yaml").read_text(encoding="utf-8").replace(
                "embedding:\n", "embedding:\n  cache_dir: old-vectors\n"), encoding="utf-8")
            config = load_config(config_path).embedding
            old = EmbeddingCache(config.model_copy(update={"cache_dir": root / "old-vectors"}))
            vector = [1.0] * config.dimensions
            old.put("preserved input", vector)
            source = next(old.directory.glob("cache_emb_*.json"))
            encoded = source.read_bytes()
            new = EmbeddingCache(config)
            self.assertEqual(new.get("preserved input"), vector)
            self.assertEqual(source.read_bytes(), encoded)
            self.assertTrue(next(new.directory.glob("cache_emb_*.md")).is_file())
            self.assertEqual(new.directory.parent, (root / "shared" / "embeddings").resolve())
