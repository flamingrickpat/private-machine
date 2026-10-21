"""Verify exact checkpoint bytes and real atomic-file cancellation."""
import json
from pathlib import Path
import tempfile
import unittest

from pm.memory.checkpoint import Checkpoint, CheckpointSerializer, JSON_SERIALIZER, read_checkpoint, write_checkpoint


class SerializerTests(unittest.TestCase):
    def test_json_bytes_keep_existing_format_and_schema(self):
        checkpoint = Checkpoint(state={"text": "dandelions 🌼"}, current_knoxel_id=0,
            current_tick_id=0, reserved_tick_id=0, revisions={}, gwt_turns={}, applied_commits={})
        previous = json.dumps(checkpoint.model_dump(mode="json"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.assertIsInstance(JSON_SERIALIZER, CheckpointSerializer)
        self.assertEqual(JSON_SERIALIZER.dumps(checkpoint), previous)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ghost.json"
            write_checkpoint(path, checkpoint, lambda: None)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(read_checkpoint(path), checkpoint)

    def test_cancellation_before_replacement_preserves_old_file(self):
        checkpoint = Checkpoint(state={}, current_knoxel_id=0, current_tick_id=0,
            reserved_tick_id=0, revisions={}, gwt_turns={}, applied_commits={})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ghost.json"
            path.write_bytes(b"previous checkpoint")

            def cancelled():
                raise RuntimeError("cancelled before replacement")

            with self.assertRaisesRegex(RuntimeError, "cancelled before replacement"):
                write_checkpoint(path, checkpoint, cancelled)
            self.assertEqual(path.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
