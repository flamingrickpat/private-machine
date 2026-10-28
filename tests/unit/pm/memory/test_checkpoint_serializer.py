"""Read legacy JSON and save typed pickle state with atomic cancellation."""
import json
from pathlib import Path
import tempfile
import unittest

from pm.memory.checkpoint import JSON_SERIALIZER, PICKLE_SERIALIZER, read_checkpoint, write_checkpoint
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature


class SerializerTests(unittest.TestCase):
    def store(self):
        store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        store.store_knoxel(Feature(content="dandelions", feature_type=FeatureType.Dialogue, embedding=[1.0, 0.5]))
        return store

    def test_pickle_is_default_and_json_remains_an_explicit_export(self):
        store = self.store()
        checkpoint = store.checkpoint_data()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ghost.pickle"
            write_checkpoint(path, checkpoint, lambda: None)
            self.assertTrue(path.read_bytes().startswith(b"\x80"))
            self.assertEqual(read_checkpoint(path).state.knoxels[0].embedding, [1.0, 0.5])
            legacy = Path(directory) / "ghost.pickle"
            write_checkpoint(legacy, checkpoint, lambda: None, serializer=JSON_SERIALIZER)
            self.assertEqual(json.loads(legacy.read_bytes())["state"]["knoxels"][0]["__type__"], "Feature")
            restored = MemoryStore.from_checkpoint(legacy, raw=True)
            self.assertEqual(restored.state_dict(), store.state_dict())
            self.assertEqual(restored.checkpoint.suffix, ".pickle")

    def test_cancellation_before_replacement_preserves_old_file(self):
        checkpoint = self.store().checkpoint_data()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ghost.pickle"
            path.write_bytes(b"previous checkpoint")
            def cancelled():
                raise RuntimeError("cancelled before replacement")
            with self.assertRaisesRegex(RuntimeError, "cancelled before replacement"):
                write_checkpoint(path, checkpoint, cancelled)
            self.assertEqual(path.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])
