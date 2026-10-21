"""Restore stable backups after later memory commits."""
from pathlib import Path
import tempfile
import unittest

from pm.memory.checkpoint import read_checkpoint
from pm.memory.store import MemoryStore
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.enums import FeatureType
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import new_feature


class BackupTests(unittest.TestCase):
    def test_backup_restores_accepted_state_and_retains_each_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ghost.json"
            store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=path)
            feature = new_feature(store, "accepted before cognition", FeatureType.Dialogue, source="Jesse")
            batch = store.begin_batch("cycle", new_tick=True)
            batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, stimulus_ids=[feature.id],
                                       knoxel_ids=[feature.id], summary="accepted"), CycleState(tick_id=batch.tick_id))
            store.commit(batch)
            before = read_checkpoint(path)
            first, second = store.backup(), store.backup()
            self.assertNotEqual(first, second)
            self.assertEqual(read_checkpoint(first), before)
            self.assertEqual(read_checkpoint(second), before)
            new_feature(store, "after the snapshot", FeatureType.Dialogue, source="Jesse")
            restored = MemoryStore.from_checkpoint(first)
            self.assertEqual(restored.state_dict(), before.state)
            self.assertEqual(restored.revisions, before.revisions)
            self.assertEqual(restored.gwt_turns, before.gwt_turns)
            self.assertEqual(restored.applied_commits, before.applied_commits)
            self.assertEqual(len(list(Path(directory).glob("ghost_*.json"))), 2)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_ephemeral_memory_writes_no_backup(self):
        self.assertIsNone(MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel")).backup())
