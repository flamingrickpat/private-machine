"""Checkpoint restart and save boundaries against real files and MemoryStore."""
from pathlib import Path
import tempfile
import unittest
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.memory_records import Entity, Feature, ActionSchemaRecord
from pm.records.enums import ActionType, EntityClass, FeatureType

class RestartTests(unittest.TestCase):
    def test_checkpoint_loses_unsaved_messages_and_does_not_replay_saved_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ghost.pickle'
            store = MemoryStore(PersonaConfig(companion_name='Spamuel', user_name='Jesse'), checkpoint=path)
            owner = store.store_knoxel(Entity(content='Spamuel', aliases=[], entity_class=EntityClass.Self))
            store.accept_import()
            original = path.read_bytes()
            batch = store.begin_batch('pending-input', embed=False)
            message = batch.create(Feature(content='unsaved question', feature_type=FeatureType.Dialogue,
                metadata={'exchange_status':'pending','exchange_correlation':'old-input'}))
            batch.preserve_on_recovery()
            store.commit(batch)
            self.assertEqual(path.read_bytes(), original)
            self.assertNotIn(message.id, MemoryStore.from_checkpoint(path).all_knoxels)
            batch = store.begin_batch('unfinished-reply', embed=False)
            batch.depend_on([message.id], authority=owner.id)
            reply = batch.create(Feature(content='unfinished answer', feature_type=FeatureType.Dialogue,
                metadata={'publication_status':'pending'}))
            action = batch.create(ActionSchemaRecord(content='reply later', action_type=ActionType.Reply, metadata={'dispatch_status':'pending'}))
            store.commit(batch)
            store.save()
            saved = path.read_bytes()
            reopened = MemoryStore.from_checkpoint(path)
            self.assertEqual(path.read_bytes(), saved)
            self.assertEqual(reopened.all_knoxels[message.id].metadata['exchange_status'], 'resolved')
            self.assertEqual(reopened.all_knoxels[reply.id].metadata['publication_status'], 'discarded')
            self.assertEqual(reopened.all_knoxels[action.id].metadata['execution_status'], 'cancelled')
            clean = reopened.export_clean(Path(directory) / 'clean.pickle')
            again = MemoryStore.from_checkpoint(clean)
            self.assertEqual(again.all_knoxels[message.id].metadata['exchange_status'], 'resolved')
            self.assertEqual(again.all_knoxels[reply.id].metadata['publication_status'], 'discarded')
            self.assertEqual(path.read_bytes(), saved)

