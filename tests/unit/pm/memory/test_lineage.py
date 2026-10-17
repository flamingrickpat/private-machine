"""Episode links retain original sources without expanding a memory grant."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.memory.agent_tools import MemoryReadSession, MemoryToolConfig, MemoryView
from pm.memory.references import referenced_ids
from pm.memory.store import MemoryStore
from pm.memory.yaml_codec import _serialize_knoxel
from pm.records.enums import FeatureType
from pm.records.lineage import RecordLineage
from pm.records.memory_records import new_feature
from tests.unit.test_design_contracts import small_ghost


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.ghost = small_ghost()
        self.source = new_feature(self.ghost, "the observed source", FeatureType.Dialogue, source="Jesse")

    def lineage(self, source_id=None):
        return RecordLineage(episode_id="private-episode", operation="imagine", reality="imagined",
                             sequence=2, parent_record_ids=[source_id or self.source.id],
                             source_event_ids=[self.source.id], topic_hints=["revisable topic hint"])

    def test_lineage_survives_checkpoint_without_changing_reality_or_original(self):
        with TemporaryDirectory() as tmp:
            self.ghost.checkpoint = Path(tmp) / "ghost.json"
            original = _serialize_knoxel(self.ghost.get_knoxel_by_id(self.source.id))
            batch = self.ghost.begin_batch("new-imagined-record")
            thought = new_feature(self.ghost, "an imagined outcome", FeatureType.InnerMonologue,
                                  source="Spamuel", changes=batch, lineage=self.lineage())
            self.ghost.commit(batch)
            restored = MemoryStore.from_checkpoint(self.ghost.checkpoint)
            self.assertEqual(restored.get_knoxel_by_id(thought.id).lineage, self.lineage())
            self.assertEqual(_serialize_knoxel(restored.get_knoxel_by_id(self.source.id)), original)
            self.assertEqual(referenced_ids(restored.get_knoxel_by_id(thought.id)), {self.source.id})

    def test_new_lineage_cannot_rewrite_an_existing_sources_history(self):
        batch = self.ghost.begin_batch("cannot-rewrite")
        self.assertEqual(batch.created_ids, frozenset())
        with self.assertRaises(ValueError):
            batch.annotate_lineage(self.source.id, self.lineage())
        self.assertIsNone(self.ghost.get_knoxel_by_id(self.source.id).lineage)

    def test_missing_or_out_of_grant_lineage_rejects_the_whole_batch(self):
        for missing in (99999, self.source.id):
            with self.subTest(missing=missing):
                batch = self.ghost.begin_batch("bad-lineage-" + str(missing), visible_ids=[])
                new_feature(self.ghost, "rejected", FeatureType.InnerMonologue,
                            source="Spamuel", changes=batch, lineage=self.lineage(missing))
                before = self.ghost.state_dict()
                with self.assertRaises(ValueError):
                    self.ghost.commit(batch)
                self.assertEqual(self.ghost.state_dict(), before)

    def test_read_session_does_not_disclose_a_hidden_lineage_source(self):
        thought = new_feature(self.ghost, "derived private content", FeatureType.InnerMonologue,
                              source="Spamuel", lineage=self.lineage())
        scope = thought.scope
        grant = MemoryView(record_ids=frozenset({thought.id}),
                           scopes=frozenset({(scope.reality_id, scope.frame.value)}))
        with MemoryReadSession(MemoryToolConfig(store=self.ghost, view=grant)) as session:
            with self.assertRaisesRegex(ValueError, "unavailable"):
                session.get(thought.id)
            self.assertEqual(session.view()["categories"], {})
            self.assertEqual(session.accessed, {})

    def test_old_records_omit_the_optional_field_on_serialization(self):
        raw = _serialize_knoxel(self.ghost.get_knoxel_by_id(self.source.id))
        self.assertNotIn("lineage", raw["data"])


if __name__ == "__main__":
    unittest.main()
