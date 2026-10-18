"""Check record exports against real committed memory and frozen GWT turns."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.introspection.record_packages import capture_cycle, capture_records, capture_gwt, dump_package
from pm.records.enums import FeatureType, SourceKind
from pm.records.memory_records import new_feature
from tests.unit.pm.introspection.test_episodes import small_ghost, _commit_thought


class RecordPackageTests(unittest.TestCase):
    def setUp(self):
        self.ghost = small_ghost()
        self.thought_id, _ = _commit_thought(self.ghost)
        self.related_id = new_feature(self.ghost, "a related source", FeatureType.Dialogue, source="Jesse",
                                      source_kind=SourceKind.User).id
        self.view = self.ghost.snapshot()

    def test_preserves_fields_revisions_and_snapshot(self):
        package = capture_records(self.view, [self.thought_id, self.thought_id], title="Private thought")
        original = self.view.serialize_knoxels()[self.thought_id]["data"]
        data = package["records"][0]["data"]
        self.assertNotIn("embedding", data)
        self.assertEqual(data["metadata"], original["metadata"])
        self.assertEqual(data["timestamp_creation"], original["timestamp_creation"])
        self.assertEqual(data["mental_state_delta_vec"], original["mental_state_delta_vec"])
        self.assertEqual(package["record_ids"], [self.thought_id])
        self.assertEqual(package["revisions"][str(self.thought_id)], self.view.revisions[self.thought_id])
        data["metadata"]["interpretation"] = "Changed only in this export"
        self.assertEqual(self.view.serialize_knoxels()[self.thought_id]["data"], original)

    def test_gwt_retains_exact_order_and_turn(self):
        tick = self.view.get(self.thought_id).tick_id
        package = capture_gwt(self.view, tick)
        self.assertEqual(package["record_ids"], self.view.serialize_gwt()[tick]["knoxel_ids"])
        self.assertEqual(package["gwt"], self.view.serialize_gwt()[tick])
        with self.assertRaises(ValueError):
            capture_records(self.view, [self.related_id], title="Wrong members", view="gwt", tick_id=tick)

    def test_chat_preserves_caller_order_and_rejects_private_types(self):
        one = new_feature(self.ghost, "later", FeatureType.Dialogue, source="Jesse", source_kind=SourceKind.User)
        two = new_feature(self.ghost, "back", FeatureType.Dialogue, source="Jesse", source_kind=SourceKind.User)
        package = capture_records(self.ghost.snapshot(), [two.id, one.id], title="Jesse's view", view="chat", perspective="Jesse")
        self.assertEqual([row["data"]["content"] for row in package["records"]], ["back", "later"])
        with self.assertRaises(ValueError):
            capture_records(self.view, [self.thought_id], title="Not public", view="chat")

    def test_missing_sources_and_ticks_raise(self):
        with self.assertRaises(KeyError):
            capture_records(self.view, [999999], title="Missing")
        with self.assertRaises(KeyError):
            capture_gwt(self.view, 999999)

    def test_dump_is_readable_and_preserves_related_membership(self):
        package = capture_records(self.view, [self.thought_id], related_ids=[self.related_id], title="Related focus")
        self.assertEqual(package["record_ids"], [self.thought_id])
        self.assertEqual(len(package["records"]), 2)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "capture.json"
            dump_package(package, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), package)


class CyclePackageTests(unittest.TestCase):
    def test_a_cycle_package_holds_the_state_the_turn_and_every_named_record(self):
        from pm.records.codelet_records import Structure
        from pm.records.cycle_records import ActionDecision, StimulusGroup
        from pm.records.cycle_state import CycleState
        from pm.records.global_workspace import GwtTurnRecord
        from pm.records.situational_model import SituationalItem
        from pm.records.workspace import SELF_SHARD, WorkspaceItem
        ghost = small_ghost()
        stimulus = new_feature(ghost, "ok, honest question", FeatureType.Dialogue, source="Jesse",
                               source_kind=SourceKind.User)
        batch = ghost.begin_batch("cycle", new_tick=True)
        structure = batch.create(Structure(content="dodging: he will not choose", internal_knoxel_ids=[stimulus.id]))
        state = CycleState(tick_id=batch.tick_id, structure_ids=[structure.id],
                           stimulus_groups=[StimulusGroup(stimulus_ids=[stimulus.id])],
                           decisions=[ActionDecision(text="Call him a coward.", stimulus_ids=[stimulus.id])])
        state.csm.items[stimulus.id] = SituationalItem(record_id=stimulus.id, activation=0.9, endorsers=["retrieval"])
        state.workspace.self_shard.items.append(WorkspaceItem(record_id=structure.id, reason="test"))
        batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, stimulus_ids=[stimulus.id],
                                    knoxel_ids=[stimulus.id, structure.id]), state)
        ghost.commit(batch)
        package = capture_cycle(ghost.snapshot(), ghost.states[-1])
        self.assertEqual(package["view"], "cycle")
        self.assertEqual(package["record_ids"], [stimulus.id, structure.id])
        self.assertEqual(package["cycle"]["decisions"][0]["text"], "Call him a coward.")
        self.assertEqual(package["cycle"]["csm"]["items"][str(stimulus.id)]["endorsers"], ["retrieval"])
        self.assertEqual(package["gwt"]["stimulus_ids"], [stimulus.id])
        self.assertNotIn("embedding", json.dumps(package))
        self.assertEqual(SELF_SHARD.kind, package["cycle"]["workspace"]["self_shard"]["key"]["kind"])


if __name__ == "__main__":
    unittest.main()
