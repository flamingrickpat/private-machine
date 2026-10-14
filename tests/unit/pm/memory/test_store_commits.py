"""Real memory batches and real checkpoint failures. No model or worker thread."""
import tempfile
import json
import unittest
from pathlib import Path
from threading import Event

from pm.memory.store import MemoryStore
from pm.memory.store import GhostConflict, GhostCancelled
from pm.records.cycle_state import PersonaConfig, CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Feature
from pm.records.enums import FeatureType
from pm.records.cognitive_records import AttentionFrameRecord
from pm.memory.yaml_codec import ghost_state_from_dict
from tests.support import load_seed_state


def small_ghost(checkpoint=None):
    return MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=checkpoint)


class GhostCommitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "ghost.json"
        self.ghost = small_ghost(self.path)

    def test_proposals_stay_private_until_complete_commit(self):
        batch = self.ghost.begin_batch("turn", new_tick=True)
        feature = batch.create(Feature(feature_type=FeatureType.Dialogue, content="a memory"))
        frame = batch.create(AttentionFrameRecord(content="focus", focus_knoxel_ids=[feature.id]))
        self.assertIsNone(self.ghost.get_knoxel_by_id(feature.id))
        self.assertEqual(batch.view().get(feature.id).content, "a memory")
        batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, attention_frame_id=frame.id, knoxel_ids=[feature.id, frame.id]),
                      CycleState(tick_id=batch.tick_id, attention_frame_id=frame.id))
        receipt = self.ghost.commit(batch)
        reopened = MemoryStore.from_checkpoint(self.path)
        self.assertEqual(reopened.get_knoxel_by_id(feature.id).content, "a memory")
        self.assertEqual(reopened.gwt_turns[batch.tick_id].attention_frame_id, frame.id)
        self.assertEqual(reopened.revisions[feature.id], 1)
        self.assertEqual(self.ghost.commit(batch), receipt)
        self.assertEqual(reopened.commit(reopened.begin_batch("turn")), receipt)

    def test_snapshot_is_stable_scoped_and_returns_copies(self):
        first = self.ghost.store_knoxel(Feature(feature_type=FeatureType.Dialogue, content="visible"))
        hidden = self.ghost.store_knoxel(Feature(feature_type=FeatureType.Dialogue, content="hidden"))
        view = self.ghost.snapshot([first.id])
        view.get(first.id).content = "caller mutation"
        batch = self.ghost.begin_batch("update")
        replacement = batch.view().get(first.id)
        replacement.content = "new accepted text"
        batch.replace(replacement)
        self.ghost.commit(batch)
        self.assertEqual(view.get(first.id).content, "visible")
        with self.assertRaises(KeyError):
            view.get(hidden.id)
        self.assertEqual(self.ghost.get_knoxel_by_id(first.id).content, "new accepted text")

    def test_unrelated_append_does_not_conflict_but_stale_update_does(self):
        original = self.ghost.store_knoxel(Feature(feature_type=FeatureType.Dialogue, content="original"))
        first = self.ghost.begin_batch("first")
        stale = self.ghost.begin_batch("stale")
        for batch, text in ((first, "winner"), (stale, "loser")):
            replacement = batch.view().get(original.id)
            replacement.content = text
            batch.replace(replacement)
        self.ghost.store_knoxel(Feature(feature_type=FeatureType.Dialogue, content="unrelated"))
        self.ghost.commit(first)
        proposed = stale.create(Feature(feature_type=FeatureType.Dialogue, content="must not leak"))
        before = self.path.read_bytes()
        with self.assertRaises(GhostConflict):
            self.ghost.commit(stale)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertIsNone(self.ghost.get_knoxel_by_id(proposed.id))
        self.assertEqual(self.ghost.get_knoxel_by_id(original.id).content, "winner")

    def test_reserved_ids_are_unique_and_gaps_survive_restart(self):
        abandoned = self.ghost.begin_batch("abandoned")
        first = abandoned.create(Feature(feature_type=FeatureType.Dialogue, content="discarded"))
        next_batch = self.ghost.begin_batch("next")
        second = next_batch.create(Feature(feature_type=FeatureType.Dialogue, content="accepted"))
        abandoned.discard()
        self.ghost.commit(next_batch)
        reopened = MemoryStore.from_checkpoint(self.path)
        third = reopened.store_knoxel(Feature(feature_type=FeatureType.Dialogue, content="after restart"))
        self.assertGreater(second.id, first.id)
        self.assertGreater(third.id, second.id)
        self.assertIsNone(reopened.get_knoxel_by_id(first.id))

    def test_invalid_reference_rejects_the_whole_batch(self):
        batch = self.ghost.begin_batch("bad-reference")
        feature = batch.create(Feature(feature_type=FeatureType.Dialogue, content="unpublished"))
        batch.create(AttentionFrameRecord(content="bad focus", focus_knoxel_ids=[999999]))
        with self.assertRaises(ValueError):
            self.ghost.commit(batch)
        self.assertIsNone(self.ghost.get_knoxel_by_id(feature.id))
        self.assertFalse(self.path.exists())

    def test_cancelled_batch_cannot_publish(self):
        cancel = Event()
        batch = self.ghost.begin_batch("cancelled", cancel=cancel)
        feature = batch.create(Feature(feature_type=FeatureType.Dialogue, content="discarded"))
        cancel.set()
        with self.assertRaises(GhostCancelled):
            self.ghost.commit(batch)
        self.assertIsNone(self.ghost.get_knoxel_by_id(feature.id))
        self.assertFalse(self.path.exists())

    def test_real_replace_failure_preserves_checkpoint_and_memory(self):
        accepted = self.ghost.store_knoxel(Feature(feature_type=FeatureType.Dialogue, content="accepted"))
        before = self.path.read_bytes()
        # A nonempty directory is a real invalid replacement target on every platform.
        blocked = Path(self.directory.name) / "blocked"
        blocked.mkdir()
        (blocked / "keep.txt").write_text("keep", encoding="utf-8")
        self.ghost.checkpoint = blocked
        batch = self.ghost.begin_batch("failed-save")
        feature = batch.create(Feature(feature_type=FeatureType.Dialogue, content="not accepted"))
        with self.assertRaises(OSError):
            self.ghost.commit(batch)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertIsNone(self.ghost.get_knoxel_by_id(feature.id))
        self.assertEqual(self.ghost.get_knoxel_by_id(accepted.id).content, "accepted")
        self.assertNotIn("failed-save", self.ghost.applied_commits)
        self.assertEqual(list(blocked.parent.glob("*.tmp")), [])

    def test_restart_marks_live_runs_interrupted_without_replaying(self):
        batch = self.ghost.begin_batch("receipts")
        batch.set_run_receipt("running-job", {"state": "running", "workflow": "local"})
        batch.set_run_receipt("done-job", {"state": "completed", "workflow": "local"})
        self.ghost.commit(batch)
        reopened = MemoryStore.from_checkpoint(self.path)
        self.assertEqual(reopened.managed_runs["running-job"]["state"], "interrupted")
        self.assertEqual(reopened.managed_runs["done-job"]["state"], "completed")

    def test_receipt_commit_preserves_records_and_checkpoint_failure_is_atomic(self):
        record = self.ghost.store_knoxel(Feature(feature_type=FeatureType.Dialogue, content="retained"))
        before = self.ghost.state_dict()
        batch = self.ghost.begin_batch("run-only")
        batch.set_run_receipt("job", {"state": "completed", "workflow": "local"})
        self.ghost.commit(batch)
        reopened = MemoryStore.from_checkpoint(self.path)
        self.assertEqual(reopened.state_dict(), before)
        self.assertEqual(reopened.managed_runs["job"]["state"], "completed")
        self.assertEqual(self.ghost.get_knoxel_by_id(record.id).content, "retained")
        blocked = Path(self.directory.name) / "blocked-receipt"
        blocked.mkdir()
        (blocked / "keep").write_text("keep", encoding="utf-8")
        self.ghost.checkpoint = blocked
        batch = self.ghost.begin_batch("failed-run-only")
        batch.set_run_receipt("job", {"state": "failed", "workflow": "local"})
        with self.assertRaises(OSError):
            self.ghost.commit(batch)
        self.assertEqual(self.ghost.managed_runs["job"]["state"], "completed")
        self.assertNotIn("failed-run-only", self.ghost.applied_commits)

    def test_narrow_batch_view_hides_revisions_and_attention_frame(self):
        batch = self.ghost.begin_batch("frame", new_tick=True)
        frame = batch.create(AttentionFrameRecord(content="hidden"))
        batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, attention_frame_id=frame.id, knoxel_ids=[]))
        self.ghost.commit(batch)
        self.assertEqual(self.ghost.snapshot([]).serialize_gwt(), {})
        narrowed = self.ghost.begin_batch("read").view([])
        self.assertEqual(dict(narrowed.revisions), {})
        self.assertEqual(narrowed.serialize_gwt(), {})

    def test_rebuild_does_not_reuse_reserved_ids(self):
        self.ghost.store_knoxel(Feature(feature_type=FeatureType.Dialogue, content="accepted"))
        pending = self.ghost.begin_batch("pending")
        reserved = pending.create(Feature(feature_type=FeatureType.Dialogue, content="pending"))
        self.ghost.rebuild_knoxel_indexes()
        accepted = self.ghost.store_knoxel(Feature(feature_type=FeatureType.Dialogue, content="another"))
        self.assertGreater(accepted.id, reserved.id)
        self.ghost.commit(pending)
        self.assertEqual(self.ghost.get_knoxel_by_id(reserved.id).content, "pending")

    def test_new_records_require_the_batch_tick(self):
        batch = self.ghost.begin_batch("wrong-tick", new_tick=True)
        with self.assertRaises(ValueError):
            batch.create(Feature(feature_type=FeatureType.Dialogue, content="bad tick", tick_id=batch.tick_id + 10))

    def test_replacement_cannot_change_type_scope_or_tick(self):
        accepted = self.ghost.store_knoxel(Feature(feature_type=FeatureType.Dialogue, content="accepted"))
        batch = self.ghost.begin_batch("bad-replacement")
        with self.assertRaises(ValueError):
            batch.replace(AttentionFrameRecord(id=accepted.id, tick_id=accepted.tick_id, content="different type"))
        replacement = batch.view().get(accepted.id)
        replacement.tick_id += 1
        with self.assertRaises(ValueError):
            batch.replace(replacement)
        replacement = batch.view().get(accepted.id)
        replacement.scope.reality_id = 123
        with self.assertRaises(ValueError):
            batch.replace(replacement)

    def test_old_mutation_helpers_cannot_bypass_commit(self):
        record = Feature(feature_type=FeatureType.Dialogue, content="uncommitted")
        with self.assertRaises(NotImplementedError):
            self.ghost.add_knoxel(record)
        with self.assertRaises(NotImplementedError):
            self.ghost.replace_knoxel(record)
        self.assertEqual(self.ghost.all_knoxels, {})

    def test_invalid_checkpoint_metadata_is_rejected(self):
        batch = self.ghost.begin_batch("turn", new_tick=True)
        batch.create(Feature(feature_type=FeatureType.Dialogue, content="accepted"))
        batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id))
        self.ghost.commit(batch)
        original = self.path.read_text(encoding="utf-8")
        corruptions = [
            ("current_knoxel_id", 0), ("current_tick_id", 0), ("reserved_tick_id", 0),
            ("revisions", {}), ("schema_id", "unknown"),
            ("gwt_turns", {"22": {"tick_id": 1}}),
            ("applied_commits", {"wrong-key": json.loads(original)["applied_commits"]["turn"]}),
        ]
        for field, value in corruptions:
            with self.subTest(field=field):
                data = json.loads(original)
                data[field] = value
                self.path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    MemoryStore.from_checkpoint(self.path)



class FullSeedCheckpointTests(unittest.TestCase):
    def test_full_spamuel_round_trip_preserves_models_and_state(self):
        seed = load_seed_state()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ghost.json"
            ghost = MemoryStore.from_state(ghost_state_from_dict(seed), checkpoint=path)
            ghost.commit(ghost.begin_batch("checkpoint-seed", visible_ids=[]))
            reopened = MemoryStore.from_checkpoint(path)
            self.assertEqual(reopened.state_dict(), seed)
            self.assertEqual(len(reopened.all_knoxels), 11202)
            self.assertEqual(reopened.get_entity_id("Jesse"), ghost.get_entity_id("Jesse"))
            self.assertEqual(reopened.current_knoxel_id, ghost.current_knoxel_id)
