"""Exercise immutable history and recovery against real checkpoint files."""
import tempfile
import unittest
from pathlib import Path

from pm.memory.store import MemoryStore
from pm.memory.snapshot import GhostConflict
from pm.records.cycle_state import PersonaConfig
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Feature
from pm.records.enums import FeatureType


class CausalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "ghost.pickle"
        self.ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=self.path)

    def feature(self, text):
        return Feature(content=text, feature_type=FeatureType.Dialogue)

    def admit(self, inputs=()):
        batch = self.ghost.begin_batch(f"main-{self.ghost.current_tick_id + 1}", new_tick=True)
        batch.depend_on(inputs)
        action = batch.create(self.feature("approved action"))
        batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, slot_id="main", stimulus_ids=list(inputs), knoxel_ids=[action.id, *inputs]))
        self.ghost.commit(batch)
        return action

    def result(self, action, inputs=()):
        batch = self.ghost.begin_batch("reply")
        batch.depend_on(inputs, authority=action.id)
        reply = batch.create(self.feature("completed reply"))
        batch.set_lifecycle(action.id, execution_status="completed", public_reply_id=reply.id)
        self.ghost.commit(batch)
        return reply

    def test_default_restart_excludes_latest_message_and_keeps_raw_history(self):
        first = self.ghost.store_knoxel(self.feature("accepted input"))
        self.admit([first.id])
        latest = self.ghost.store_knoxel(self.feature("failed latest input"))
        clean = MemoryStore.from_checkpoint(self.path)
        self.assertIsNotNone(clean.get_knoxel_by_id(first.id))
        self.assertIsNone(clean.get_knoxel_by_id(latest.id))
        raw = MemoryStore.from_checkpoint(self.path, raw=True)
        self.assertEqual(raw.get_knoxel_by_id(latest.id).content, "failed latest input")
        clean.save()
        self.assertIsNotNone(MemoryStore.from_checkpoint(self.path, raw=True).get_knoxel_by_id(latest.id))

    def test_older_approved_reply_survives_new_unfinished_input(self):
        action = self.admit()
        latest = self.ghost.store_knoxel(self.feature("new provisional input"))
        reply = self.result(action)
        clean = MemoryStore.from_checkpoint(self.path)
        self.assertIsNone(clean.get_knoxel_by_id(latest.id))
        self.assertIsNotNone(clean.get_knoxel_by_id(reply.id))
        self.assertEqual(clean.get_knoxel_by_id(action.id).metadata["execution_status"], "completed")

    def test_reply_that_read_unaccepted_input_is_excluded_and_action_stays_pending(self):
        action = self.admit()
        latest = self.ghost.store_knoxel(self.feature("new provisional input"))
        reply = self.result(action, [latest.id])
        clean = MemoryStore.from_checkpoint(self.path)
        self.assertIsNone(clean.get_knoxel_by_id(reply.id))
        self.assertNotIn("public_reply_id", clean.get_knoxel_by_id(action.id).metadata)
        self.admit([latest.id])
        self.assertIsNotNone(MemoryStore.from_checkpoint(self.path).get_knoxel_by_id(reply.id))

    def test_direct_host_reads_capture_late_inputs_without_a_new_snapshot(self):
        from pm.memory.read_scope import ReadScope
        action = self.admit()
        latest = self.ghost.store_knoxel(self.feature("late input"))
        with ReadScope():
            self.ghost.get_knoxel_by_id(latest.id)
            reply = self.result(action)
        self.assertIsNone(MemoryStore.from_checkpoint(self.path).get_knoxel_by_id(reply.id))

    def test_live_collection_reads_capture_their_stable_membership(self):
        from pm.memory.read_scope import ReadScope
        action = self.admit()
        with ReadScope():
            frozen = self.ghost.all_features
            self.assertEqual(len(frozen), 1)
        latest = self.ghost.store_knoxel(self.feature("late input"))
        self.assertEqual(len(frozen), 1)
        with ReadScope():
            self.assertIn(latest.id, [record.id for record in self.ghost.all_features])
            reply = self.result(action)
        self.assertIsNone(MemoryStore.from_checkpoint(self.path).get_knoxel_by_id(reply.id))

    def test_unrelated_entity_lookup_does_not_admit_or_depend_on_a_late_message(self):
        from pm.memory.read_scope import ReadScope
        from pm.records.memory_records import Entity
        from pm.records.enums import EntityClass
        entity = self.ghost.store_knoxel(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        action = self.admit([entity.id])
        latest = self.ghost.store_knoxel(self.feature("late input"))
        with ReadScope():
            self.assertEqual(self.ghost.get_entity_name(entity.id), "Jesse")
            reply = self.result(action)
        clean = MemoryStore.from_checkpoint(self.path)
        self.assertIsNotNone(clean.get_knoxel_by_id(reply.id))
        self.assertIsNone(clean.get_knoxel_by_id(latest.id))

    def test_receipt_inputs_capture_status_changes_without_a_new_knoxel(self):
        from pm.memory.read_scope import ReadScope
        action = self.admit()
        latest = self.ghost.store_knoxel(self.feature("late input"))
        status = self.ghost.begin_batch("run-status")
        status.depend_on([latest.id], authority=action.id)
        status.set_run_receipt("run", {"state": "running", "causal_commit_id": status.commit_id})
        self.ghost.commit(status)
        with ReadScope():
            self.assertEqual(self.ghost.read_run_receipts()["run"]["state"], "running")
            reply = self.result(action)
        self.assertIsNone(MemoryStore.from_checkpoint(self.path).get_knoxel_by_id(reply.id))

    def test_collection_absence_depends_on_an_unaccepted_deletion(self):
        from pm.memory.read_scope import ReadScope
        source = self.ghost.store_knoxel(self.feature("an accepted event"))
        action = self.admit([source.id])
        deletion = self.ghost.begin_batch("unaccepted-deletion")
        deletion.delete(source.id)
        self.ghost.commit(deletion)
        with ReadScope():
            self.assertEqual([record.id for record in self.ghost.all_features], [action.id])
            reply = self.result(action)
        clean = MemoryStore.from_checkpoint(self.path)
        self.assertIsNotNone(clean.get_knoxel_by_id(source.id))
        self.assertIsNone(clean.get_knoxel_by_id(reply.id))

    def test_reading_private_cycle_state_keeps_its_unaccepted_dependency(self):
        from pm.memory.read_scope import ReadScope
        action = self.admit()
        private = self.ghost.begin_batch("private-cycle", new_tick=True)
        thought = private.create(self.feature("unaccepted private event"))
        private.set_gwt(GwtTurnRecord(tick_id=private.tick_id, slot_id="inner", knoxel_ids=[thought.id]),
                        CycleState(tick_id=private.tick_id))
        self.ghost.commit(private)
        with ReadScope():
            self.ghost.read_cycles()
            self.ghost.read_workspace_turns()
            reply = self.result(action)
        self.assertIsNone(MemoryStore.from_checkpoint(self.path).get_knoxel_by_id(reply.id))
        self.admit([thought.id])
        self.assertIsNotNone(MemoryStore.from_checkpoint(self.path).get_knoxel_by_id(reply.id))

    def test_successor_keeps_original_payload_and_stale_head_conflicts(self):
        action = self.admit()
        stale = self.ghost.begin_batch("stale")
        changes = self.ghost.begin_batch("change")
        successor = action.copy_for_edit()
        successor.content = "new state"
        changes.replace(successor)
        stale.replace(action.copy_for_edit())
        self.ghost.commit(changes)
        self.assertNotEqual(action.id, successor.id)
        self.assertEqual(successor.supersedes_id, action.id)
        self.assertEqual(self.ghost.get_knoxel_by_id(action.id).content, "approved action")
        self.assertNotIn(action.id, self.ghost.all_knoxels)
        with self.assertRaises(GhostConflict):
            self.ghost.commit(stale)
        clean = MemoryStore.from_checkpoint(self.path)
        self.assertIn(action.id, clean.all_knoxels)
        self.assertNotIn(successor.id, clean.all_knoxels)

    def test_excluded_tombstone_does_not_delete_accepted_event(self):
        action = self.admit()
        changes = self.ghost.begin_batch("delete")
        changes.delete(action.id)
        self.ghost.commit(changes)
        self.assertNotIn(action.id, self.ghost.all_knoxels)
        self.assertIn(action.id, MemoryStore.from_checkpoint(self.path).all_knoxels)

    def test_lifecycle_is_separate_from_original_knoxel(self):
        action = self.admit()
        self.result(action)
        self.assertNotIn("execution_status", self.ghost._history[action.id].metadata)
        clean = MemoryStore.from_checkpoint(self.path)
        self.assertEqual(clean.get_knoxel_by_id(action.id).metadata["execution_status"], "completed")
        self.assertNotIn("execution_status", clean._history[action.id].metadata)

    def test_clean_restart_never_reuses_excluded_record_ids(self):
        self.admit()
        excluded = self.ghost.store_knoxel(self.feature("excluded"))
        clean = MemoryStore.from_checkpoint(self.path)
        next_record = clean.store_knoxel(self.feature("new branch"))
        self.assertGreater(next_record.id, excluded.id)

    def test_clean_export_preserves_source_and_drops_excluded_branch(self):
        action = self.admit()
        excluded = self.ghost.store_knoxel(self.feature("excluded"))
        before = self.path.read_bytes()
        destination = self.path.with_name("clean.pickle")
        self.ghost.export_clean(destination)
        exported = MemoryStore.from_checkpoint(destination, raw=True)
        self.assertIsNotNone(exported.get_knoxel_by_id(action.id))
        self.assertIsNone(exported.get_knoxel_by_id(excluded.id))
        self.assertEqual(self.path.read_bytes(), before)
        exported_bytes = destination.read_bytes()
        with self.assertRaises(FileExistsError):
            self.ghost.export_clean(destination)
        self.assertEqual(destination.read_bytes(), exported_bytes)
        exported.store_knoxel(self.feature("branch after clean export"))
        MemoryStore.from_checkpoint(destination)

    def test_historical_ids_remain_readable_but_leave_normal_queries(self):
        action = self.admit()
        batch = self.ghost.begin_batch("successor")
        successor = action.copy_for_edit()
        successor.content = "changed goal"
        batch.replace(successor)
        self.ghost.commit(batch)
        view = self.ghost.snapshot()
        self.assertEqual(view.get(action.id).content, "approved action")
        self.assertNotIn(action.id, view.all_knoxels)
        self.assertIn(successor.id, view.all_knoxels)

    def test_legacy_baseline_retains_cycles_without_gwt_records(self):
        self.ghost.states = [CycleState(tick_id=7)]
        self.ghost.current_tick_id = 7
        self.ghost._reserved_tick_id = 7
        self.ghost.save()
        restored = MemoryStore.from_checkpoint(self.path)
        self.assertEqual([item.tick_id for item in restored.states], [7])

    def test_legacy_baseline_preserves_commit_replay_after_migration_and_clean_export(self):
        from pm.memory.checkpoint import write_checkpoint
        action = self.admit()
        legacy = self.ghost.checkpoint_data().model_copy(update={"causal_history": None})
        write_checkpoint(self.path, legacy, lambda: None)
        migrated = MemoryStore.from_checkpoint(self.path)
        migrated.save()
        reopened = MemoryStore.from_checkpoint(self.path)
        self.assertEqual(reopened.applied_commits["main-1"].added_ids, (action.id,))
        target = Path(self.directory.name) / "clean.pickle"
        reopened.export_clean(target)
        self.assertIn("main-1", MemoryStore.from_checkpoint(target).applied_commits)

    def test_excluded_commit_id_can_retry_without_reviving_old_result(self):
        action = self.admit()
        provisional = self.ghost.store_knoxel(self.feature("unaccepted"))
        excluded = self.result(action, [provisional.id])
        restored = MemoryStore.from_checkpoint(self.path)
        restored.store_knoxel(self.feature("another unaccepted input"))
        batch = restored.begin_batch("reply")
        batch.depend_on([action.id], authority=action.id)
        replacement = batch.create(self.feature("retried against accepted input"))
        restored.commit(batch)
        reopened = MemoryStore.from_checkpoint(self.path)
        self.assertIsNone(reopened.get_knoxel_by_id(excluded.id))
        self.assertIsNotNone(reopened.get_knoxel_by_id(replacement.id))
        repeated = reopened.begin_batch("reply")
        duplicate = repeated.create(self.feature("must not repeat the accepted result"))
        receipt = reopened.commit(repeated)
        self.assertEqual(receipt.added_ids, (replacement.id,))
        self.assertIsNone(reopened.get_knoxel_by_id(duplicate.id))

    def test_committed_payloads_reject_direct_changes_and_edit_copies_remain_mutable(self):
        batch = self.ghost.begin_batch("event")
        event = batch.create(self.feature("original"))
        event.metadata["nested"] = {"sources": [1, 2]}
        from pm.records.mental_state import FullMentalState
        event.metadata["appraisal"] = FullMentalState()
        self.ghost.commit(batch)
        with self.assertRaises(TypeError):
            event.content = "bypass"
        with self.assertRaises(TypeError):
            event.metadata["nested"]["sources"].append(3)
        with self.assertRaises(TypeError):
            event.metadata["appraisal"].state_emotions.joy = 1.0
        edited = event.copy_for_edit()
        edited.content = "isolated edit"
        edited.metadata["appraisal"].state_emotions.joy = 1.0
        edited.metadata["nested"]["sources"].append(3)
        self.assertEqual(event.content, "original")
        self.assertEqual(event.metadata["nested"]["sources"], [1, 2])

    def test_receipt_overlay_keeps_new_derived_vectors_in_raw_checkpoint(self):
        action = self.admit()
        self.result(action)
        live = self.ghost.get_knoxel_by_id(action.id)
        live.embedding = [1.0, 2.0]
        live.embedding_model = "new-model"
        self.ghost.save()
        restored = MemoryStore.from_checkpoint(self.path)
        self.assertEqual(restored.get_knoxel_by_id(action.id).embedding, [1.0, 2.0])
        self.assertEqual(restored.get_knoxel_by_id(action.id).embedding_model, "new-model")
