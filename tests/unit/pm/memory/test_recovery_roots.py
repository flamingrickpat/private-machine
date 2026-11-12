"""Exercise explicit recovery roots against actual stores and checkpoint files.

These deterministic persistence checks use no model or service substitute.
Existing causal-recovery checks cover ordinary main and action admission.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.memory.bash_view import MemoryBashTarget
from pm.memory.causal import CausalHistory, Transaction
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature


class RecoveryRootTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "ghost.pickle"
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=self.path)
        self.store.accept_import()

    def root(self, commit_id: str, text: str, inputs: tuple[int, ...] = ()) -> Feature:
        """Commit one real append-only input or control receipt."""
        batch = self.store.begin_batch(commit_id, embed=False)
        batch.preserve_on_recovery()
        batch.depend_on(inputs)
        record = batch.create(Feature(content=text, feature_type=FeatureType.StateUpdate))
        self.store.commit(batch)
        return record

    def test_import_baseline_does_not_advertise_removed_transaction_dependencies(self) -> None:
        imported = self.store.store_knoxel(Feature(content="Imported identity context", feature_type=FeatureType.StateUpdate))
        before = self.store.snapshot(capture=False)
        self.assertTrue(before.commit_ids)
        self.store.accept_import()
        view = self.store.snapshot(capture=False)
        history = self.store.causal_history
        assert history is not None
        self.assertTrue(before.commit_ids.issubset(history.baseline_commit_ids))
        self.assertTrue(view.commit_ids.issubset(history.transactions))
        self.assertEqual(view.commit_ids, frozenset())
        batch = self.store.begin_batch("first-after-import", embed=False)
        batch.depend_on([imported.id])
        batch.depend_on_commits(view.commit_ids)
        batch.preserve_on_recovery()
        receipt = batch.create(Feature(content="Accepted immediate update", feature_type=FeatureType.StateUpdate))
        self.store.commit(batch)
        restored = MemoryStore.from_checkpoint(self.path)
        restored_view = restored.snapshot(capture=False)
        restored_history = restored.causal_history
        assert restored_history is not None
        self.assertTrue(restored_view.commit_ids.issubset(restored_history.transactions))
        self.assertEqual(restored_view.get(imported.id).content, imported.content)
        self.assertEqual(restored_view.get(receipt.id).content, receipt.content)
        follow_up = restored.begin_batch("second-after-import", embed=False)
        follow_up.depend_on_commits(restored_view.commit_ids)
        follow_up.preserve_on_recovery()
        restored.commit(follow_up)
        self.assertIn("second-after-import", MemoryStore.from_checkpoint(self.path).snapshot(capture=False).commit_ids)

    def test_default_restart_retains_roots_without_promoting_unrelated_branches(self) -> None:
        source = self.root("external-input", "A durable external input")
        speculative = self.store.begin_batch("speculation", embed=False)
        speculative.depend_on([source.id])
        draft = speculative.create(Feature(content="An unaccepted private draft", feature_type=FeatureType.Thought))
        self.store.commit(speculative)
        receipt = self.root("control-receipt", "The input requires reconsideration", (source.id,))
        projection = self.store.recovery_snapshot(capture=False)
        counters = (self.store.current_knoxel_id, self.store.current_tick_id, self.store._reserved_tick_id)
        clean = MemoryStore.from_checkpoint(self.path)
        raw = MemoryStore.from_checkpoint(self.path, raw=True)
        self.assertEqual(set(clean.snapshot().all_knoxels), {source.id, receipt.id})
        self.assertEqual(set(projection.all_knoxels), {source.id, receipt.id})
        self.assertIn(draft.id, raw.snapshot().all_knoxels)
        self.assertEqual(counters, (clean.current_knoxel_id, clean.current_tick_id, clean._reserved_tick_id))
        self.assertEqual(clean.states, [])
        self.assertEqual(dict(clean.gwt_turns), {})
        self.assertTrue(all(not item.main for item in clean.causal_history.transactions.values()))
        clean.save()
        self.assertIn(draft.id, MemoryStore.from_checkpoint(self.path, raw=True).snapshot().all_knoxels)

    def test_root_rejects_unaccepted_dependencies_before_checkpoint_changes(self) -> None:
        provisional = self.store.begin_batch("unaccepted", embed=False)
        draft = provisional.create(Feature(content="Unaccepted evidence", feature_type=FeatureType.Thought))
        self.store.commit(provisional)
        before = self.path.read_bytes()
        batch = self.store.begin_batch("invalid-root", embed=False)
        batch.preserve_on_recovery()
        batch.depend_on([draft.id])
        result = batch.create(Feature(content="Must not become a root", feature_type=FeatureType.StateUpdate))
        with self.assertRaisesRegex(ValueError, "unaccepted transactions"):
            self.store.commit(batch)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertNotIn(result.id, self.store.snapshot().all_knoxels)

    def test_root_rejects_mutation_of_existing_state(self) -> None:
        source = self.root("source", "Original external input")
        batch = self.store.begin_batch("invalid-state-change", embed=False)
        batch.preserve_on_recovery()
        batch.set_lifecycle(source.id, exchange_status="resolved")
        with self.assertRaisesRegex(ValueError, "only append"):
            self.store.commit(batch)
        self.assertNotIn("exchange_status", self.store.snapshot().get(source.id).metadata)

    def test_recovery_snapshot_restores_accepted_heads_without_speculative_overlays(self) -> None:
        source = self.root("source", "Accepted source")
        overlay = self.store.begin_batch("unaccepted-overlay", embed=False)
        overlay.set_lifecycle(source.id, publication_status="published")
        self.store.commit(overlay)
        replacement = self.store.begin_batch("unaccepted-successor", embed=False)
        successor = replacement.view().get(source.id).copy_for_edit()
        successor.content = "Speculative replacement"
        replacement.replace(successor)
        self.store.commit(replacement)
        before = set(self.store.snapshot().all_knoxels)
        projection = self.store.recovery_snapshot(capture=False)
        self.assertEqual(set(projection.all_knoxels), {source.id})
        self.assertEqual(projection.get(source.id).content, "Accepted source")
        self.assertNotIn("publication_status", projection.get(source.id).metadata)
        self.assertIsNone(projection.embedding_index)
        self.assertEqual(set(self.store.snapshot().all_knoxels), before)
        self.assertIn(successor.id, before)

    def test_recovery_only_environment_captures_only_accepted_dependencies(self) -> None:
        from pm.memory.read_scope import ReadScope

        source = self.root("source", "Accepted source")
        provisional = self.store.begin_batch("unaccepted", embed=False)
        draft = provisional.create(Feature(content="Unaccepted draft", feature_type=FeatureType.Thought))
        self.store.commit(provisional)
        with ReadScope():
            environment = MemoryBashTarget(self.store, recovery_only=True).open("read-accepted")
            self.assertIn(source.id, environment.visible_knoxels)
            self.assertNotIn(draft.id, environment.visible_knoxels)
            receipt = self.root("read-receipt", "A receipt over accepted context", (source.id,))
        clean = MemoryStore.from_checkpoint(self.path)
        self.assertIn(receipt.id, clean.snapshot().all_knoxels)
        self.assertNotIn(draft.id, clean.snapshot().all_knoxels)

    def test_old_serialized_transaction_without_root_marker_remains_unaccepted(self) -> None:
        original = Transaction(commit_id="old", sequence=1, created_ids=(1,), tick_id=0)
        payload = original.model_dump(mode="json", exclude={"recovery_root"})
        restored = Transaction.model_validate_json(json.dumps(payload))
        self.assertFalse(restored.recovery_root)
        self.assertEqual(CausalHistory(transactions={restored.commit_id: restored}).accepted(), set())

    def test_stored_root_with_speculative_dependency_fails_validation(self) -> None:
        speculative = Transaction(commit_id="unaccepted", sequence=1, created_ids=(1,), tick_id=0)
        root = Transaction(commit_id="invalid-root", sequence=2, input_ids=(1,),
                           dependencies=(speculative.commit_id,), created_ids=(2,), tick_id=0, recovery_root=True)
        history = CausalHistory(transactions={speculative.commit_id: speculative, root.commit_id: root})
        with self.assertRaisesRegex(ValueError, "unaccepted transactions"):
            history.accepted()
