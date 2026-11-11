"""Source guards against actual stores, lifecycle commits, and checkpoints."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.memory.dependencies import SourceDependency, capture_source, source_fingerprint
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.records.base import Provenance, RecordBase, RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType, FrameEnum, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.lineage import RecordLineage
from pm.records.memory_records import Feature, SemanticRecord
from pm.records.taxonomy import EpistemicStatus


class SourceDependencyTests(unittest.TestCase):
    """Exercise semantic guards without model inference or substitute services."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path: Path = Path(directory.name) / "ghost.pickle"
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=self.path)
        batch = self.store.begin_batch("original", embed=False)
        self.source: Feature = batch.create(Feature(content="The codeword is copper lantern.", feature_type=FeatureType.Dialogue))
        batch.preserve_on_recovery()
        self.store.commit(batch)

    def capture(self, *, current_head: bool = True) -> SourceDependency:
        return capture_source(self.store.snapshot(), self.source.id, current_head=current_head)

    def test_fingerprint_excludes_only_explicit_bookkeeping_and_embeddings(self) -> None:
        original_fingerprint = source_fingerprint(self.source)
        for key in ("exchange_status", "public_reply_id", "publication_status", "recovery_outcome",
                    "dispatch_status", "execution_status"):
            with self.subTest(key=key):
                changed = self.source.copy_for_edit()
                changed.metadata[key] = "completed"
                self.assertEqual(source_fingerprint(changed), original_fingerprint)
        embedded = self.source.copy_for_edit()
        embedded.embedding = [0.4, 0.8]
        embedded.embedding_model = "derived-vector"
        self.assertEqual(source_fingerprint(embedded), original_fingerprint)

    def test_fingerprint_preserves_content_scope_provenance_time_and_output(self) -> None:
        changed_content = self.source.copy_for_edit()
        changed_content.content = "The codeword is silver birch."
        changed_scope = self.source.copy_for_edit()
        changed_scope.scope = RecordScope(frame=FrameEnum.Meta)
        changed_reality = self.source.copy_for_edit()
        changed_reality.scope = RecordScope(reality_id=42)
        changed_provenance = self.source.copy_for_edit()
        changed_provenance.provenance = Provenance(source_kind=SourceKind.User, source_ref="Jesse")
        changed_creation_time = self.source.copy_for_edit()
        changed_creation_time.timestamp_creation += timedelta(seconds=1)
        changed_world_begin = self.source.copy_for_edit()
        changed_world_begin.timestamp_world_begin += timedelta(seconds=1)
        changed_world_end = self.source.copy_for_edit()
        changed_world_end.timestamp_world_end += timedelta(seconds=1)
        changed_lineage = self.source.copy_for_edit()
        changed_lineage.lineage = RecordLineage(episode_id="evidence", operation="source")
        changed_output = self.source.copy_for_edit()
        changed_output.metadata["output_record_ids"] = [42]
        variants: list[RecordBase] = [changed_content, changed_scope, changed_reality, changed_provenance,
                                    changed_creation_time, changed_world_begin, changed_world_end,
                                    changed_lineage, changed_output]
        for index, changed in enumerate(variants):
            with self.subTest(index=index):
                self.assertNotEqual(source_fingerprint(changed), source_fingerprint(self.source))
        claim = SemanticRecord(content="A claim", confidence=0.7)
        revised_claim = claim.copy_for_edit()
        revised_claim.epistemic_status = EpistemicStatus.Inferred
        self.assertNotEqual(source_fingerprint(claim), source_fingerprint(revised_claim))

    def test_lifecycle_revision_survives_checkpoint_without_invalidating_meaning(self) -> None:
        dependency = self.capture()
        proposal = self.store.begin_batch("selection", embed=False)
        proposal.expect_source(dependency)
        selected = proposal.create(Feature(content="Accepted selection", feature_type=FeatureType.Thought))
        proposal.preserve_on_recovery()
        lifecycle = self.store.begin_batch("completion", embed=False, new_tick=True)
        lifecycle.set_lifecycle(self.source.id, execution_status="completed", exchange_status="resolved")
        lifecycle.set_gwt(GwtTurnRecord(tick_id=lifecycle.tick_id, knoxel_ids=[self.source.id]))
        lifecycle_receipt = self.store.commit(lifecycle)
        self.assertEqual(self.store.revisions[self.source.id], dependency.revision + 1)
        self.assertEqual(self.store.commit(lifecycle), lifecycle_receipt)
        self.assertEqual(self.store.revisions[self.source.id], dependency.revision + 1)
        self.store.commit(proposal)
        reopened = MemoryStore.from_checkpoint(self.path)
        self.assertEqual(reopened.revisions[self.source.id], 2)
        self.assertEqual(reopened.snapshot().get(self.source.id).metadata["execution_status"], "completed")
        self.assertEqual(reopened.snapshot().get(selected.id).content, "Accepted selection")
        resumed = reopened.begin_batch("resumed-selection", embed=False)
        resumed.expect_source(dependency)
        reopened.commit(resumed)
        self.assertEqual(dependency.fingerprint, source_fingerprint(reopened.snapshot().get(self.source.id)))

    def test_raw_revision_guard_stays_strict_in_either_registration_order(self) -> None:
        dependency = self.capture()
        for semantic_first in (True, False):
            with self.subTest(semantic_first=semantic_first):
                proposal = self.store.begin_batch(f"strict-{semantic_first}", embed=False)
                if semantic_first:
                    proposal.expect_source(dependency)
                proposal.expect_revision(dependency.record_id, dependency.revision)
                if not semantic_first:
                    proposal.expect_source(dependency)
                proposed = proposal.create(Feature(content="Must not be accepted", feature_type=FeatureType.Thought))
                lifecycle = self.store.begin_batch(f"lifecycle-{semantic_first}", embed=False)
                lifecycle.set_lifecycle(self.source.id, execution_status="completed")
                self.store.commit(lifecycle)
                with self.assertRaises(GhostConflict):
                    self.store.commit(proposal)
                self.assertIsNone(self.store.get_knoxel_by_id(proposed.id))
                self.assertNotIn(proposal.commit_id, self.store.applied_commits)

    def test_meaning_change_rejects_original_fingerprint_even_with_fresh_batch(self) -> None:
        dependency = self.capture()
        mutation = self.store.begin_batch("interpretation-change", embed=False)
        mutation.set_lifecycle(self.source.id, evidence_role="hypothetical")
        self.store.commit(mutation)
        proposal = self.store.begin_batch("fresh-batch-old-source", embed=False)
        proposal.expect_source(dependency)
        proposed = proposal.create(Feature(content="Stale interpretation", feature_type=FeatureType.Thought))
        with self.assertRaisesRegex(GhostConflict, "Stale source meanings"):
            self.store.commit(proposal)
        self.assertIsNone(self.store.get_knoxel_by_id(proposed.id))
        reopened = MemoryStore.from_checkpoint(self.path, raw=True)
        self.assertIsNone(reopened.get_knoxel_by_id(proposed.id))
        self.assertEqual(dependency.fingerprint, source_fingerprint(self.source))

    def test_successor_rejects_current_authority_but_preserves_explicit_history(self) -> None:
        current = self.capture()
        historical = self.capture(current_head=False)
        successor_batch = self.store.begin_batch("correction", embed=False)
        successor = self.source.copy_for_edit()
        successor.content = "The codeword is silver birch."
        successor_batch.replace(successor)
        self.store.commit(successor_batch)
        proposal = self.store.begin_batch("stale-head", embed=False)
        proposal.expect_source(current)
        with self.assertRaises(GhostConflict):
            self.store.commit(proposal)
        historical_batch = self.store.begin_batch("historical-evidence", embed=False)
        historical_batch.expect_source(historical)
        self.store.commit(historical_batch)
        self.assertIn("historical-evidence", self.store.applied_commits)
        with self.assertRaises(ValueError):
            capture_source(self.store.snapshot(), self.source.id, current_head=True)

    def test_lifecycle_on_new_record_and_idempotent_commit_advance_once(self) -> None:
        batch = self.store.begin_batch("new-status", embed=False)
        created = batch.create(Feature(content="New action", feature_type=FeatureType.Thought))
        batch.set_lifecycle(created.id, execution_status="completed")
        first_receipt = self.store.commit(batch)
        self.assertEqual(self.store.revisions[created.id], 1)
        self.assertEqual(self.store.commit(batch), first_receipt)
        self.assertEqual(self.store.revisions[created.id], 1)
        reopened = MemoryStore.from_checkpoint(self.path, raw=True)
        self.assertEqual(reopened.revisions[created.id], 1)
        self.assertEqual(reopened.snapshot().get(created.id).metadata["execution_status"], "completed")

    def test_unrelated_append_does_not_invalidate_source(self) -> None:
        dependency = self.capture()
        proposal = self.store.begin_batch("unrelated-safe", embed=False)
        proposal.expect_source(dependency)
        unrelated = self.store.begin_batch("other-input", embed=False)
        unrelated.create(Feature(content="Unrelated conversation", feature_type=FeatureType.Dialogue))
        self.store.commit(unrelated)
        self.store.commit(proposal)
        self.assertIn("unrelated-safe", self.store.applied_commits)

    def test_dependency_outside_batch_grant_fails_before_commit(self) -> None:
        dependency = self.capture()
        restricted = self.store.begin_batch("restricted", visible_ids=[], embed=False)
        with self.assertRaises(ValueError):
            restricted.expect_source(dependency)
