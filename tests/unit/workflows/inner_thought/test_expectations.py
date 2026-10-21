"""Expectation lifecycle tests against the real MemoryStore commit boundary."""
from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from pm.memory.store import MemoryStore
from pm.records.base import Provenance
from pm.records.enums import SourceKind
from pm.records.cognitive_records import ExpectationRecord, Intention

from workflows.inner_thought.expectations import open_expectation, pending_expectations, resolve_expectation


def ghost() -> MemoryStore:
    from tests.unit.test_design_contracts import small_ghost
    return small_ghost()


def commit(ghost, action):
    return ghost.commit(action[0]) if isinstance(action, tuple) else action


class ExpectationLifecycleTests(unittest.TestCase):
    def test_open_expectation_commits_pending_record(self):
        ghost_ = ghost()
        with TemporaryDirectory() as tmp:
            ghost_.checkpoint = None
            batch = ghost_.begin_batch("open-1")
            record = open_expectation(
                batch, anticipation="Jesse replies tonight",
                expected_outcome="a friendly reply", expected_valence=0.4,
                uncertainty=0.3, goal_ids=[], evidence_ids=[1, 2],
            )
            ghost_.commit(batch)
            stored = ghost_.get_knoxel_by_id(record.id)
            self.assertIsInstance(stored, ExpectationRecord)
            self.assertEqual(stored.status, "pending")
            self.assertEqual(stored.evidence_ids, [1, 2])
            self.assertEqual(stored.provenance.source_kind, SourceKind.Self)
            self.assertEqual([item.id for item in pending_expectations(ghost_)], [stored.id])

    def test_resolve_records_observation_and_clears_pending(self):
        ghost_ = ghost()
        batch = ghost_.begin_batch("open-2")
        record = open_expectation(
            batch, anticipation="Jesse replies tonight", expected_outcome="a friendly reply",
            expected_valence=0.4, uncertainty=0.3, goal_ids=[], evidence_ids=[],
        )
        ghost_.commit(batch)
        revision = ghost_.revisions[record.id]
        batch = ghost_.begin_batch("resolve-1")
        resolved = resolve_expectation(batch, record, status="contradicted",
                                       observation_ids=[50, 51], reason="no reply came",
                                       revision=revision)
        ghost_.commit(batch)
        stored = ghost_.get_knoxel_by_id(record.id)
        self.assertEqual(stored.status, "contradicted")
        self.assertEqual(stored.observation_ids, [50, 51])
        self.assertEqual(stored.resolved_tick_id, stored.tick_id)
        self.assertEqual(pending_expectations(ghost_), [])
        self.assertEqual(resolved.status, "contradicted")

    def test_resolve_without_observations_is_rejected(self):
        ghost_ = ghost()
        batch = ghost_.begin_batch("open-3")
        record = open_expectation(
            batch, anticipation="x", expected_outcome="y",
            expected_valence=0.0, uncertainty=0.5, goal_ids=[], evidence_ids=[],
        )
        ghost_.commit(batch)
        batch = ghost_.begin_batch("resolve-2")
        with self.assertRaisesRegex(ValueError, "observation"):
            resolve_expectation(batch, record, status="supported",
                                 observation_ids=[], reason="", revision=1)

    def test_stale_resolution_conflicts_at_commit(self):
        from pm.memory.snapshot import GhostConflict
        ghost_ = ghost()
        batch = ghost_.begin_batch("open-4")
        record = open_expectation(
            batch, anticipation="x", expected_outcome="y",
            expected_valence=0.0, uncertainty=0.5, goal_ids=[], evidence_ids=[],
        )
        ghost_.commit(batch)
        revision = ghost_.revisions[record.id]
        # Another writer replaces the record first.
        other = ghost_.begin_batch("other-writer")
        replaced = record.model_copy(deep=True)
        replaced.metadata = {"note": "changed elsewhere"}
        other.expect_revision(record.id, revision)
        other.replace(replaced)
        ghost_.commit(other)
        batch = ghost_.begin_batch("resolve-3")
        with self.assertRaises(GhostConflict):
            resolve_expectation(batch, record, status="unmet",
                                 observation_ids=[9], reason="", revision=revision)
            ghost_.commit(batch)

    def test_pending_ignores_intention_records(self):
        from pm.records.enums import IntentionType
        ghost_ = ghost()
        ghost_.store_knoxel(Intention(
            content="keep trying", entity_id=0, affective_valence=0.1,
            intention_type=IntentionType.WorldState, status="active", active=True,
        ))
        self.assertEqual(pending_expectations(ghost_), [])


if __name__ == "__main__":
    unittest.main()
