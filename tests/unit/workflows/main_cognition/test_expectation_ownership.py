"""Main cognition owns the expectation check; one observation moves an expectation once.

Real store, no model (B07 Pass 2, decisions D2 and D3). Each check that would
need a model call is shown to make none: the guard embedder below fails if it
is ever asked for a vector.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.memory.dependencies import capture_source, changed_dependencies
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.cognitive_records import ExpectationRecord
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.enums import EntityClass, FeatureType, FrameEnum, InterlocusType, SourceKind
from pm.records.lineage import RecordLineage
from pm.records.memory_records import Entity, Feature
from workflows.inner_thought.expectations import open_expectation, resolve_expectation
from workflows.main_cognition.cycle_commits import apply_goal_changes
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.expectations import (
    SOURCE_REF, is_observed, offered_expectations, outcome_of, relevant_expectations, uncounted_stimuli,
)
from workflows.main_cognition.host import apply_expectation_outcomes
from workflows.main_cognition.schemas import GoalChange, OpenExpectation, Stimulus


class NoEmbedding:
    """Fail the check if an expectation that has nothing new to rate reaches the embedding endpoint."""

    def prefetch(self, texts: list[str]) -> dict[str, list[float]]:
        raise AssertionError(f"No embedding request is allowed here: {texts}")


class ExpectationOwnershipTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
                                 checkpoint=Path(directory.name) / "expectations.pickle")
        batch = self.store.begin_batch("inputs", embed=False)
        self.observer = batch.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.jesse = batch.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.reply = batch.create(Feature(content="Yes, let's finish the story tomorrow.",
            feature_type=FeatureType.Dialogue, causal=True, entity_id=self.jesse.id,
            addressee_entity_id=self.observer.id, interlocus=InterlocusType.Public,
            provenance=Provenance(source_kind=SourceKind.User), metadata={"memory_kind": "utterance"}))
        self.thought = batch.create(Feature(content="Jesse said yes to finishing the story.",
            feature_type=FeatureType.InnerMonologue, entity_id=self.observer.id,
            interlocus=InterlocusType.PrivateInternal, provenance=Provenance(source_kind=SourceKind.Self),
            metadata={"memory_kind": "interpretation"},
            lineage=RecordLineage(episode_id="private", operation="expectation_review",
                                  source_event_ids=[self.reply.id])))
        self.main_expectation = batch.create(ExpectationRecord(content="Jesse agrees to finish the story.",
            anticipation="Jesse agrees to finish the story.", expected_outcome="Jesse agrees to finish the story.",
            expected_valence=0.6, urgency=0.6, provenance=Provenance(source_kind=SourceKind.Self, source_ref=SOURCE_REF)))
        self.private_expectation = open_expectation(batch, anticipation="Jesse wants to keep writing together.",
            expected_outcome="Jesse proposes another session.", expected_valence=0.5, uncertainty=0.5,
            goal_ids=[], evidence_ids=[self.reply.id])
        imagined = open_expectation(batch, anticipation="The Minecraft server stays online tonight.",
            expected_outcome="The server stays up.", expected_valence=0.4, uncertainty=0.5, goal_ids=[], evidence_ids=[])
        imagined.scope = RecordScope(frame=FrameEnum.Meta)
        batch.revise(imagined)
        self.imagined_expectation = imagined
        batch.preserve_on_recovery()
        self.store.commit(batch)

    def work(self, stimulus_ids: list[int]) -> CycleWork:
        """Build one cycle's request the way the host does: offered expectations and observed stimuli."""
        view = self.store.snapshot(capture=False)
        stimuli = [Stimulus(knoxel_id=record_id, text=view.get(record_id).content) for record_id in stimulus_ids]
        observed = [record_id for record_id in stimulus_ids if is_observed(view, view.get(record_id))]
        request = CycleRequest(stimuli=stimuli, open_expectations=offered_expectations(view),
                               observed_stimulus_ids=observed)
        return CycleWork(request=request, state=CycleState(tick_id=self.store.current_tick_id))

    def offered(self, work: CycleWork, record_id: int) -> OpenExpectation:
        return next(item for item in work.request.open_expectations if item.record_id == record_id)

    def commit_check(self, work: CycleWork) -> None:
        """Stage the check results through the main commit function and accept them."""
        changes: WriteBatch = self.store.begin_batch("main-check", new_tick=True, embed=False)
        apply_expectation_outcomes(changes, changes.view(), work)
        self.store.commit(changes)

    def head(self, content: str) -> ExpectationRecord:
        heads = [item for item in self.store.all_knoxels.values()
                 if isinstance(item, ExpectationRecord) and item.content == content]
        self.assertEqual(len(heads), 1)
        return heads[0]

    def test_every_pending_expectation_of_the_accepted_reality_is_offered_with_its_origin(self) -> None:
        offered = offered_expectations(self.store.snapshot(capture=False))
        self.assertEqual([(item.record_id, item.origin) for item in offered],
                         [(self.main_expectation.id, "main_cognition"), (self.private_expectation.id, "private_loop")])
        self.assertTrue(all(item.counted_observation_ids == [] for item in offered))

    def test_only_observed_stimuli_enter_the_check(self) -> None:
        view = self.store.snapshot(capture=False)
        self.assertTrue(is_observed(view, view.get(self.reply.id)))
        self.assertFalse(is_observed(view, view.get(self.thought.id)))
        # A thought that restates the reply moves no expectation and costs no call.
        work = self.work([self.thought.id])
        self.assertEqual(work.request.observed_stimulus_ids, [])
        self.assertEqual(relevant_expectations(work, NoEmbedding()), [])

    def test_the_same_observation_moves_an_expectation_once(self) -> None:
        first = self.work([self.reply.id])
        expectation = self.offered(first, self.main_expectation.id)
        self.assertEqual([item.knoxel_id for item in uncounted_stimuli(first, expectation)], [self.reply.id])
        # A partial fulfilment: the expectation moves and stays pending. The
        # private expectation is rated too and does not move.
        first.expectations = [outcome_of(expectation, 0.6, 0.8, [self.reply.id]),
                              outcome_of(self.offered(first, self.private_expectation.id), 0.5, 0.0, [self.reply.id])]
        self.commit_check(first)
        moved = self.head(self.main_expectation.content)
        self.assertEqual(moved.status, "pending")
        self.assertEqual(moved.counted_observation_ids, [self.reply.id])
        self.assertGreater(moved.fulfillment, 0.0)
        # The same observation is presented again (Critical requeue, wait flush, restart).
        second = self.work([self.reply.id])
        again = self.offered(second, moved.id)
        self.assertEqual(uncounted_stimuli(second, again), [])
        # Nothing is left to rate, so the check makes no embedding and no model call.
        self.assertEqual(relevant_expectations(second, NoEmbedding()), [])
        second.expectations = [outcome_of(again, 1.0, 1.0, [self.reply.id])]
        changes = self.store.begin_batch("double-count", new_tick=True, embed=False)
        with self.assertRaisesRegex(ValueError, "already counted"):
            apply_expectation_outcomes(changes, changes.view(), second)

    def test_a_rating_that_does_not_move_the_expectation_still_counts_the_observation(self) -> None:
        work = self.work([self.reply.id])
        expectation = self.offered(work, self.main_expectation.id)
        work.expectations = [outcome_of(expectation, 0.5, 0.0, [self.reply.id])]
        self.commit_check(work)
        unmoved = self.head(self.main_expectation.content)
        self.assertEqual((unmoved.fulfillment, unmoved.counted_observation_ids), (0.0, [self.reply.id]))

    def test_a_private_expectation_is_resolved_once_and_the_late_private_resolution_is_refused(self) -> None:
        # The private loop read the expectation with current authority before main checked it.
        captured = capture_source(self.store.snapshot(capture=False), self.private_expectation.id, current_head=True)
        old_record = self.store.snapshot(capture=False).get(self.private_expectation.id)
        assert isinstance(old_record, ExpectationRecord)
        old_revision = self.store.revisions[self.private_expectation.id]
        work = self.work([self.reply.id])
        expectation = self.offered(work, self.private_expectation.id)
        work.expectations = [outcome_of(expectation, 1.0, 1.0, [self.reply.id])]
        self.commit_check(work)
        resolved = self.head(self.private_expectation.content)
        self.assertEqual(resolved.status, "supported")
        self.assertEqual(resolved.observation_ids, [self.reply.id])
        # The old payload still reads pending; only the head guard sees the change.
        self.assertEqual(old_record.status, "pending")
        late = self.store.begin_batch("late-private-resolution", embed=False)
        resolve_expectation(late, old_record, status="supported", observation_ids=[self.reply.id],
                            reason="The reply agrees.", revision=old_revision)
        with self.assertRaises(GhostConflict):
            self.store.commit(late)
        # The private loop sees the same change before its next model call and
        # closes that inquiry (B07 Q1) instead of repeating the refused commit.
        self.assertEqual(changed_dependencies(self.store.snapshot(capture=False), [captured]),
                         (self.private_expectation.id,))
        supported = [item for item in self.store.all_knoxels.values()
                     if isinstance(item, ExpectationRecord) and item.status == "supported"]
        self.assertEqual(len(supported), 1)

    def test_an_expectation_record_from_before_b07_reads_as_no_check_history(self) -> None:
        stored = self.main_expectation.model_dump(mode="json")
        del stored["counted_observation_ids"]
        loaded = ExpectationRecord.model_validate(stored)
        self.assertEqual(loaded.counted_observation_ids, [])

    def test_completing_a_goal_twice_completes_it_once(self) -> None:
        add = self.store.begin_batch("add-goal", embed=False)
        apply_goal_changes(add, add.view(), [GoalChange(change="add", goal="Finish the story with Jesse",
                                                        goal_key="finish-story", reason="We agreed.")])
        self.store.commit(add)
        complete = GoalChange(change="complete", goal="Finish the story with Jesse", goal_key="finish-story",
                              reason="The story is finished.")
        first = self.store.begin_batch("complete-1", embed=False)
        receipts = apply_goal_changes(first, first.view(), [complete])
        self.store.commit(first)
        second = self.store.begin_batch("complete-2", embed=False)
        again = apply_goal_changes(second, second.view(), [complete])
        self.store.commit(second)
        self.assertEqual([receipts[0]["result"], again[0]["result"]], ["completed", "missing"])
        completed = [item for item in self.store.all_knoxels.values()
                     if type(item).__name__ == "Intention" and item.metadata.get("goal_key") == "finish-story"]
        self.assertEqual([item.status for item in completed], ["completed"])


if __name__ == "__main__":
    unittest.main()
