"""The interval review shows each action once, under the goal version that held then.

Real store, no model (B07 Pass 3, decision D4).
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.memory.dependencies import capture_source
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.cognitive_records import ExpectationRecord, Intention
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.enums import ActionType, EntityClass, FeatureType, InterlocusType, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord, Entity, Feature
from pm.records.mental_state import FullMentalState
from workflows.interaction.schemas import InteractionDecision, InteractionOutcome, InteractionReceipt
from workflows.main_cognition.cycle_commits import apply_goal_changes
from workflows.main_cognition.expectations import SOURCE_REF
from workflows.main_cognition.interval_review import build_interval_review, render_interval_review
from workflows.main_cognition.schemas import GoalChange
from workflows.public_speech.dependencies import WorkDependencies
from workflows.public_speech.situational_guidance import GuidanceDisposition


class IntervalReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "review.pickle"
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=self.path)
        batch = self.store.begin_batch("anchors", embed=False)
        self.observer = batch.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.jesse = batch.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.message = batch.create(Feature(content="Can we finish the story tomorrow?", causal=True,
            feature_type=FeatureType.Dialogue, entity_id=self.jesse.id, addressee_entity_id=self.observer.id,
            interlocus=InterlocusType.Public, provenance=Provenance(source_kind=SourceKind.User)))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        self.goal_v1 = self.change_goal("add")

    def change_goal(self, kind: str) -> int:
        """Apply one goal change through the main commit function; return the active or final head id."""
        batch = self.store.begin_batch(f"goal-{kind}-{self.store.current_knoxel_id}", embed=False)
        receipts = apply_goal_changes(batch, batch.view(), [GoalChange(change=kind, goal="Finish the story with Jesse",
            goal_key="finish-story", reason="The story matters.", priority=0.8 if kind == "reprioritize" else 0.5)])
        self.store.commit(batch)
        heads = [record for record in self.store.all_knoxels.values()
                 if isinstance(record, Intention) and record.metadata.get("goal_key") == "finish-story"]
        self.assertEqual(len(heads), 1, receipts)
        return heads[0].id

    def act(self, content: str, *, intention_ids: list[int]) -> ActionSchemaRecord:
        """Commit one main action with its governing versions and one expectation, as the main commit does."""
        batch = self.store.begin_batch(f"act-{content[:8]}", embed=False)
        action = batch.create(ActionSchemaRecord(content=content, action_type=ActionType.Reply,
            intention_ids=list(intention_ids),
            metadata={"action_id": f"act:{content}", "dispatch_status": "pending",
                      "dispatch_payload": {"action_origin": "main"}}))
        batch.create(ExpectationRecord(content="Jesse says yes.", created_from_action_id=action.id,
            goal_ids=list(intention_ids), provenance=Provenance(source_kind=SourceKind.Self, source_ref=SOURCE_REF)))
        self.store.commit(batch)
        return action

    def frontier(self) -> int:
        return max(self.store.snapshot(capture=False).revisions)

    def accept_main_cycle(self, batch: WriteBatch, *, through: int) -> None:
        """Commit a main turn whose state records the review frontier, as ``_accepted_state`` does."""
        batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, slot_id="main", knoxel_ids=[self.message.id],
                                    summary="Main cognition took stock."),
                      CycleState(tick_id=batch.tick_id, mental_state=FullMentalState(), interval_reviewed_through=through))
        self.store.commit(batch)

    def latest_main_frontier(self, store: MemoryStore) -> int | None:
        main_ticks = {tick for tick, turn in store.gwt_turns.items() if turn.slot_id == "main"}
        return [state for state in store.read_cycles() if state.tick_id in main_ticks][-1].interval_reviewed_through

    def test_an_action_keeps_the_goal_version_it_was_chosen_under(self) -> None:
        start = self.frontier()
        action = self.act("Ask Jesse whether tomorrow works.", intention_ids=[self.goal_v1])
        goal_v2 = self.change_goal("reprioritize")
        self.assertNotEqual(goal_v2, self.goal_v1)
        review = build_interval_review(self.store.snapshot(capture=False), after_record_id=start)
        self.assertEqual([item.move.action.record_id for item in review.moves], [action.id])
        governing = review.moves[0].governing[0]
        self.assertEqual((governing.intention_id, governing.text, governing.later_state, governing.current_id),
                         (self.goal_v1, "Finish the story with Jesse", "changed", goal_v2))
        self.assertEqual(review.moves[0].move.status, "selected")
        self.assertEqual(len(review.moves[0].expectation_ids), 1)
        self.change_goal("complete")
        completed = build_interval_review(self.store.snapshot(capture=False), after_record_id=start)
        self.assertEqual(completed.moves[0].governing[0].later_state, "completed")
        self.assertIsNone(completed.moves[0].governing[0].current_id)
        text = render_interval_review(completed, companion_name="Spamuel")
        self.assertIn(f"[action {action.id}]", text)
        self.assertIn(f"goal then [{self.goal_v1}]: Finish the story with Jesse (since completed)", text)

    def test_an_accepted_cycle_moves_the_frontier_and_the_next_review_does_not_repeat_the_action(self) -> None:
        start = self.frontier()
        action = self.act("Ask Jesse whether tomorrow works.", intention_ids=[self.goal_v1])
        first = build_interval_review(self.store.snapshot(capture=False), after_record_id=start)
        self.assertEqual([item.move.action.record_id for item in first.moves], [action.id])
        self.accept_main_cycle(self.store.begin_batch("main-1", new_tick=True, embed=False),
                               through=first.through_record_id)
        self.assertEqual(self.latest_main_frontier(self.store), first.through_record_id)
        second = build_interval_review(self.store.snapshot(capture=False), after_record_id=first.through_record_id)
        self.assertEqual(second.moves, [])

    def test_a_cycle_that_is_not_accepted_leaves_the_frontier_unchanged(self) -> None:
        start = self.frontier()
        self.accept_main_cycle(self.store.begin_batch("main-0", new_tick=True, embed=False), through=start)
        action = self.act("Ask Jesse whether tomorrow works.", intention_ids=[self.goal_v1])
        review = build_interval_review(self.store.snapshot(capture=False), after_record_id=start)
        refused = self.store.begin_batch("main-refused", new_tick=True, embed=False)
        refused.expect_source(capture_source(refused.view(), self.goal_v1, current_head=True))
        self.change_goal("reprioritize")  # A concurrent goal change makes the cycle stale.
        with self.assertRaises(GhostConflict):
            self.accept_main_cycle(refused, through=review.through_record_id)
        self.assertEqual(self.latest_main_frontier(self.store), start)
        again = build_interval_review(self.store.snapshot(capture=False), after_record_id=start)
        self.assertEqual([item.move.action.record_id for item in again.moves], [action.id])

    def test_a_set_aside_guidance_appears_once_with_its_reason(self) -> None:
        start = self.frontier()
        view = self.store.snapshot(capture=False)
        receipt = InteractionReceipt(selection_id="select-1", source_ids=[self.message.id],
            outcome=InteractionOutcome.Quiet,
            decision=InteractionDecision(variants=[], guidance_disposition=GuidanceDisposition.Suspend,
                guidance_reason="Jesse is busy right now, so the planned question waits."),
            dependencies=WorkDependencies(sources=[capture_source(view, self.message.id, current_head=True)],
                input_commit_ids=[], guidance_tick=0, critical_epoch=0, partner_entity_id=self.jesse.id,
                scopes=[RecordScope()], source_ids=[self.message.id], latest_partner_input_id=self.message.id))
        batch = self.store.begin_batch("interaction", embed=False)
        record = batch.create(Feature(content="Ordinary character choice: wait.", feature_type=FeatureType.StateUpdate,
            interlocus=InterlocusType.PrivateInternal, provenance=Provenance(source_kind=SourceKind.Self, source_ref="interaction"),
            metadata={"interaction_receipt": receipt.model_dump(mode="json"), "exchange_correlation": "c-1"}))
        self.store.commit(batch)
        review = build_interval_review(self.store.snapshot(capture=False), after_record_id=start)
        self.assertEqual([(item.receipt_id, item.disposition, item.reason) for item in review.suspensions],
                         [(record.id, GuidanceDisposition.Suspend, "Jesse is busy right now, so the planned question waits.")])
        self.assertIn("set the accepted direction aside (suspend)", render_interval_review(review, companion_name="Spamuel"))
        later = build_interval_review(self.store.snapshot(capture=False), after_record_id=review.through_record_id)
        self.assertEqual(later.suspensions, [])

    def test_a_state_without_a_frontier_starts_at_the_current_head_and_reviews_nothing(self) -> None:
        self.act("Ask Jesse whether tomorrow works.", intention_ids=[self.goal_v1])
        review = build_interval_review(self.store.snapshot(capture=False), after_record_id=None)
        self.assertTrue(review.first_review)
        self.assertEqual((review.moves, review.suspensions), ([], []))
        self.assertEqual(review.through_record_id, self.frontier())
        self.assertIn("nothing", render_interval_review(review, companion_name="Spamuel"))
        # A checkpoint state from before B07 carries no frontier.
        self.assertIsNone(CycleState.model_validate({"tick_id": 1}).interval_reviewed_through)

    def test_a_clean_restart_keeps_the_frontier(self) -> None:
        through = self.frontier()
        self.accept_main_cycle(self.store.begin_batch("main-1", new_tick=True, embed=False), through=through)
        restored = MemoryStore.load(self.path)
        self.assertEqual(self.latest_main_frontier(restored), through)


if __name__ == "__main__":
    unittest.main()
