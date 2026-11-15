"""Close private inquiries whose retained grant went stale, on a real store, without a model.

B07 Pass 0 step 4 found that an active inquiry whose goal, expectation, or
source got a successor stayed in the private slot while every later step was
refused at commit. These checks show the owner-chosen rule (B07 option A):
the work closes before any model call, and its question returns as a new
episode under a fresh grant.
"""
from __future__ import annotations

from pathlib import Path
from threading import Event
import unittest

from pm.memory.store import MemoryStore
from pm.records.base import RecordScope
from pm.records.cognitive_records import ExpectationRecord
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Feature
from pm.records.mental_state import FullMentalState
from tests.unit.workflows.inner_thought.test_inquiry_displacement import DisplacementFixture, forbidden_stage
from workflows.cognitive_control.attention import captured_private_focus
from workflows.focus_management.host import MentalOutcome, unfinished_notice
from workflows.inner_thought.expectations import open_expectation
from workflows.inner_thought.graph_execution import run_graph_opportunity
from workflows.inner_thought.graph_host import (
    STALE_OUTCOME_KEY, FrozenProjectionPayload, PrivateInquiryOwner, active_inquiry, attach_inner_projection, displaced_inquiries,
    parked_inquiries, pending_findings,
)
from workflows.inner_thought.graph_outcomes import attach_graph_outcome, capture_graph_outcome
from workflows.inner_thought.inquiry import (
    OUTDATED_REASON, InquiryRecord, displaced_offer, inquiry_history, stage_inquiry_outdated, stage_inquiry_step,
)
from workflows.inner_thought.nominations import capture_inquiry_grant
from workflows.inner_thought.schemas import (
    DisclosureChoice, ExpectationUpdate, InnerRequest, InquiryGrant, IntegrationResult,
)


class OutdatedInquiryTests(DisplacementFixture):
    """A changed grant source frees the slot; the question survives."""

    def replace_source(self) -> int:
        """Replace the governing source the way main cognition replaces a goal or expectation."""
        correction = self.store.begin_batch("main-replaces-source", embed=False)
        source = correction.view().get(self.source.id).copy_for_edit()
        source.content = "The sketch is finished."
        successor = correction.replace(source)
        self.store.commit(correction)
        return successor.id

    def observer_grant(self) -> InquiryGrant:
        """Capture a current opportunity grant that cites no replaced record."""
        return capture_inquiry_grant(self.store.snapshot(capture=False), observer_entity_id=self.observer.id,
            scope=RecordScope(), source_ids=[], critical_epoch=self.old_epoch)

    def test_an_opportunity_closes_outdated_work_before_any_model_call(self) -> None:
        working = self.working()
        self.replace_source()
        result = run_graph_opportunity(InnerRequest(), owner=self.owner, grant=self.observer_grant(),
            run_stage=forbidden_stage, mental_state=FullMentalState(), control_generation=0, pending_input_ids=(),
            focus=captured_private_focus(self.store.snapshot(capture=False), self.now))
        self.assertEqual(result.summary, "No private opportunity requires nomination.")
        restored = MemoryStore.load(self.path)
        view = restored.snapshot(capture=False)
        self.assertIsNone(active_inquiry(view, observer_entity_id=self.observer.id))
        head = inquiry_history(view, working.receipt.state.episode_id)[-1]
        self.assertEqual(head.receipt.parent_receipt_id, working.record.id)
        self.assertEqual(head.receipt.state.status, "displaced")
        self.assertEqual(head.receipt.state.stop_reason, OUTDATED_REASON)
        displacement = head.receipt.displacement
        assert displacement is not None
        self.assertEqual(displacement.cause, "governing_source_changed")
        self.assertEqual(displacement.changed_source_ids, (self.source.id,))
        self.assertEqual(displacement.superseding_critical_epoch, self.old_epoch)
        self.assertEqual(head.receipt.state.node_ids, working.receipt.state.node_ids)
        waiting = displaced_inquiries(view, observer_entity_id=self.observer.id)
        self.assertEqual([item.record.id for item in waiting], [head.record.id])
        offer = displaced_offer(view, waiting[0])
        assert offer is not None
        self.assertEqual(offer.displaced_reason, OUTDATED_REASON)
        # A second opportunity finds nothing left to close.
        self.assertEqual(self.owner.close_outdated(observer_entity_id=self.observer.id), ())

    def test_the_question_restarts_under_a_fresh_grant_and_can_execute_again(self) -> None:
        working = self.working()
        self.replace_source()
        closed = self.owner.close_outdated(observer_entity_id=self.observer.id)
        self.assertEqual(len(closed), 1)
        restarted = self.admit(closed[0], offered_grant=self.observer_grant())
        self.assertEqual(restarted.receipt.state.status, "active")
        self.assertNotEqual(restarted.receipt.state.episode_id, working.receipt.state.episode_id)
        granted = {source.record_id: source for source in restarted.receipt.state.grant.sources}
        # The replaced record stays citable as history behind the earlier nodes,
        # never again as current authority.
        self.assertFalse(granted[self.source.id].current_head)
        step = self.generation().model_copy(update={"operation": "feasibility_check", "alternatives": (),
            "next_operation": "future_projection"})
        accepted = self.owner.accept_step(restarted, step, grant=self.owner.execution_grant(restarted),
            commit_id="restarted-step", active_seconds=1, now_utc=self.now, mental_state=FullMentalState(),
            control_generation=0)
        self.assertEqual(accepted.receipt.state.sequence, 1)
        self.assertEqual(self.owner.close_outdated(observer_entity_id=self.observer.id), ())

    def test_current_work_is_not_closed(self) -> None:
        working = self.working()
        self.assertEqual(self.owner.close_outdated(observer_entity_id=self.observer.id), ())
        with self.assertRaisesRegex(ValueError, "all current is not outdated"):
            stage_inquiry_outdated(self.store.begin_batch("not-outdated", embed=False),
                                   parent_receipt_id=working.record.id)

    def test_parked_outdated_work_is_closed_too(self) -> None:
        self.owner.limits = self.limits.model_copy(update={"stalled_steps": 1})
        parked = self.working()
        self.assertEqual(parked.receipt.state.status, "paused")
        self.replace_source()
        closed = self.owner.close_outdated(observer_entity_id=self.observer.id)
        self.assertEqual([item.receipt.parent_receipt_id for item in closed], [parked.record.id])
        view = self.store.snapshot(capture=False)
        self.assertEqual(parked_inquiries(view, observer_entity_id=self.observer.id), ())

    def test_deliberate_notice_names_the_changed_ground(self) -> None:
        outcome = MentalOutcome(finding_id=None, mental_status="displaced", execution_status="cancelled",
                                stop_reason=OUTDATED_REASON, displacement_cause="governing_source_changed")
        notice = unfinished_notice("Compare the two endings.", outcome)
        self.assertIn("A goal or expectation it rested on changed", notice)
        critical = MentalOutcome(finding_id=None, mental_status="displaced", execution_status="cancelled",
                                 stop_reason="", displacement_cause="critical_epoch")
        self.assertIn("Something critical came up", unfinished_notice("Compare the two endings.", critical))


class StaleProjectionTests(DisplacementFixture):
    """A graph root accepted before a restart still projects once its outcome's source changed."""

    def expectation_inquiry(self) -> tuple[InquiryRecord, ExpectationRecord]:
        """Admit an inquiry whose grant governs one pending expectation."""
        batch = self.store.begin_batch("expectation", embed=False)
        expectation = open_expectation(batch, anticipation="Jesse finishes the sketch.",
            expected_outcome="Jesse says it is finished.", expected_valence=0.5, uncertainty=0.5,
            goal_ids=[], evidence_ids=[self.source.id])
        self.commit_root(batch)
        self.grant = capture_inquiry_grant(self.store.snapshot(capture=False), observer_entity_id=self.observer.id,
            scope=RecordScope(), source_ids=[self.source.id, expectation.id],
            governing_ids=[self.source.id, expectation.id], critical_epoch=self.old_epoch)
        return self.select("expectation-sketch"), expectation

    def test_projection_after_a_changed_outcome_source_keeps_the_finding_without_the_effect(self) -> None:
        selected, expectation = self.expectation_inquiry()
        step = self.generation(outcome="information_gap")
        grant = self.owner.execution_grant(selected)
        integration = IntegrationResult(summary="Jesse said the sketch is unfinished; the expectation stays open.",
            expectation=ExpectationUpdate(action="resolve", expectation_id=expectation.id,
                observation_ids=[self.source.id], status="contradicted", reason="The message says unfinished."),
            disclosure=DisclosureChoice(mode="withhold"))
        outcome = capture_graph_outcome(grant, step, integration)
        # The graph root commits; the process stops before its inner turn.
        root = self.store.begin_batch("graph-root-before-restart", new_tick=True, embed=False)
        accepted = stage_inquiry_step(root, parent_receipt_id=selected.record.id, step=step, grant=grant,
            active_seconds=1, now_utc=self.now, limits=self.limits)
        operation_id = accepted.receipt.operation_record_id
        assert operation_id is not None
        operation = root.view().get(operation_id)
        assert isinstance(operation, Feature)
        operation.metadata["private_inquiry_finding"] = True
        attach_inner_projection(operation, FrozenProjectionPayload(mental_state=FullMentalState()))
        attach_graph_outcome(operation, step, outcome)
        root.revise(operation)
        self.commit_root(root)
        # Main cognition then checks the expectation and replaces it.
        checked = self.store.begin_batch("main-partial-check", new_tick=True, embed=False)
        record = checked.view().get(expectation.id).copy_for_edit()
        assert isinstance(record, ExpectationRecord)
        record.fulfillment = 0.3
        main_version = checked.replace(record)
        # A real main check commits with its accepted main turn, which keeps it across restart.
        checked.set_gwt(GwtTurnRecord(tick_id=checked.tick_id, slot_id="main", knoxel_ids=[main_version.id],
                                      summary="Main cognition checked the expectation."),
                        CycleState(tick_id=checked.tick_id, mental_state=FullMentalState()))
        self.store.commit(checked)

        restored = MemoryStore.load(self.path)
        history = inquiry_history(restored.snapshot(capture=False), selected.receipt.state.episode_id)
        owner = PrivateInquiryOwner(restored, commit=restored.commit, cancel=Event(), limits=self.limits)
        owner.ensure_inner_turn(history[-1])
        turns = [turn for turn in restored.gwt_turns.values() if turn.slot_id == "inner" and operation_id in turn.knoxel_ids]
        self.assertEqual(len(turns), 1)
        view = restored.snapshot(capture=False)
        notes = [view.get(record_id) for record_id in turns[0].knoxel_ids
                 if STALE_OUTCOME_KEY in view.get(record_id).metadata]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].metadata[STALE_OUTCOME_KEY]["changed_source_ids"], [expectation.id])
        self.assertEqual(notes[0].metadata[STALE_OUTCOME_KEY]["expectation_action"], "resolve")
        current = [item for item in view.all_knoxels.values() if isinstance(item, ExpectationRecord)]
        self.assertEqual([(item.id, item.status) for item in current], [(main_version.id, "pending")])
        self.assertEqual([item.id for item in pending_findings(restored)], [operation_id])
        before = Path(self.path).read_bytes()
        owner.ensure_inner_turn(history[-1])
        self.assertEqual(Path(self.path).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
