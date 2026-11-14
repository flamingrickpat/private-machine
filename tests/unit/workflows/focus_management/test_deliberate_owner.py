"""Exercise the deliberate owner against real stores, a real bus, and durable checkpoints.

The action records are synthetic storage inputs standing in for a dispatch that
main cognition accepted. The inquiry steps are explicit boundary inputs standing
in for the private loop's model stages. No model, agent, or substitute service
runs here, so these checks establish admission, parking, lifecycle, and delivery
mechanics only. They make no claim about generated thought quality; the recorded
``deliberate-entry`` scenario covers the real child kernels.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
from typing import Literal
import unittest

from pm import vclock
from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, new_envelope
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordLineage, RecordScope
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.enums import ActionType, EntityClass, FeatureType, InterlocusType, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord, Entity, Feature
from pm.records.mental_state import FullMentalState
from tests.support import Recorder
from workflows.cognitive_control.host import CognitiveControl
from workflows.cognitive_control.schemas import ControlConfig, FocusProposal, MentalPlan
from workflows.focus_management.host import FocusManagement, MentalActionLifecycle
from workflows.inner_thought.graph_execution import (
    DeliberateBudget, graph_thought_instruction, graph_thought_request, validate_graph_step,
)
from workflows.inner_thought.graph_host import (
    PrivateInquiryOwner, active_inquiry, deliberate_inquiry, pending_findings,
)
from workflows.inner_thought.pipeline import ModelProposalError
from workflows.inner_thought.graph_outcomes import GraphOutcomePayload, capture_graph_outcome
from workflows.inner_thought.inquiry import (
    PREEMPTED_REASON, InquiryRecord, inquiry_history, inquiry_origin, parked_offer, stage_attention,
)
from workflows.inner_thought.nominations import capture_inquiry_grant, stage_nominations
from workflows.inner_thought.schemas import (
    InnerRequest, InquiryGrant, InquiryProgress, InquiryResumeDecision, InquiryStep, IntegrationResult,
    ThoughtAttentionDecision, ThoughtJudgments, ThoughtNomination,
)
from workflows.input_triage.schemas import EarlyTriageAssessment, TriageEvidence
from workflows.main_cognition.schemas import ActionArguments, ActionRequest, Stimulus, TriageItem, TriageResult
from workflows.public_speech.public_context import source_fingerprint
from workflows.public_speech.user_input import UserInputProxy


class DeliberateOwnerCase(unittest.TestCase):
    """Shared real store, bus, and boundary inputs for the deliberate owner checks."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path: Path = Path(directory.name) / "deliberate-owner.pickle"
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
                                              checkpoint=self.path)
        self.bus: BusSystem = BusSystem()
        self.thoughts: Recorder = Recorder(self.bus, "thought.created")
        self.receipts: Recorder = Recorder(self.bus, "action.receipt")
        self.deferrals: Recorder = Recorder(self.bus, "mental.deferred")
        self.now: datetime = datetime(2026, 11, 14, 9, 0)
        anchors = self.store.begin_batch("anchors", embed=False)
        self.observer: Entity = anchors.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.partner: Entity = anchors.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.source: Feature = anchors.create(Feature(content="I stopped talking about the story.", causal=True,
            feature_type=FeatureType.Dialogue, entity_id=self.partner.id, addressee_entity_id=self.observer.id,
            interlocus=InterlocusType.Public, provenance=Provenance(source_kind=SourceKind.User)))
        self.commit(anchors)
        self.judgments: ThoughtJudgments = ThoughtJudgments(interestingness=0.5, character_relevance=0.7,
            plausibility=0.2, consequence=0.4, desirability=0.5, action_suitability=0.3,
            reason="A withdrawn topic needs explaining.")
        self.module: FocusManagement = self.build_module(self.store)

    def module_error(self, error: BaseException) -> None:
        """Surface a module error in the test instead of losing it on a bus thread."""
        raise error

    def commit(self, batch: WriteBatch) -> None:
        """Persist an append-only record through an actual clean recovery root."""
        batch.preserve_on_recovery()
        self.store.commit(batch)

    def build_module(self, store: MemoryStore) -> FocusManagement:
        """Construct the real deliberate owner over a store, without a control module."""
        return FocusManagement("focus_management", self.bus, self.module_error, time.monotonic, None, ghost=store)

    def private_owner(self, store: MemoryStore, *, limits: ControlConfig | None = None) -> PrivateInquiryOwner:
        """Build the private loop's owner, which commits the executed steps."""

        def commit_step(changes: WriteBatch) -> None:
            """Commit graph and projection batches exactly as staged."""
            store.commit(changes)

        return PrivateInquiryOwner(store, commit=commit_step, cancel=Event(), limits=limits or ControlConfig())

    def dispatch(self, instruction: str, *, operation: str = "", correlation: str = "corr-story",
                 dispatch_status: str = "pending", workflow_name: str = "focus_management",
                 partner_entity_id: int | None = None) -> tuple[ActionSchemaRecord, Envelope]:
        """Commit one accepted mental action and build the request main cognition publishes for it."""
        request = ActionRequest(action_id=f"action-{correlation}", workflow_name=workflow_name,
            content="I want to think this through on my own.", partner_entity_id=partner_entity_id,
            arguments=ActionArguments(instruction=instruction, operation=operation) if operation
            else ActionArguments(instruction=instruction),
            stimuli=[Stimulus(knoxel_id=self.source.id, source="user_input", text=self.source.content)])
        batch = self.store.begin_batch(f"dispatch-{correlation}", embed=False)
        action = batch.create(ActionSchemaRecord(content=request.content, action_type=ActionType.Loopback,
            provenance=Provenance(source_kind=SourceKind.Workflow),
            lineage=RecordLineage(episode_id=f"dispatch:{correlation}", operation="action_dispatch",
                parent_record_ids=[self.source.id]),
            metadata={"action_id": request.action_id, "workflow_name": workflow_name,
                      "dispatch_status": dispatch_status, "dispatch_target": workflow_name,
                      "dispatch_correlation": correlation, "evidence_ids": [self.source.id]}))
        self.commit(batch)
        payload = request.model_copy(update={"knoxel_id": action.id}).payload()
        envelope = new_envelope("main_cog", "data", "action.request", correlation, payload, target=workflow_name)
        return action, envelope

    def spontaneous(self, episode_id: str = "spontaneous-story") -> InquiryRecord:
        """Give the private slot to a spontaneous nomination, as shared attention would."""
        grant = capture_inquiry_grant(self.store.snapshot(capture=False), observer_entity_id=self.observer.id,
            scope=RecordScope(), source_ids=[self.source.id], critical_epoch=0)
        batch = self.store.begin_batch("nomination-" + episode_id, embed=False)
        staged = stage_nominations(batch, [ThoughtNomination(text="The withdrawn topic may hide a concern.",
            question="Why did the topic disappear?", evidence_ids=(self.source.id,), reality="inferred",
            judgments=self.judgments, suggested_operation="perspective_shift")], grant=grant, batch_id=episode_id)
        self.commit(batch)
        attention = self.store.begin_batch("attention-" + episode_id, embed=False)
        selected = stage_attention(attention, [staged[0].record.id],
            ThoughtAttentionDecision(nomination_id=staged[0].record.id, disposition="select",
                reason="Inspect this private question."), episode_id=episode_id, now_utc=self.now)
        self.commit(attention)
        assert selected.inquiry is not None
        return selected.inquiry

    def step(self, operation: str, *, outcome: Literal["continue", "information_gap"],
             next_operation: str = "") -> InquiryStep:
        """State one executed operation; the outcome decides whether it is a finding."""
        gap = "Jesse's own reason is unknown." if outcome == "information_gap" else ""
        return InquiryStep(operation=operation, thought="The topic faded after a shared setback.",
            evidence_ids=(self.source.id,), assumptions=(), reality="inferred", judgments=self.judgments,
            relations=(), alternatives=(), progress=InquiryProgress(kind="none", inspected_source_ids=(),
                compared_node_ids=(), meaningful_difference=False, reason="Nothing outside was acquired."),
            outcome=outcome, reason="The report names no motive.", information_gap=gap,
            next_operation=next_operation)

    def execute(self, store: MemoryStore, head: InquiryRecord, step: InquiryStep, *,
                integration: IntegrationResult | None = None, limits: ControlConfig | None = None) -> InquiryRecord:
        """Accept one step through the private owner, with its integration at an outcome."""
        owner = self.private_owner(store, limits=limits)
        grant: InquiryGrant = owner.execution_grant(head)
        outcome: GraphOutcomePayload | None = None
        if integration is not None:
            outcome = capture_graph_outcome(grant, step, integration)
        return owner.accept_step(head, step, grant=grant, commit_id=f"step-{head.record.id}", active_seconds=1.0,
            now_utc=self.now + timedelta(minutes=1), mental_state=FullMentalState(), control_generation=0,
            outcome=outcome)

    def lifecycle(self, store: MemoryStore, action_id: int) -> MentalActionLifecycle:
        """Read the action's lifecycle from the live view."""
        return MentalActionLifecycle.model_validate(store.snapshot(capture=False).get(action_id).metadata)

    def deliberate(self, store: MemoryStore, action_id: int) -> InquiryRecord:
        """Read the episode an action owns; the test requires it to exist."""
        head = deliberate_inquiry(store.snapshot(capture=False), action_id)
        assert head is not None
        return head



class DeliberateOwnerTests(DeliberateOwnerCase):
    """Keep admission, parking, the action lifecycle, and delivery on their own records."""

    # ------------------------------------------------------------ admission

    def test_request_opens_a_deliberate_episode_that_starts_with_recall(self) -> None:
        action, envelope = self.dispatch("Work out why Jesse stopped describing the story.")
        self.module.handle_batch([envelope])
        head = self.deliberate(self.store, action.id)
        state = head.receipt.state
        self.assertEqual((state.status, state.sequence, state.node_ids), ("active", 0, ()))
        self.assertEqual(state.next_operation, "memory_search")
        origin = inquiry_origin(self.store.snapshot(capture=False), state)
        self.assertEqual(origin.kind, "deliberate_action")
        self.assertEqual(origin.question, "Work out why Jesse stopped describing the story.")
        self.assertEqual(origin.evidence_ids, (self.source.id,))
        self.assertIn(self.source.id, {source.record_id for source in state.grant.sources})
        # Admission executes nothing: no turn, no result, no delivery.
        self.assertEqual(self.store.gwt_turns, {})
        self.assertIsNone(self.lifecycle(self.store, action.id).mental_result_id)
        self.assertEqual(self.thoughts.events, [])
        restored = MemoryStore.load(self.path)
        self.assertEqual(self.deliberate(restored, action.id).receipt, head.receipt)

    def test_perspective_request_carries_the_partner_into_the_entry_receipt(self) -> None:
        action, envelope = self.dispatch("Work out when Jesse is usually up.",
                                         workflow_name="theory_of_mind", correlation="corr-tom",
                                         partner_entity_id=self.partner.id)
        self.module.handle_batch([envelope])
        head = self.deliberate(self.store, action.id)
        self.assertEqual(head.receipt.state.status, "active")
        self.assertEqual(head.receipt.state.next_operation, "memory_search")
        self.assertEqual(head.receipt.deliberate.subject_entity_id, self.partner.id)
        self.assertEqual(inquiry_origin(self.store.snapshot(capture=False), head.receipt.state).kind,
                         "deliberate_action")
        # The entry receipt is the same as the head: no step ran yet.
        self.assertEqual(self.store.gwt_turns, {})
        self.assertEqual(self.thoughts.events, [])

    def test_plain_deliberate_entry_stores_no_subject_key(self) -> None:
        action, envelope = self.dispatch("Work out why Jesse stopped describing the story.")
        self.module.handle_batch([envelope])
        head = self.deliberate(self.store, action.id)
        self.assertIsNone(head.receipt.deliberate.subject_entity_id)
        entry = self.store.snapshot(capture=False).get(head.record.id)
        stored = entry.metadata["inquiry_receipt"]["deliberate"]
        self.assertNotIn("subject_entity_id", stored)

    def test_perspective_subject_must_be_another_entity(self) -> None:
        observer_action, observer_envelope = self.dispatch("Think about the observer.",
            workflow_name="theory_of_mind", correlation="corr-observer", partner_entity_id=self.observer.id)
        with self.assertRaises(ValueError):
            self.module.handle_batch([observer_envelope])

        not_entity_action, feature_envelope = self.dispatch("Think about a message.",
            workflow_name="theory_of_mind", correlation="corr-feature", partner_entity_id=self.source.id)
        with self.assertRaises(ValueError):
            self.module.handle_batch([feature_envelope])

        missing_envelope = self.dispatch("Think about an unknown entity.",
            workflow_name="theory_of_mind", correlation="corr-missing", partner_entity_id=999_999)[1]
        with self.assertRaises(ValueError):
            self.module.handle_batch([missing_envelope])

    def test_only_a_delivered_operation_name_replaces_recall(self) -> None:
        named, named_request = self.dispatch("Compare the two endings.", operation="feasibility_check",
                                             correlation="corr-named")
        self.module.handle_batch([named_request])
        self.assertEqual(self.deliberate(self.store, named.id).receipt.state.next_operation, "feasibility_check")
        self.execute(self.store, self.deliberate(self.store, named.id),
                     self.step("feasibility_check", outcome="information_gap"), integration=IntegrationResult())
        # "status" is wording for another workflow, not an inquiry operation.
        worded, worded_request = self.dispatch("Recall the hobby.", operation="status", correlation="corr-worded")
        self.module.handle_batch([worded_request])
        self.assertEqual(self.deliberate(self.store, worded.id).receipt.state.next_operation, "memory_search")

    def test_replayed_request_and_restart_open_no_second_episode(self) -> None:
        action, envelope = self.dispatch("Work out why Jesse stopped describing the story.")
        self.module.handle_batch([envelope])
        first = self.deliberate(self.store, action.id)
        self.module.handle_batch([envelope])
        restored = MemoryStore.load(self.path)
        self.build_module(restored).handle_batch([envelope])
        history = inquiry_history(restored.snapshot(capture=False), f"deliberate:{action.id}")
        self.assertEqual(tuple(item.record.id for item in history), (first.record.id,))

    def test_withdrawn_action_never_enters_the_slot(self) -> None:
        action, envelope = self.dispatch("Recall the hobby.", dispatch_status="deferred")
        self.module.handle_batch([envelope])
        self.module.handle_batch([])
        self.assertIsNone(deliberate_inquiry(self.store.snapshot(capture=False), action.id))
        self.assertEqual(len(self.deferrals.events), 1)
        self.assertIn("withdrew", str(self.deferrals.events[0].payload["reason"]))

    # -------------------------------------------------------------- parking

    def test_deliberate_work_parks_spontaneous_work_and_names_the_action(self) -> None:
        spontaneous = self.spontaneous()
        action, envelope = self.dispatch("Work out why Jesse stopped describing the story.")
        self.module.handle_batch([envelope])
        view = self.store.snapshot(capture=False)
        parked = inquiry_history(view, "spontaneous-story")[-1]
        before = spontaneous.receipt.state
        after = parked.receipt.state
        self.assertEqual(after.status, "paused")
        self.assertEqual(after.stop_reason, PREEMPTED_REASON)
        assert parked.receipt.preemption is not None
        self.assertEqual(parked.receipt.preemption.action_record_id, action.id)
        # Parking is not progress: the graph, the next operation, and the budgets are untouched.
        self.assertEqual((after.node_ids, after.sequence, after.next_operation, after.active_elapsed_seconds,
                          after.stalled_steps, after.grant),
                         (before.node_ids, before.sequence, before.next_operation, before.active_elapsed_seconds,
                          before.stalled_steps, before.grant))
        holder = active_inquiry(view, observer_entity_id=self.observer.id)
        assert holder is not None
        self.assertEqual(holder.receipt.state.episode_id, f"deliberate:{action.id}")
        self.assertEqual(self.store.gwt_turns, {})
        restored = MemoryStore.load(self.path)
        self.assertEqual(inquiry_history(restored.snapshot(capture=False), "spontaneous-story")[-1].receipt,
                         parked.receipt)

    def test_parked_question_can_return_through_attention_once_the_slot_is_free(self) -> None:
        self.spontaneous()
        action, envelope = self.dispatch("Compare the two endings.", operation="feasibility_check")
        self.module.handle_batch([envelope])
        self.execute(self.store, self.deliberate(self.store, action.id),
                     self.step("feasibility_check", outcome="information_gap"), integration=IntegrationResult())
        view = self.store.snapshot(capture=False)
        parked = inquiry_history(view, "spontaneous-story")[-1]
        owner = self.private_owner(self.store)
        grant = owner.execution_grant(parked)
        offer = parked_offer(view, parked, grant=grant, now_utc=vclock.utcnow(), limits=ControlConfig())
        assert offer is not None
        self.assertTrue(offer.deadline_elapsed)
        self.assertEqual(offer.stop_reason, PREEMPTED_REASON)
        reason = "The story question still matters."
        resumed = owner.accept_resume(parked,
            InquiryResumeDecision(operation="perspective_shift", reason=reason, trigger_source_ids=()),
            attention=ThoughtAttentionDecision(parked_receipt_id=parked.record.id, disposition="select",
                admitted_operation="perspective_shift", reason=reason),
            grant=grant, commit_id="resume-after-deliberate", now_utc=vclock.utcnow())
        self.assertEqual(resumed.receipt.state.status, "active")
        self.assertEqual(resumed.receipt.state.episode_id, "spontaneous-story")

    def test_second_deliberate_action_waits_and_enters_when_the_first_leaves(self) -> None:
        first, first_request = self.dispatch("Compare the two endings.", operation="feasibility_check",
                                             correlation="corr-first")
        second, second_request = self.dispatch("Recall the hobby.", correlation="corr-second")
        self.module.handle_batch([first_request, second_request])
        self.module.handle_batch([])
        self.assertIsNone(deliberate_inquiry(self.store.snapshot(capture=False), second.id))
        self.assertEqual(self.deliberate(self.store, first.id).receipt.state.status, "active")
        # The wait is reported once, not on every timer tick.
        self.assertEqual([item.payload["action_schema_id"] for item in self.deferrals.events], [second.id])
        self.execute(self.store, self.deliberate(self.store, first.id),
                     self.step("feasibility_check", outcome="information_gap"), integration=IntegrationResult())
        self.module.handle_batch([])
        self.assertEqual(self.deliberate(self.store, second.id).receipt.state.status, "active")
        self.assertIsNone(self.deliberate(self.store, first.id).receipt.preemption)

    # ------------------------------------------------- the awaited outcome

    def test_deliberate_step_is_told_its_budget_and_must_end_at_the_last_operation(self) -> None:
        action, envelope = self.dispatch("Compare the two endings.", operation="feasibility_check")
        self.module.handle_batch([envelope])
        head = self.deliberate(self.store, action.id)
        owner = self.private_owner(self.store)
        supplied = graph_thought_request(self.store.snapshot(capture=False), InnerRequest(companion_name="Spamuel"),
                                         head, owner.execution_grant(head))
        # A request without a budget serializes exactly as before: no new key reaches the prompt.
        self.assertNotIn("deliberate_budget", supplied.model_dump(mode="json"))
        self.assertNotIn("deliberate mental work", graph_thought_instruction(supplied))
        early = supplied.model_copy(update={"deliberate_budget": DeliberateBudget(
            operation_number=2, operation_limit=6, stalled_operations=0, stalled_limit=2)})
        instruction = graph_thought_instruction(early)
        self.assertIn("Main cognition waits for its result.", instruction)
        self.assertIn("This is operation 2 of at most 6.", instruction)
        self.assertIn("0 of 2 consecutive operations without progress are used", instruction)
        self.assertNotIn("This is the last operation", instruction)
        self.assertNotIn("must make such progress", instruction)
        continuing = self.step("feasibility_check", outcome="continue", next_operation="perspective_shift")
        validate_graph_step(early, continuing)
        # One more operation without progress would park the work, so it has to end instead.
        stalling = supplied.model_copy(update={"deliberate_budget": DeliberateBudget(
            operation_number=3, operation_limit=6, stalled_operations=1, stalled_limit=2)})
        self.assertIn("This operation must make such progress or state an outcome.",
                      graph_thought_instruction(stalling))
        with self.assertRaisesRegex(ModelProposalError, "would park the deliberate mental work without a result"):
            validate_graph_step(stalling, continuing)
        validate_graph_step(stalling, self.step("feasibility_check", outcome="information_gap"))
        last = supplied.model_copy(update={"deliberate_budget": DeliberateBudget(
            operation_number=6, operation_limit=6, stalled_operations=0, stalled_limit=2)})
        self.assertIn("This is the last operation, so outcome='continue' is not available.",
                      graph_thought_instruction(last))
        with self.assertRaisesRegex(ModelProposalError, "last operation of deliberate mental work"):
            validate_graph_step(last, continuing)
        validate_graph_step(last, self.step("feasibility_check", outcome="information_gap"))
        # The budget survives the request's JSON round trip into the thought stage.
        restored = type(last).model_validate(last.model_dump(mode="json"))
        self.assertEqual(restored.deliberate_budget, last.deliberate_budget)

    # --------------------------------------------------- result and delivery

    def test_finding_closes_the_action_and_is_delivered_once_under_its_correlation(self) -> None:
        action, envelope = self.dispatch("Compare the two endings.", operation="feasibility_check")
        self.module.handle_batch([envelope])
        advice = FocusProposal(instruction="Keep the unfinished story in view.", promote_ids=[self.source.id],
                               strength=0.3, seconds=600, reason="Jesse dropped the topic without explaining.")
        final = self.execute(self.store, self.deliberate(self.store, action.id),
            self.step("feasibility_check", outcome="information_gap"),
            integration=IntegrationResult(summary="The reason is still unknown.", focus=advice))
        finding_id = final.receipt.operation_record_id
        assert finding_id is not None
        # The private loop must not deliver a deliberate finding as its own.
        self.assertEqual(pending_findings(self.store), ())
        self.module.handle_batch([])
        self.module.handle_batch([])
        lifecycle = self.lifecycle(self.store, action.id)
        self.assertEqual(lifecycle.mental_result_id, finding_id)
        self.assertEqual(lifecycle.dispatch_status, "admitted")
        metadata = self.store.snapshot(capture=False).get(action.id).metadata
        self.assertEqual((metadata["mental_status"], metadata["execution_status"]), ("proposed", "completed"))
        self.assertEqual(len(self.thoughts.events), 1)
        delivered = self.thoughts.events[0]
        self.assertEqual((delivered.source, delivered.correlation_id), ("focus_management", "corr-story"))
        self.assertEqual(delivered.payload["knoxel_id"], finding_id)
        turn = self.store.gwt_turns_of("inner", limit=1)[0]
        self.assertEqual(delivered.payload["gwt_tick"], turn.tick_id)
        self.assertEqual([item.payload["status"] for item in self.receipts.events], ["proposed"])
        # Main cognition reads the proposal, with its focus advice, through the existing contract.
        plan = MentalPlan.model_validate(self.store.snapshot(capture=False).get(finding_id).metadata["mental_plan"])
        assert plan.focus is not None
        self.assertEqual(plan.focus.promote_ids, [self.source.id])
        self.assertEqual(plan.disposition, "propose")

    def test_accepted_main_cycle_ends_delivery_across_a_restart(self) -> None:
        action, envelope = self.dispatch("Compare the two endings.", operation="feasibility_check")
        self.module.handle_batch([envelope])
        final = self.execute(self.store, self.deliberate(self.store, action.id),
            self.step("feasibility_check", outcome="information_gap"), integration=IntegrationResult())
        finding_id = final.receipt.operation_record_id
        assert finding_id is not None
        self.module.handle_batch([])
        main = self.store.begin_batch("main-appraisal", new_tick=True, embed=False)
        main.depend_on([finding_id])
        main.set_gwt(GwtTurnRecord(tick_id=main.tick_id, slot_id="main", knoxel_ids=[finding_id],
                                  summary="Appraise the deliberate finding."),
                     CycleState(tick_id=main.tick_id, mental_state=FullMentalState(),
                                stimulus_groups=[StimulusGroup(stimulus_ids=[finding_id])]))
        self.store.commit(main)
        restored = MemoryStore.load(self.path)
        self.build_module(restored).handle_batch([])
        self.assertEqual(len(self.thoughts.events), 1)
        self.assertEqual(self.lifecycle(restored, action.id).mental_result_id, finding_id)

    def test_restart_delivers_only_after_the_inner_turn_exists(self) -> None:
        action, envelope = self.dispatch("Compare the two endings.", operation="feasibility_check")
        self.module.handle_batch([envelope])
        final = self.execute(self.store, self.deliberate(self.store, action.id),
            self.step("feasibility_check", outcome="information_gap"), integration=IntegrationResult())
        finding_id = final.receipt.operation_record_id
        assert finding_id is not None
        # A clean restart keeps the graph and drops the unclaimed derived inner turn.
        restored = MemoryStore.load(self.path)
        self.assertEqual(restored.gwt_turns_of("inner", limit=1), [])
        module = self.build_module(restored)
        module.handle_batch([])
        self.assertEqual(self.lifecycle(restored, action.id).mental_result_id, finding_id)
        self.assertEqual(self.thoughts.events, [])
        self.private_owner(restored).ensure_inner_turn(self.deliberate(restored, action.id))
        module.handle_batch([])
        self.assertEqual([item.payload["knoxel_id"] for item in self.thoughts.events], [finding_id])
        self.assertEqual(len(inquiry_history(restored.snapshot(capture=False), f"deliberate:{action.id}")), 2)

    def test_episode_parked_at_a_bound_closes_the_action_with_a_notice(self) -> None:
        action, envelope = self.dispatch("Compare the two endings.", operation="feasibility_check")
        self.module.handle_batch([envelope])
        one_step = ControlConfig(episode_steps=1)
        parked = self.execute(self.store, self.deliberate(self.store, action.id),
            self.step("feasibility_check", outcome="continue", next_operation="perspective_shift"), limits=one_step)
        self.assertEqual((parked.receipt.state.status, parked.receipt.state.stop_reason), ("paused", "step limit"))
        self.module.handle_batch([])
        self.module.handle_batch([])
        lifecycle = self.lifecycle(self.store, action.id)
        self.assertEqual(lifecycle.mental_status, "parked")
        view = self.store.snapshot(capture=False)
        self.assertEqual(view.get(action.id).metadata["execution_status"], "parked")
        assert lifecycle.mental_result_id is not None
        notice = view.get(lifecycle.mental_result_id)
        assert isinstance(notice, Feature)
        # Main cognition selected this work, so it hears that the work stopped and why.
        self.assertEqual(notice.content, "I set out to think this through: Compare the two endings. "
                         "I stopped before I reached a conclusion (step limit). The question is parked for now.")
        self.assertEqual(notice.feature_type, FeatureType.Thought)
        assert notice.lineage is not None
        self.assertEqual(notice.lineage.parent_record_ids, [action.id, parked.record.id])
        # The notice proposes nothing: no plan, no goal change, no focus advice.
        self.assertNotIn("mental_plan", notice.metadata)
        self.assertEqual([(item.correlation_id, item.payload["knoxel_id"]) for item in self.thoughts.events],
                         [("corr-story", notice.id)])
        self.assertEqual([item.payload["status"] for item in self.receipts.events], ["parked"])
        # The parked graph itself is unchanged and no second notice appears after a restart.
        restored = MemoryStore.load(self.path)
        self.build_module(restored).handle_batch([])
        self.assertEqual(self.lifecycle(restored, action.id).mental_result_id, notice.id)
        self.assertEqual(self.deliberate(restored, action.id).record.id, parked.record.id)


class DeliberateControlFenceTests(DeliberateOwnerCase):
    """Admit and close deliberate work under the real control fence.

    The classifications are explicit boundary inputs, as in the early-triage
    checks. An accepted wait is not armed here: arming one embeds its record,
    which needs the remote embedding endpoint.
    """

    def setUp(self) -> None:
        super().setUp()
        self.control: CognitiveControl = CognitiveControl("control", self.bus, self.module_error, time.monotonic, None,
                                                          ghost=self.store)
        self.control.enable_input_triage()
        self.input: UserInputProxy = UserInputProxy("user_input", self.bus, self.module_error, time.monotonic, self.store,
                                                    lambda text: None)
        self.input.register_triage_input = self.control.register_input
        self.module.cognitive_control = self.control

    def submit(self, text: str) -> int:
        """Commit input through actual intake and return its source identity."""
        correlation = self.input.submit(text)
        return next(item.id for item in self.store.all_features
                    if item.metadata.get("exchange_correlation") == correlation)

    def assess(self, source_id: int, category: str) -> None:
        """Supply an explicitly named classification at the control boundary."""
        view = self.store.recovery_snapshot(capture=False)
        fingerprints = {source_id: source_fingerprint(view.get(source_id))}
        self.control.accept_triage(EarlyTriageAssessment(
            source_ids=[source_id], source_fingerprints=fingerprints, guidance_tick=None,
            evidence=TriageEvidence(record_fingerprints=fingerprints, commit_ids=sorted(view.commit_ids),
                                    current_head_ids=[source_id]),
            result=TriageResult(items=[TriageItem(knoxel_id=source_id, category=category,
                                                 reasoning="Explicit scheduling-boundary input.")])))

    def accept_cycle(self, source_ids: list[int]) -> None:
        """Accept a real main cycle that claims the sources, without an LLM calculation."""
        batch = self.store.begin_batch(f"boundary-cycle-{len(self.store.states)}", new_tick=True, embed=False)
        batch.depend_on(source_ids)
        state = CycleState(tick_id=batch.tick_id, stimulus_groups=[StimulusGroup(stimulus_ids=source_ids)])
        batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, stimulus_ids=source_ids), state)
        self.control.commit(batch, None, self.control.generation, source_ids=source_ids)

    def test_unclassified_input_holds_admission_until_its_assessment(self) -> None:
        action, envelope = self.dispatch("Work out why Jesse stopped describing the story.")
        message = self.submit("Something arrived just now.")
        self.module.handle_batch([envelope])
        self.module.handle_batch([])
        self.assertIsNone(deliberate_inquiry(self.store.snapshot(capture=False), action.id))
        self.assess(message, "Moderate")
        self.module.handle_batch([])
        state = self.deliberate(self.store, action.id).receipt.state
        self.assertEqual((state.status, state.grant.critical_epoch), ("active", 0))

    def test_work_from_an_older_critical_epoch_is_displaced_not_parked(self) -> None:
        self.spontaneous()
        message = self.submit("Stop. There is a fire in the kitchen right now.")
        self.assess(message, "Critical")
        action, envelope = self.dispatch("Work out what to do about the kitchen.")
        self.module.handle_batch([envelope])
        self.assertIsNone(deliberate_inquiry(self.store.snapshot(capture=False), action.id),
                          "Required reconsideration must hold deliberate admission.")
        self.accept_cycle([message])
        self.module.handle_batch([])
        view = self.store.snapshot(capture=False)
        stale = inquiry_history(view, "spontaneous-story")[-1]
        self.assertEqual(stale.receipt.state.status, "displaced")
        self.assertIsNone(stale.receipt.preemption)
        state = self.deliberate(self.store, action.id).receipt.state
        self.assertEqual((state.status, state.grant.critical_epoch), ("active", 1))

    def test_critical_reconsideration_closes_the_deliberate_episode_and_its_action(self) -> None:
        action, envelope = self.dispatch("Compare the two endings.", operation="feasibility_check")
        self.module.handle_batch([envelope])
        message = self.submit("Stop. There is a fire in the kitchen right now.")
        self.assess(message, "Critical")
        self.accept_cycle([message])
        # The private loop closes superseded work at its next opportunity.
        displaced = self.private_owner(self.store).displace_superseded(observer_entity_id=self.observer.id,
                                                                        critical_epoch=1)
        self.assertEqual(len(displaced), 1)
        self.module.handle_batch([])
        lifecycle = self.lifecycle(self.store, action.id)
        self.assertEqual(lifecycle.mental_status, "displaced")
        view = self.store.snapshot(capture=False)
        self.assertEqual(view.get(action.id).metadata["execution_status"], "cancelled")
        assert lifecycle.mental_result_id is not None
        notice = view.get(lifecycle.mental_result_id)
        assert isinstance(notice, Feature)
        self.assertIn("Something critical came up before I reached a conclusion", notice.content)
        assert notice.lineage is not None
        self.assertEqual(notice.lineage.parent_record_ids, [action.id, displaced[0].record.id])
        self.assertEqual([item.payload["knoxel_id"] for item in self.thoughts.events], [notice.id])
        self.assertEqual([item.payload["status"] for item in self.receipts.events], ["displaced"])


if __name__ == "__main__":
    unittest.main()
