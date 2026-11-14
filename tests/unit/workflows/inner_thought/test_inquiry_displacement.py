"""Verify Critical displacement of private inquiries through actual store and checkpoint boundaries.

The proposals are synthetic storage inputs. These checks execute no nomination,
attention, or thought model, and no substitute service. They cover the durable
closing of work from an older Critical epoch, the question that stays offered
to shared attention, and the new episode that restarts it under a fresh grant.
One class supplies the Critical epoch as a plain number. The other drives the
real ``CognitiveControl`` owner: an explicitly supplied Critical assessment, an
accepted main cycle, and every private commit through its control fence.
The generated behaviour belongs to the recorded scenario and the live gates.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import time
import unittest

from pydantic import JsonValue, ValidationError

from pm import vclock
from pm.bus.bus import BusSystem
from pm.memory.snapshot import GhostCancelled, GhostConflict
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.enums import EntityClass, FeatureType, InterlocusType, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Entity, Feature
from pm.records.mental_state import FullMentalState
from workflows.cognitive_control.attention import captured_private_focus, validate_private_attention
from workflows.cognitive_control.host import CognitiveControl
from workflows.cognitive_control.schemas import ControlConfig
from workflows.inner_thought.graph_execution import (
    displaced_candidate, graph_thought_instruction, graph_thought_request, run_graph_opportunity,
    validate_graph_step,
)
from workflows.inner_thought.graph_host import (
    PrivateInquiryOwner, active_inquiry, displaced_inquiries, parked_inquiries,
)
from workflows.inner_thought.inquiry import (
    DISPLACED_REASON, InquiryRecord, inquiry_history, inquiry_origin, stage_attention, stage_inquiry_displacement,
    stage_inquiry_resume, stage_inquiry_step,
)
from workflows.inner_thought.nominations import capture_inquiry_grant, stage_nominations
from workflows.input_triage.schemas import EarlyTriageAssessment, TriageEvidence
from workflows.main_cognition.schemas import TriageItem, TriageResult
from workflows.public_speech.public_context import source_fingerprint
from workflows.public_speech.user_input import UserInputProxy
from workflows.inner_thought.pipeline import ModelProposalError
from workflows.inner_thought.schemas import (
    InnerRequest, InquiryAlternative, InquiryGrant, InquiryProgress, InquiryResumeDecision, InquiryStep,
    PrivateAttentionContext, ThoughtAttentionDecision, ThoughtAttentionReceipt, ThoughtJudgments, ThoughtNomination,
    ThoughtRelation,
)


def forbidden_stage(name: str, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Fail the check if a displacement alone reaches any child model stage."""
    raise AssertionError(f"Displacement must not call the {name} stage.")


class DisplacementFixture(unittest.TestCase):
    """A real store with one observer, one outside source, and the private owner."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path: Path = Path(directory.name) / "displacement.pickle"
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=self.path)
        self.now: datetime = datetime(2026, 11, 14, 0, 0)
        previous_clock = vclock.active()
        self.addCleanup(vclock.install, previous_clock)
        vclock.install(vclock.VirtualClock(self.now))
        self.limits: ControlConfig = ControlConfig()
        anchors = self.store.begin_batch("anchors", embed=False)
        self.observer: Entity = anchors.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.partner: Entity = anchors.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.source: Feature = anchors.create(Feature(content="The sketch remains unfinished.", causal=True,
            feature_type=FeatureType.Dialogue, entity_id=self.partner.id, addressee_entity_id=self.observer.id,
            interlocus=InterlocusType.Public, provenance=Provenance(source_kind=SourceKind.User)))
        self.commit_root(anchors)
        self.old_epoch: int = 2
        self.new_epoch: int = 3
        self.grant: InquiryGrant = self.capture(self.old_epoch)
        self.judgments: ThoughtJudgments = ThoughtJudgments(interestingness=0.6, character_relevance=0.6,
            plausibility=0.3, consequence=0.4, desirability=0.4, action_suitability=0.2,
            reason="An unfinished sketch leaves a question, not an observed motive.")
        self.owner: PrivateInquiryOwner = PrivateInquiryOwner(self.store, commit=self.commit, cancel=Event(),
                                                              limits=self.limits)

    def commit(self, batch: WriteBatch) -> None:
        """Use actual durable production acceptance; the owner marks its own recovery roots."""
        self.store.commit(batch)

    def commit_root(self, batch: WriteBatch) -> None:
        """Persist an append-only test input as a clean recovery root."""
        batch.preserve_on_recovery()
        self.store.commit(batch)

    def capture(self, critical_epoch: int) -> InquiryGrant:
        """Capture the grant one private opportunity would hold at this accepted Critical epoch."""
        return capture_inquiry_grant(self.store.snapshot(capture=False), observer_entity_id=self.observer.id,
            scope=RecordScope(), source_ids=[self.source.id], governing_ids=[self.source.id],
            critical_epoch=critical_epoch)

    def select(self, episode_id: str, *, grant: InquiryGrant | None = None) -> InquiryRecord:
        """Allocate the private slot from a committed nomination through its storage owner."""
        selected_grant = grant or self.grant
        batch = self.store.begin_batch("nomination-" + episode_id, embed=False)
        records = stage_nominations(batch, [ThoughtNomination(text="The sketch could reflect several concerns.",
            question="What remains unknown about the sketch?", evidence_ids=(self.source.id,), reality="inferred",
            judgments=self.judgments, suggested_operation="future_projection")], grant=selected_grant, batch_id=episode_id)
        self.commit_root(batch)
        attention = self.store.begin_batch("attention-" + episode_id, embed=False)
        selected = stage_attention(attention, [records[0].record.id],
            ThoughtAttentionDecision(nomination_id=records[0].record.id, disposition="select", reason="Inspect this private question."),
            episode_id=episode_id, now_utc=self.now)
        self.commit_root(attention)
        assert selected.inquiry is not None
        return selected.inquiry

    def generation(self, *, outcome: str = "continue") -> InquiryStep:
        """Supply two possible readings without claiming a conclusion."""
        branches = tuple(InquiryAlternative(text=text, evidence_ids=(self.source.id,), assumptions=(assumption,),
            reality="imagined", judgments=self.judgments, disposition="open", reason="A possible cause, not observation.")
            for text, assumption in [("Jesse revisits the sketch.", "Interest remains."),
                                     ("Jesse chooses another subject.", "A new interest matters more.")])
        continuing = outcome == "continue"
        return InquiryStep.model_validate({
            "operation": "future_projection", "thought": "Both explanations remain possible.",
            "evidence_ids": (self.source.id,), "assumptions": (), "reality": "inferred", "judgments": self.judgments,
            "relations": (), "alternatives": branches,
            "progress": InquiryProgress(kind="none", inspected_source_ids=(), compared_node_ids=(),
                meaningful_difference=False, reason="Possibilities provide no new outside evidence."),
            "outcome": outcome, "reason": "Compare the assumptions before concluding.",
            "information_gap": "Unknown motive.", "next_operation": "feasibility_check" if continuing else ""})

    def working(self, episode_id: str = "sketch", *, outcome: str = "continue") -> InquiryRecord:
        """Admit an inquiry under the old epoch and execute one real generative operation."""
        selected = self.select(episode_id)
        return self.owner.accept_step(selected, self.generation(outcome=outcome),
            grant=self.owner.execution_grant(selected), commit_id="generation:" + episode_id, active_seconds=1.5,
            now_utc=self.now, mental_state=FullMentalState(), control_generation=0)

    def displace(self) -> InquiryRecord:
        """Close the stale work the way a private opportunity at the new epoch does."""
        closed = self.owner.displace_superseded(observer_entity_id=self.observer.id, critical_epoch=self.new_epoch)
        self.assertEqual(len(closed), 1)
        return closed[0]

    def admit(self, displaced: InquiryRecord, *, batch_id: str = "opportunity-one",
              offered_grant: InquiryGrant | None = None) -> InquiryRecord:
        """Commit the shared attention decision that restarts a displaced question."""
        offered = offered_grant or self.capture(self.new_epoch)
        decision = ThoughtAttentionDecision(displaced_receipt_id=displaced.record.id, disposition="select",
            admitted_operation="feasibility_check", reason="The interrupted question still matters.")
        return self.owner.admit_displaced_inquiry(displaced, attention=decision, nomination_ids=(),
            offered_grant=offered, grant=self.owner.displaced_grant(displaced, offered_grant=offered),
            batch_id=batch_id, episode_id="inquiry:" + batch_id, now_utc=self.now,
            focus=captured_private_focus(self.store.snapshot(capture=False), self.now))

    def attention_outcomes(self, batch_id: str) -> list[ThoughtAttentionReceipt]:
        """Read the retained attention outcomes recorded for one opportunity identity."""
        view = self.store.snapshot(capture=False)
        return [ThoughtAttentionReceipt.model_validate(record.metadata["thought_attention"])
                for record in view.all_knoxels.values()
                if record.metadata.get("thought_attention_batch") == batch_id]


class InquiryDisplacementTests(DisplacementFixture):
    """Keep the question, drop the authority, and never hold the slot with stale work."""

    def test_displacement_frees_the_slot_and_keeps_the_graph_as_history(self) -> None:
        working = self.working()
        before = working.receipt.state
        inner_turns_before = len(self.store.gwt_turns_of("inner", limit=10))
        closed = self.displace()
        restored = MemoryStore.load(self.path)
        view = restored.snapshot(capture=False)
        self.assertIsNone(active_inquiry(view, observer_entity_id=self.observer.id))
        self.assertEqual(parked_inquiries(view, observer_entity_id=self.observer.id), ())
        history = inquiry_history(view, "sketch")
        self.assertEqual(history[-1].record.id, closed.record.id)
        after = history[-1].receipt.state
        self.assertEqual(after.status, "displaced")
        self.assertEqual(after.stop_reason, DISPLACED_REASON)
        self.assertEqual(after.next_operation, "")
        self.assertEqual(after.grant, before.grant)
        self.assertEqual(after.node_ids, before.node_ids)
        self.assertEqual(after.pending_generation_ids, before.pending_generation_ids)
        self.assertEqual(after.sequence, before.sequence)
        self.assertEqual(after.active_elapsed_seconds, before.active_elapsed_seconds)
        displacement = history[-1].receipt.displacement
        assert displacement is not None
        self.assertEqual(displacement.superseding_critical_epoch, self.new_epoch)
        self.assertIsNone(history[-1].receipt.operation_record_id)
        # Closing the episode executes nothing, so it projects no inner turn.
        self.assertEqual(len(self.store.gwt_turns_of("inner", limit=10)), inner_turns_before)

    def test_work_from_the_current_epoch_is_not_displaced(self) -> None:
        working = self.working()
        closed = self.owner.displace_superseded(observer_entity_id=self.observer.id, critical_epoch=self.old_epoch)
        self.assertEqual(closed, ())
        active = active_inquiry(self.store.snapshot(capture=False), observer_entity_id=self.observer.id)
        assert active is not None
        self.assertEqual(active.record.id, working.record.id)

    def test_parked_work_from_an_older_epoch_is_displaced_too(self) -> None:
        parked = self.working(outcome="pause")
        self.assertEqual(parked.receipt.state.status, "paused")
        closed = self.displace()
        view = self.store.snapshot(capture=False)
        self.assertEqual(parked_inquiries(view, observer_entity_id=self.observer.id), ())
        waiting = displaced_inquiries(view, observer_entity_id=self.observer.id)
        self.assertEqual(tuple(head.record.id for head in waiting), (closed.record.id,))

    def test_a_corrected_source_blocks_neither_displacement_nor_the_new_episode(self) -> None:
        self.working()
        original_id = self.source.id
        corrected = self.store.begin_batch("correct-original", embed=False)
        source = corrected.view().get(original_id).copy_for_edit()
        source.content = "The sketch is finished."
        successor = corrected.replace(source)
        self.store.commit(corrected)
        self.assertNotEqual(successor.id, original_id)
        closed = self.displace()
        self.assertEqual(closed.receipt.state.status, "displaced")
        # This opportunity cites no current source of its own. The correction
        # above is not an accepted main cycle, so a private recovery root
        # could not depend on its successor record anyway.
        observer_only = capture_inquiry_grant(self.store.snapshot(capture=False),
            observer_entity_id=self.observer.id, scope=RecordScope(), source_ids=[], critical_epoch=self.new_epoch)
        restarted = self.admit(closed, offered_grant=observer_only)
        self.assertEqual(restarted.receipt.state.status, "active")
        granted = {item.record_id: item for item in restarted.receipt.state.grant.sources}
        # The earlier thinking came from the superseded record. It stays citable
        # as history, without current authority, and is the root of that thinking.
        self.assertFalse(granted[original_id].current_head)
        self.assertNotIn(successor.id, granted)
        self.assertEqual(restarted.receipt.state.root_source_ids, (original_id,))

    def test_a_displaced_inquiry_accepts_no_further_transition(self) -> None:
        working = self.working()
        closed = self.displace()
        before = self.path.read_bytes()
        step = self.generation()
        with self.assertRaisesRegex(ValueError, "Only an admitted active inquiry"):
            stage_inquiry_step(self.store.begin_batch("late-step", embed=False), parent_receipt_id=closed.record.id,
                step=step, grant=self.owner.execution_grant(closed), active_seconds=1, now_utc=self.now, limits=self.limits)
        with self.assertRaisesRegex(GhostConflict, "exact current receipt parent"):
            stage_inquiry_step(self.store.begin_batch("stale-parent", embed=False), parent_receipt_id=working.record.id,
                step=step, grant=self.owner.execution_grant(working), active_seconds=1, now_utc=self.now, limits=self.limits)
        reason = "Resume the closed question."
        with self.assertRaisesRegex(ValueError, "Only a paused inquiry"):
            stage_inquiry_resume(self.store.begin_batch("late-resume", embed=False), parent_receipt_id=closed.record.id,
                decision=InquiryResumeDecision(operation="feasibility_check", reason=reason, trigger_source_ids=()),
                attention=ThoughtAttentionDecision(parked_receipt_id=closed.record.id, disposition="select",
                    admitted_operation="feasibility_check", reason=reason),
                grant=closed.receipt.state.grant, now_utc=self.now, limits=self.limits)
        with self.assertRaisesRegex(ValueError, "Only active or parked work"):
            stage_inquiry_displacement(self.store.begin_batch("twice", embed=False),
                parent_receipt_id=closed.record.id, superseding_critical_epoch=self.new_epoch + 1)
        self.assertEqual(self.path.read_bytes(), before)

    def test_only_a_newer_epoch_can_displace(self) -> None:
        working = self.working()
        with self.assertRaisesRegex(ValueError, "newer accepted Critical epoch"):
            stage_inquiry_displacement(self.store.begin_batch("same-epoch", embed=False),
                parent_receipt_id=working.record.id, superseding_critical_epoch=self.old_epoch)

    def test_the_question_stays_offered_through_deferral_and_dismissal(self) -> None:
        self.working()
        closed = self.displace()
        current = self.capture(self.new_epoch)
        focus = captured_private_focus(self.store.snapshot(capture=False), self.now)
        for batch_id, disposition in [("deferred", "defer"), ("dismissed", "dismiss")]:
            candidate = displaced_candidate(self.owner, observer_entity_id=self.observer.id, grant=current)
            assert candidate is not None
            self.assertEqual(candidate.offer.receipt_id, closed.record.id)
            self.assertEqual(candidate.offer.question, "What remains unknown about the sketch?")
            self.assertEqual(candidate.offer.displaced_reason, DISPLACED_REASON)
            self.assertEqual(candidate.offer.earlier_operation_count, 1)
            context = PrivateAttentionContext(candidates=(), displaced=(candidate.offer,), pending_input_ids=(),
                accepted_focus=None, active_intention_ids=(), attention_summary="")
            decision = validate_private_attention(context,
                ThoughtAttentionDecision.model_validate({"disposition": disposition, "reason": "Not now."}))
            self.owner.record_attention_outcome(batch_id=batch_id, nomination_ids=(), decision=decision,
                grant=current, focus=focus)
        still_offered = displaced_candidate(self.owner, observer_entity_id=self.observer.id, grant=current)
        assert still_offered is not None
        self.assertEqual(still_offered.offer.receipt_id, closed.record.id)

    def test_no_decision_can_name_a_displaced_question_without_admitting_it(self) -> None:
        for disposition in ("defer", "dismiss"):
            with self.assertRaisesRegex(ValidationError, "cannot be dismissed"):
                ThoughtAttentionDecision.model_validate(
                    {"displaced_receipt_id": 7, "disposition": disposition, "reason": "Drop it."})
        with self.assertRaisesRegex(ValidationError, "exactly one"):
            ThoughtAttentionDecision(nomination_id=5, displaced_receipt_id=7, disposition="select",
                admitted_operation="feasibility_check", reason="Both.")
        with self.assertRaisesRegex(ValidationError, "names the delivered operation"):
            ThoughtAttentionDecision(displaced_receipt_id=7, disposition="select", reason="No operation.")
        context = PrivateAttentionContext(candidates=(), pending_input_ids=(), accepted_focus=None,
            active_intention_ids=(), attention_summary="")
        with self.assertRaisesRegex(ValueError, "outside the supplied offers"):
            validate_private_attention(context, ThoughtAttentionDecision(displaced_receipt_id=7,
                disposition="select", admitted_operation="feasibility_check", reason="Not offered."))

    def test_admission_opens_a_new_episode_under_the_current_grant(self) -> None:
        working = self.working()
        earlier = working.receipt.state
        closed = self.displace()
        restarted = self.admit(closed)
        state = restarted.receipt.state
        self.assertEqual(state.status, "active")
        self.assertNotEqual(state.episode_id, earlier.episode_id)
        self.assertEqual(state.origin_record_id, closed.record.id)
        self.assertEqual(state.grant.critical_epoch, self.new_epoch)
        self.assertEqual(state.node_ids, ())
        self.assertEqual(state.pending_generation_ids, ())
        self.assertEqual(state.sequence, 0)
        self.assertEqual(state.active_elapsed_seconds, 0)
        self.assertEqual(state.stalled_steps, 0)
        self.assertEqual(state.next_operation, "feasibility_check")
        self.assertEqual(state.root_source_ids, (self.source.id,))
        granted = {source.record_id for source in state.grant.sources}
        self.assertTrue({closed.record.id, *earlier.node_ids}.issubset(granted))
        view = self.store.snapshot(capture=False)
        origin = inquiry_origin(view, state)
        self.assertEqual(origin.kind, "displaced_inquiry")
        self.assertEqual(origin.question, "What remains unknown about the sketch?")
        self.assertIsNone(origin.judgments)
        active = active_inquiry(view, observer_entity_id=self.observer.id)
        assert active is not None
        self.assertEqual(active.record.id, restarted.record.id)
        self.assertEqual(inquiry_history(view, earlier.episode_id)[-1].receipt.state.status, "displaced")
        self.assertEqual(displaced_inquiries(view, observer_entity_id=self.observer.id), ())
        self.assertIsNone(displaced_candidate(self.owner, observer_entity_id=self.observer.id,
                                              grant=self.capture(self.new_epoch)))

    def test_admission_spends_one_opportunity_and_survives_restart(self) -> None:
        self.working()
        closed = self.displace()
        inner_turns_before = len(self.store.gwt_turns_of("inner", limit=10))
        restarted = self.admit(closed)
        # Admission allocates the slot; it executes no thought and projects no inner turn.
        self.assertEqual(len(self.store.gwt_turns_of("inner", limit=10)), inner_turns_before)
        outcomes = self.attention_outcomes("opportunity-one")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0].decision.displaced_receipt_id, closed.record.id)
        self.assertEqual(outcomes[0].decision.admitted_operation, "feasibility_check")
        restored = MemoryStore.load(self.path)
        view = restored.snapshot(capture=False)
        history = inquiry_history(view, restarted.receipt.state.episode_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].receipt, restarted.receipt)
        with self.assertRaisesRegex(ValueError, "exactly one retained outcome"):
            self.admit(closed)
        with self.assertRaisesRegex(GhostConflict, "already opened a new episode"):
            self.admit(closed, batch_id="opportunity-two")

    def test_an_active_inquiry_keeps_the_slot_against_a_displaced_question(self) -> None:
        self.working()
        closed = self.displace()
        self.select("different-question", grant=self.capture(self.new_epoch))
        with self.assertRaisesRegex(GhostConflict, "already holds an active episode"):
            self.admit(closed)

    def test_earlier_nodes_are_private_history_and_never_relation_targets(self) -> None:
        working = self.working()
        earlier_nodes = working.receipt.state.node_ids
        restarted = self.admit(self.displace())
        request = graph_thought_request(self.store.snapshot(capture=False), InnerRequest(), restarted,
                                       self.owner.execution_grant(restarted))
        self.assertEqual(request.nodes, ())
        self.assertEqual(tuple(artifact.record_id for artifact in request.earlier_artifacts), earlier_nodes)
        instruction = graph_thought_instruction(request)
        self.assertIn("not corroboration", instruction)
        self.assertIn(str(list(earlier_nodes)), instruction)
        citing = InquiryStep(operation="feasibility_check", thought="The earlier readings need a fresh look.",
            evidence_ids=(self.source.id, earlier_nodes[0]), assumptions=(), reality="inferred",
            judgments=self.judgments, relations=(), alternatives=(),
            progress=InquiryProgress(kind="none", inspected_source_ids=(), compared_node_ids=(),
                meaningful_difference=False, reason="Earlier thinking is not new evidence."),
            outcome="continue", reason="Generate the readings again in the present situation.",
            information_gap="Unknown motive.", next_operation="future_projection")
        self.assertEqual(validate_graph_step(request, citing), citing)
        related = citing.model_copy(update={"relations": (ThoughtRelation(target_record_id=earlier_nodes[0],
            kind="extends", reason="Continue the old node."),)})
        with self.assertRaisesRegex(ModelProposalError, "No graph node exists yet"):
            validate_graph_step(request, related)
        claimed = citing.model_copy(update={"progress": InquiryProgress(kind="new_evidence",
            inspected_source_ids=(earlier_nodes[0],), compared_node_ids=(), meaningful_difference=False,
            reason="Treat the earlier thought as acquired evidence.")})
        with self.assertRaisesRegex(ValueError, "Internal derivations cannot count"):
            stage_inquiry_step(self.store.begin_batch("fake-evidence", embed=False),
                parent_receipt_id=restarted.record.id, step=claimed, grant=self.owner.execution_grant(restarted),
                active_seconds=1, now_utc=self.now, limits=self.limits)
        accepted = self.owner.accept_step(restarted, citing, grant=self.owner.execution_grant(restarted),
            commit_id="restart-step", active_seconds=1, now_utc=self.now, mental_state=FullMentalState(),
            control_generation=0)
        # Citing the earlier thought resolves to the same original record and adds no root.
        self.assertEqual(accepted.receipt.state.root_source_ids, (self.source.id,))
        self.assertEqual(accepted.receipt.state.sequence, 1)
        self.assertEqual(accepted.receipt.state.stalled_steps, 1)

    def test_a_restarted_question_can_be_displaced_again_and_keeps_its_whole_history(self) -> None:
        first = self.working()
        restarted = self.admit(self.displace())
        second_step = self.generation().model_copy(update={"operation": "feasibility_check", "alternatives": (),
            "next_operation": "future_projection"})
        second = self.owner.accept_step(restarted, second_step, grant=self.owner.execution_grant(restarted),
            commit_id="second-episode-step", active_seconds=1, now_utc=self.now, mental_state=FullMentalState(),
            control_generation=0)
        latest_epoch = self.new_epoch + 1
        closed_again = self.owner.displace_superseded(observer_entity_id=self.observer.id, critical_epoch=latest_epoch)
        self.assertEqual(len(closed_again), 1)
        offered = self.capture(latest_epoch)
        candidate = displaced_candidate(self.owner, observer_entity_id=self.observer.id, grant=offered)
        assert candidate is not None
        self.assertEqual(candidate.offer.receipt_id, closed_again[0].record.id)
        self.assertEqual(candidate.offer.question, "What remains unknown about the sketch?")
        decision = ThoughtAttentionDecision(displaced_receipt_id=closed_again[0].record.id, disposition="select",
            admitted_operation="future_projection", reason="Still unresolved.")
        third = self.owner.admit_displaced_inquiry(closed_again[0], attention=decision, nomination_ids=(),
            offered_grant=offered, grant=candidate.grant, batch_id="opportunity-three",
            episode_id="inquiry:opportunity-three", now_utc=self.now,
            focus=captured_private_focus(self.store.snapshot(capture=False), self.now))
        request = graph_thought_request(self.store.snapshot(capture=False), InnerRequest(), third,
                                       self.owner.execution_grant(third))
        self.assertEqual(tuple(artifact.record_id for artifact in request.earlier_artifacts),
                         (*first.receipt.state.node_ids, *second.receipt.state.node_ids))
        self.assertEqual(third.receipt.state.grant.critical_epoch, latest_epoch)
        self.assertEqual(third.receipt.state.root_source_ids, (self.source.id,))

    def test_an_opportunity_closes_stale_work_durably_without_a_model_call(self) -> None:
        working = self.working()
        result = run_graph_opportunity(InnerRequest(), owner=self.owner, grant=self.capture(self.new_epoch),
            run_stage=forbidden_stage, mental_state=FullMentalState(), control_generation=0, pending_input_ids=(),
            focus=captured_private_focus(self.store.snapshot(capture=False), self.now))
        self.assertEqual(result.summary, "No private opportunity requires nomination.")
        restored = MemoryStore.load(self.path)
        view = restored.snapshot(capture=False)
        self.assertIsNone(active_inquiry(view, observer_entity_id=self.observer.id))
        waiting = displaced_inquiries(view, observer_entity_id=self.observer.id)
        self.assertEqual(len(waiting), 1)
        self.assertEqual(waiting[0].receipt.parent_receipt_id, working.record.id)


class ControlDisplacementTests(DisplacementFixture):
    """Drive the Critical epoch through the real control owner and its private commit fence."""

    def setUp(self) -> None:
        super().setUp()
        self.bus: BusSystem = BusSystem()
        self.control: CognitiveControl = CognitiveControl("control", self.bus, self.fail, time.monotonic, None,
                                                          ghost=self.store)
        self.control.enable_input_triage()
        self.input: UserInputProxy = UserInputProxy("user_input", self.bus, self.fail, time.monotonic, self.store,
                                                    lambda text: None)
        self.input.register_triage_input = self.control.register_input
        self.old_epoch = self.control.publication_epoch()
        self.grant = self.capture(self.old_epoch)
        #: The control generation the current private opportunity captured.
        self.captured_generation: int = self.control.generation

    def commit(self, batch: WriteBatch) -> None:
        """Accept private work only through the real control fence, as the private host does."""
        self.control.commit_private(batch, self.captured_generation)

    def assess_critical(self, text: str) -> int:
        """Commit real input and supply its Critical classification at the control boundary."""
        correlation = self.input.submit(text)
        source_id = next(item.id for item in self.store.all_features
                         if item.metadata.get("exchange_correlation") == correlation)
        view = self.store.recovery_snapshot(capture=False)
        fingerprints = {source_id: source_fingerprint(view.get(source_id))}
        self.control.accept_triage(EarlyTriageAssessment(
            source_ids=[source_id], source_fingerprints=fingerprints, guidance_tick=None,
            evidence=TriageEvidence(record_fingerprints=fingerprints, commit_ids=sorted(view.commit_ids),
                                    current_head_ids=[source_id]),
            result=TriageResult(items=[TriageItem(knoxel_id=source_id, category="Critical",
                                                 reasoning="Explicit scheduling-boundary input.")])))
        return source_id

    def accept_main_cycle(self, source_ids: list[int]) -> None:
        """Accept a real main state batch that claims the Critical source, without an LLM calculation."""
        batch = self.store.begin_batch(f"boundary-cycle-{len(self.store.states)}", new_tick=True, embed=False)
        batch.depend_on(source_ids)
        state = CycleState(tick_id=batch.tick_id, stimulus_groups=[StimulusGroup(stimulus_ids=source_ids)])
        batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, stimulus_ids=source_ids), state)
        self.control.commit(batch, None, self.control.generation, source_ids=source_ids)

    def test_accepted_critical_reconsideration_displaces_work_and_readmits_its_question(self) -> None:
        working = self.working()
        source = self.assess_critical("Stop. There is a fire in the kitchen right now.")
        # Until main cognition accepts the reconsideration, no private write lands,
        # so the stale inquiry is neither advanced nor closed early.
        self.assertTrue(self.control.reconsideration_required())
        with self.assertRaises(GhostCancelled):
            self.owner.displace_superseded(observer_entity_id=self.observer.id, critical_epoch=self.old_epoch + 1)
        still_active = active_inquiry(self.store.snapshot(capture=False), observer_entity_id=self.observer.id)
        assert still_active is not None
        self.assertEqual(still_active.record.id, working.record.id)

        self.accept_main_cycle([source])
        self.assertFalse(self.control.reconsideration_required())
        self.new_epoch = self.control.publication_epoch()
        self.assertGreater(self.new_epoch, self.old_epoch)
        # The next private opportunity captures the current authority.
        self.captured_generation = self.control.generation
        result = run_graph_opportunity(InnerRequest(), owner=self.owner, grant=self.capture(self.new_epoch),
            run_stage=forbidden_stage, mental_state=FullMentalState(), control_generation=self.captured_generation,
            pending_input_ids=(), focus=captured_private_focus(self.store.snapshot(capture=False), self.now))
        self.assertEqual(result.summary, "No private opportunity requires nomination.")
        view = self.store.snapshot(capture=False)
        self.assertIsNone(active_inquiry(view, observer_entity_id=self.observer.id))
        waiting = displaced_inquiries(view, observer_entity_id=self.observer.id)
        self.assertEqual(len(waiting), 1)
        displacement = waiting[0].receipt.displacement
        assert displacement is not None
        self.assertEqual(displacement.superseding_critical_epoch, self.new_epoch)
        self.assertEqual(waiting[0].receipt.state.grant.critical_epoch, self.old_epoch)
        self.assertEqual(waiting[0].receipt.state.node_ids, working.receipt.state.node_ids)

        restarted = self.admit(waiting[0])
        self.assertEqual(restarted.receipt.state.status, "active")
        self.assertEqual(restarted.receipt.state.grant.critical_epoch, self.new_epoch)
        self.assertEqual(restarted.receipt.state.sequence, 0)
        # A second opportunity at the same epoch leaves the restarted episode alone.
        self.assertEqual(self.owner.displace_superseded(observer_entity_id=self.observer.id,
                                                        critical_epoch=self.new_epoch), ())


if __name__ == "__main__":
    unittest.main()
