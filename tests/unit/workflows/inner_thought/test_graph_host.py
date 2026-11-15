"""Exercise graph-host acceptance with real stores and durable checkpoints.

Boundary proposals are explicit test inputs. These checks do not substitute
for the separate generated workflow gate.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from pydantic import ValidationError

from pm.memory.snapshot import GhostCancelled, GhostConflict
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.cognitive_records import ExpectationRecord
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.enums import EntityClass, FeatureType, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Entity, Feature
from pm.records.mental_state import FullMentalState
from pm.records.qualia_records import QualeRecord
from workflows.cognitive_control.schemas import ControlConfig
from workflows.inner_thought.graph_execution import (
    graph_thought_instruction, graph_thought_request, validate_graph_step,
)
from workflows.inner_thought.pipeline import ModelProposalError
from workflows.inner_thought.graph_host import (
    FrozenProjectionPayload, MENTAL_ACTION_KEY, PrivateInquiryOwner, accepted_source_ids, active_inquiry,
    attach_inner_projection, capture_felt_state, deliberate_inquiry, pending_findings,
)
from workflows.inner_thought.graph_outcomes import capture_graph_outcome
from workflows.inner_thought.inquiry import InquiryRecord, inquiry_history, stage_inquiry_step
from workflows.inner_thought.metacognition import thought_activity
from workflows.inner_thought.nominations import capture_inquiry_grant, stage_nominations
from workflows.inner_thought.request_views import InquiryThoughtView, RecentThoughtView, ThoughtActivityView
from workflows.inner_thought.schemas import (
    DeliberateAdmission, ExpectationUpdate, InnerRequest, InquiryAlternative, InquiryProgress, InquiryStep,
    IntegrationResult, NominationBatch, ThoughtAttentionDecision, ThoughtJudgments, ThoughtNomination,
    ThoughtRelation,
)
from workflows.focus_management.host import mental_outcome
from pm.records.memory_records import ActionSchemaRecord
from pm.records.enums import ActionType


class GraphHostTests(unittest.TestCase):
    """Keep selection, actual operations, and main appraisal separate."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path: Path = Path(directory.name) / "private-graph.pickle"
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
                                              checkpoint=self.path)
        batch = self.store.begin_batch("anchors", embed=False)
        self.observer: Entity = batch.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.partner: Entity = batch.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.source: Feature = batch.create(Feature(content="I left the sketch unfinished.",
            feature_type=FeatureType.Dialogue, entity_id=self.partner.id,
            provenance=Provenance(source_kind=SourceKind.User)))
        batch.preserve_on_recovery()
        self.commit(batch)
        self.cancel: Event = Event()
        self.owner: PrivateInquiryOwner = PrivateInquiryOwner(self.store, commit=self.commit,
            cancel=self.cancel, limits=ControlConfig())
        self.now: datetime = datetime(2026, 11, 12, 17, 0)
        self.judgments: ThoughtJudgments = ThoughtJudgments(interestingness=0.8, character_relevance=0.7,
            plausibility=0.3, consequence=0.4, desirability=0.1, action_suitability=0.2,
            reason="The unfinished sketch matters; its cause remains unknown.")
        self.grant = capture_inquiry_grant(self.store.snapshot(capture=False), observer_entity_id=self.observer.id,
            scope=RecordScope(), source_ids=[self.source.id], governing_ids=[self.source.id], critical_epoch=0)

    def commit(self, batch: WriteBatch) -> None:
        """Use actual durable production acceptance, with no inference substitute."""
        self.store.commit(batch)

    def begin(self) -> InquiryRecord:
        """Persist nomination and separate attention through the host adapter."""
        batch = self.store.begin_batch("nomination", embed=False)
        nomination = ThoughtNomination(text="The sketch might reflect hesitation or a new interest.",
            question="Which explanation has evidence?", evidence_ids=(self.source.id,), reality="inferred",
            judgments=self.judgments, suggested_operation="future_projection")
        records = stage_nominations(batch, [nomination], grant=self.grant, batch_id="nomination")
        batch.preserve_on_recovery()
        self.commit(batch)
        selected = self.owner.accept_attention([records[0].record.id],
            ThoughtAttentionDecision(nomination_id=records[0].record.id, disposition="select",
                reason="This unresolved question fits current private attention."),
            commit_id="attention", episode_id="private-sketch", now_utc=self.now)
        assert selected.inquiry is not None
        return selected.inquiry

    def generation(self) -> InquiryStep:
        """Supply possible interpretations without claiming a conclusion."""
        branches = tuple(InquiryAlternative(text=text, evidence_ids=(self.source.id,), assumptions=(assumption,),
            reality="imagined", judgments=self.judgments, disposition="open", reason="A possible cause, not observation.")
            for text, assumption in [("Jesse revisits the sketch.", "Interest remains."),
                                     ("Jesse chooses another subject.", "A new interest matters more.")])
        return InquiryStep(operation="future_projection", thought="Both explanations remain possible.",
            evidence_ids=(self.source.id,), assumptions=(), reality="inferred", judgments=self.judgments,
            relations=(), alternatives=branches,
            progress=InquiryProgress(kind="none", inspected_source_ids=(), compared_node_ids=(),
                meaningful_difference=False, reason="Possibilities provide no new outside evidence."),
            outcome="continue", reason="Compare the assumptions before concluding.", information_gap="Unknown motive.",
            next_operation="feasibility_check")

    def advance(self, parent: InquiryRecord, step: InquiryStep, commit_id: str) -> InquiryRecord:
        """Accept a real graph transition with exact source and parent guards."""
        return self.owner.accept_step(parent, step, grant=self.owner.execution_grant(parent),
            commit_id=commit_id, active_seconds=1.5, now_utc=self.now,
            mental_state=FullMentalState(), control_generation=0)

    def test_attention_allocates_slot_without_gwt_or_executed_thought(self) -> None:
        selected = self.begin()
        restored = MemoryStore.load(self.path)
        active = active_inquiry(restored.snapshot(capture=False), observer_entity_id=self.observer.id)
        assert active is not None
        self.assertEqual(active.record.id, selected.record.id)
        self.assertEqual(active.receipt.state.node_ids, ())
        self.assertEqual(restored.gwt_turns, {})
        self.assertEqual(pending_findings(restored), ())

    def test_generation_retains_graph_and_derived_inner_turn_without_main_finding(self) -> None:
        selected = self.begin()
        generated = self.advance(selected, self.generation(), "generation")
        turn = self.store.gwt_turns_of("inner", limit=1)[0]
        restored = MemoryStore.load(self.path)
        self.assertEqual(turn.knoxel_ids, [generated.receipt.operation_record_id])
        self.assertEqual(turn.summary, self.generation().thought)
        self.assertEqual(len(generated.receipt.state.node_ids), 3)
        self.assertEqual(pending_findings(restored), ())
        grant = self.owner.execution_grant(generated)
        request = graph_thought_request(restored.snapshot(capture=False), InnerRequest(), generated, grant)
        self.assertEqual(tuple(node.record_id for node in request.nodes), generated.receipt.state.node_ids)
        self.assertEqual(request.state.next_operation, "feasibility_check")

    def test_outcome_replay_stops_only_after_main_claims_actual_finding(self) -> None:
        generated = self.advance(self.begin(), self.generation(), "generation")
        node_ids = generated.receipt.state.node_ids
        comparison = InquiryStep(operation="feasibility_check", thought="The report does not identify a motive.",
            evidence_ids=(self.source.id,), assumptions=(), reality="inferred", judgments=self.judgments,
            relations=tuple(ThoughtRelation(target_record_id=node_id, kind="merges", reason="Compare uncertain causes.")
                            for node_id in node_ids), alternatives=(),
            progress=InquiryProgress(kind="discrimination", inspected_source_ids=(), compared_node_ids=node_ids,
                meaningful_difference=True, reason="Each possible cause predicts a different next choice."),
            outcome="information_gap", reason="The missing motive prevents a conclusion.",
            information_gap="Jesse's reason for stopping.", next_operation="")
        final = self.advance(generated, comparison, "comparison")
        finding_id = final.receipt.operation_record_id
        assert finding_id is not None
        restored = MemoryStore.load(self.path)
        self.assertEqual(tuple(record.id for record in pending_findings(restored)), (finding_id,))
        self.assertEqual(tuple(record.id for record in pending_findings(restored)), (finding_id,))
        main = restored.begin_batch("main-appraisal", new_tick=True, embed=False)
        main.depend_on([finding_id])
        main.set_gwt(GwtTurnRecord(tick_id=main.tick_id, slot_id="main", knoxel_ids=[finding_id],
                                  summary="Appraise the actual private finding."),
            CycleState(tick_id=main.tick_id, mental_state=FullMentalState(),
                       stimulus_groups=[StimulusGroup(stimulus_ids=[finding_id])]))
        restored.commit(main)
        self.assertEqual(pending_findings(restored), ())
        self.assertIsNone(active_inquiry(restored.snapshot(capture=False), observer_entity_id=self.observer.id))

    def test_cancellation_does_not_commit_graph_or_inner_turn(self) -> None:
        selected = self.begin()
        before = self.path.read_bytes()
        self.cancel.set()
        with self.assertRaises(GhostCancelled):
            self.advance(selected, self.generation(), "cancelled")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.store.gwt_turns, {})
        self.assertEqual(len(inquiry_history(self.store.snapshot(capture=False), "private-sketch")), 1)

    def test_governing_source_successor_rejects_without_late_graph_or_gwt(self) -> None:
        selected = self.begin()
        replacement = self.store.begin_batch("source-correction", embed=False)
        corrected = replacement.view().get(self.source.id).copy_for_edit()
        corrected.content = "I finished the sketch."
        replacement.replace(corrected)
        self.store.commit(replacement)
        before = self.path.read_bytes()
        with self.assertRaises(GhostConflict):
            self.advance(selected, self.generation(), "stale-generation")
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.store.gwt_turns, {})
        self.assertEqual(pending_findings(self.store), ())

    def test_operational_source_revision_keeps_original_dependency(self) -> None:
        selected = self.begin()
        original = next(source for source in self.grant.sources if source.record_id == self.source.id)
        lifecycle = self.store.begin_batch("exchange-resolved", embed=False)
        lifecycle.set_lifecycle(self.source.id, exchange_status="resolved")
        self.store.commit(lifecycle)
        grant = self.owner.execution_grant(selected)
        retained = next(source for source in grant.sources if source.record_id == self.source.id)
        self.assertEqual(retained, original)
        self.assertGreater(self.store.revisions[self.source.id], original.revision)
        accepted = self.advance(selected, self.generation(), "generation-after-operational-revision")
        self.assertEqual(accepted.receipt.state.sequence, 1)

    def test_missing_inner_turn_repairs_from_accepted_graph_without_reexecution(self) -> None:
        selected = self.begin()
        step = self.generation()
        batch = self.store.begin_batch("graph-without-projection", new_tick=True, embed=False)
        accepted = stage_inquiry_step(batch, parent_receipt_id=selected.record.id, step=step,
            grant=self.owner.execution_grant(selected), active_seconds=1.5, now_utc=self.now, limits=ControlConfig())
        operation_id = accepted.receipt.operation_record_id
        assert operation_id is not None
        operation = batch.view().get(operation_id)
        assert isinstance(operation, Feature)
        attach_inner_projection(operation, FrozenProjectionPayload(mental_state=FullMentalState()))
        batch.revise(operation)
        batch.preserve_on_recovery()
        self.commit(batch)
        restored = MemoryStore.load(self.path)
        self.assertEqual(restored.gwt_turns, {})
        history = inquiry_history(restored.snapshot(capture=False), "private-sketch")
        self.assertEqual(history[-1].record.id, accepted.record.id)

        def commit_restored(projection: WriteBatch) -> None:
            """Apply the actual durable projection through the restored store."""
            restored.commit(projection)

        restored_owner = PrivateInquiryOwner(restored, commit=commit_restored, cancel=Event(), limits=ControlConfig())
        restored_owner.ensure_inner_turn(history[-1])
        first_projection = self.path.read_bytes()
        restored_owner.ensure_inner_turn(history[-1])
        self.assertEqual(self.path.read_bytes(), first_projection)
        self.assertEqual(restored.gwt_turns_of("inner", limit=1)[0].knoxel_ids, [accepted.receipt.operation_record_id])
        self.assertEqual(len(inquiry_history(restored.snapshot(capture=False), "private-sketch")), 2)

    def test_clean_restart_after_projection_rebuilds_once_from_same_operation(self) -> None:
        generated = self.advance(self.begin(), self.generation(), "generation")
        operation_id = generated.receipt.operation_record_id
        assert operation_id is not None
        original_operation = self.store.snapshot(capture=False).get(operation_id).model_dump_json()
        restored = MemoryStore.load(self.path)
        accepted = inquiry_history(restored.snapshot(capture=False), "private-sketch")[-1]

        def commit_restored(projection: WriteBatch) -> None:
            """Persist the derived projection against the actual restored store."""
            restored.commit(projection)

        restored_owner = PrivateInquiryOwner(restored, commit=commit_restored, cancel=Event(), limits=ControlConfig())
        restored_owner.ensure_inner_turn(accepted)
        after_repair = self.path.read_bytes()
        restored_owner.ensure_inner_turn(accepted)
        self.assertEqual(self.path.read_bytes(), after_repair)
        self.assertEqual(restored.snapshot(capture=False).get(operation_id).model_dump_json(), original_operation)
        self.assertEqual(restored.gwt_turns_of("inner", limit=1)[0].knoxel_ids, [operation_id])
        self.assertEqual(len(inquiry_history(restored.snapshot(capture=False), "private-sketch")), 2)

    def test_main_turn_does_not_mask_missing_inner_projection(self) -> None:
        selected = self.begin()
        graph = self.store.begin_batch("graph-before-main", new_tick=True, embed=False)
        accepted = stage_inquiry_step(graph, parent_receipt_id=selected.record.id, step=self.generation(),
            grant=self.owner.execution_grant(selected), active_seconds=1.5, now_utc=self.now, limits=ControlConfig())
        operation_id = accepted.receipt.operation_record_id
        assert operation_id is not None
        operation = graph.view().get(operation_id)
        assert isinstance(operation, Feature)
        attach_inner_projection(operation, FrozenProjectionPayload(mental_state=FullMentalState()))
        graph.revise(operation)
        graph.preserve_on_recovery()
        self.commit(graph)
        operation_id = accepted.receipt.operation_record_id
        assert operation_id is not None
        main = self.store.begin_batch("main-sees-operation", new_tick=True, embed=False)
        main.depend_on([operation_id])
        main.set_gwt(GwtTurnRecord(tick_id=main.tick_id, slot_id="main", knoxel_ids=[operation_id],
                                  summary="Main encountered the accepted operation."),
                     CycleState(tick_id=main.tick_id, mental_state=FullMentalState()))
        self.commit(main)
        self.assertEqual(self.store.gwt_turns_of("inner", limit=1), [])
        self.owner.ensure_inner_turn(accepted)
        self.assertEqual(self.store.gwt_turns_of("inner", limit=1)[0].knoxel_ids, [operation_id])

    def test_restart_repair_uses_original_state_after_new_main_state(self) -> None:
        selected = self.begin()
        original = FullMentalState()
        original.state_emotions.joy = 0.1
        accepted = self.owner.accept_step(selected, self.generation(), grant=self.owner.execution_grant(selected),
            commit_id="original-state", active_seconds=1.5, now_utc=self.now,
            mental_state=original, control_generation=0)
        operation_id = accepted.receipt.operation_record_id
        assert operation_id is not None
        original.state_emotions.joy = 0.7  # Later caller mutation cannot rewrite the retained payload.
        newer = FullMentalState()
        newer.state_emotions.joy = 0.9
        main = self.store.begin_batch("new-main-state", new_tick=True, embed=False)
        main.depend_on([operation_id])
        main.set_gwt(GwtTurnRecord(tick_id=main.tick_id, slot_id="main", knoxel_ids=[operation_id],
                                  summary="A later accepted state."),
                     CycleState(tick_id=main.tick_id, mental_state=newer))
        self.commit(main)
        restored = MemoryStore.load(self.path)
        self.assertEqual(restored.read_cycles()[-1].mental_state.state_emotions.joy, 0.9)
        retained = inquiry_history(restored.snapshot(capture=False), "private-sketch")[-1]

        def commit_restored(projection: WriteBatch) -> None:
            """Persist the derived turn through the actual restored store."""
            restored.commit(projection)

        owner = PrivateInquiryOwner(restored, commit=commit_restored, cancel=Event(), limits=ControlConfig())
        owner.ensure_inner_turn(retained)
        turn = restored.gwt_turns_of("inner", limit=1)[0]
        cycle = next(item for item in restored.read_cycles() if item.tick_id == turn.tick_id)
        self.assertEqual(cycle.mental_state.state_emotions.joy, 0.1)
        self.assertEqual(retained.receipt.operation_record_id, operation_id)

    def test_missing_original_projection_fails_without_invented_state(self) -> None:
        selected = self.begin()
        graph = self.store.begin_batch("missing-projection-payload", new_tick=True, embed=False)
        accepted = stage_inquiry_step(graph, parent_receipt_id=selected.record.id, step=self.generation(),
            grant=self.owner.execution_grant(selected), active_seconds=1.5, now_utc=self.now, limits=ControlConfig())
        graph.preserve_on_recovery()
        self.commit(graph)
        before = self.path.read_bytes()
        with self.assertRaises(ValidationError):
            self.owner.ensure_inner_turn(accepted)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.store.gwt_turns_of("inner", limit=1), [])

    def comparison(self, node_ids: tuple[int, ...], *, evidence_ids: tuple[int, ...]) -> InquiryStep:
        """Compare the generated nodes and stop at an information gap."""
        return InquiryStep(operation="feasibility_check", thought="The report does not identify a motive.",
            evidence_ids=evidence_ids, assumptions=(), reality="inferred", judgments=self.judgments,
            relations=tuple(ThoughtRelation(target_record_id=node_id, kind="merges", reason="Compare uncertain causes.")
                            for node_id in node_ids), alternatives=(),
            progress=InquiryProgress(kind="discrimination", inspected_source_ids=(), compared_node_ids=node_ids,
                meaningful_difference=True, reason="Each possible cause predicts a different next choice."),
            outcome="information_gap", reason="The missing motive prevents a conclusion.",
            information_gap="Jesse's reason for stopping.", next_operation="")

    def inner_cycle(self, store: MemoryStore, operation_id: int) -> tuple[GwtTurnRecord, CycleState]:
        """Read the one inner turn that projects an operation, with its cycle."""
        turns = [turn for turn in store.gwt_turns.values()
                 if turn.slot_id == "inner" and operation_id in turn.knoxel_ids]
        self.assertEqual(len(turns), 1)
        cycle = next(item for item in store.read_cycles() if item.tick_id == turns[0].tick_id)
        return turns[0], cycle

    def moved_state(self) -> FullMentalState:
        """Return a state whose joy moved from 0 to 1, which moves valence by 1/5."""
        moved = FullMentalState()
        moved.state_emotions.joy = 1.0
        return moved

    def qualia(self, store: MemoryStore) -> list[QualeRecord]:
        """Read every committed felt-state record."""
        return [record for record in store.snapshot(capture=False).all_knoxels.values()
                if isinstance(record, QualeRecord)]

    def test_first_accepted_operation_samples_every_channel_without_a_quale(self) -> None:
        generated = self.advance(self.begin(), self.generation(), "generation")
        operation_id = generated.receipt.operation_record_id
        assert operation_id is not None
        turn, cycle = self.inner_cycle(self.store, operation_id)
        self.assertEqual(set(cycle.qualia_channels), {"affect.valence", "affect.arousal",
            "relationship.warmth", "attention.engagement", "goal.conflict"})
        self.assertEqual(cycle.timestamp, self.now)
        for channel in cycle.qualia_channels.values():
            self.assertEqual((channel.velocity, channel.acceleration), (0.0, 0.0))
            self.assertEqual(channel.last_driver_ids, [self.source.id])
        # A first sample has no measured movement, so it emits no record.
        self.assertEqual(turn.knoxel_ids, [operation_id])

    def test_channel_movement_between_accepted_operations_emits_a_reportable_quale(self) -> None:
        generated = self.advance(self.begin(), self.generation(), "generation")
        final = self.owner.accept_step(generated,
            self.comparison(generated.receipt.state.node_ids, evidence_ids=(self.source.id,)),
            grant=self.owner.execution_grant(generated), commit_id="comparison", active_seconds=1.5,
            now_utc=self.now + timedelta(seconds=2), mental_state=self.moved_state(), control_generation=0)
        operation_id = final.receipt.operation_record_id
        assert operation_id is not None
        turn, cycle = self.inner_cycle(self.store, operation_id)
        # Joy 0 -> 1 moves valence by 1/5 over two seconds.
        self.assertAlmostEqual(cycle.qualia_channels["affect.valence"].position, 0.2)
        self.assertAlmostEqual(cycle.qualia_channels["affect.valence"].velocity, 0.1)
        self.assertEqual(cycle.timestamp, self.now + timedelta(seconds=2))
        qualia = self.qualia(self.store)
        self.assertEqual([quale.channel_id for quale in qualia], ["affect.valence"])
        quale = qualia[0]
        self.assertEqual(turn.knoxel_ids, [operation_id, quale.id])
        self.assertAlmostEqual(quale.position, 0.2)
        self.assertAlmostEqual(quale.velocity, 0.1)
        self.assertEqual(quale.driver_ids, [self.source.id])
        self.assertTrue(quale.reportable)
        self.assertEqual(quale.scope, self.store.snapshot(capture=False).get(operation_id).scope)
        # The finding is still the operation, never the felt-state record.
        self.assertEqual(tuple(record.id for record in pending_findings(self.store)), (operation_id,))

    def test_movement_without_a_cited_driver_stays_unreportable(self) -> None:
        generated = self.advance(self.begin(), self.generation(), "generation")
        self.owner.accept_step(generated, self.comparison(generated.receipt.state.node_ids, evidence_ids=()),
            grant=self.owner.execution_grant(generated), commit_id="comparison", active_seconds=1.5,
            now_utc=self.now + timedelta(seconds=2), mental_state=self.moved_state(), control_generation=0)
        qualia = self.qualia(self.store)
        self.assertEqual([quale.channel_id for quale in qualia], ["affect.valence"])
        self.assertEqual(qualia[0].driver_ids, [])
        self.assertFalse(qualia[0].reportable)

    def test_repair_emits_the_retained_sample_after_a_later_accepted_state(self) -> None:
        generated = self.advance(self.begin(), self.generation(), "generation")
        moved = self.moved_state()
        sampled_at = self.now + timedelta(seconds=2)
        step = self.comparison(generated.receipt.state.node_ids, evidence_ids=(self.source.id,))
        graph = self.store.begin_batch("graph-without-projection", new_tick=True, embed=False)
        accepted = stage_inquiry_step(graph, parent_receipt_id=generated.record.id, step=step,
            grant=self.owner.execution_grant(generated), active_seconds=1.5, now_utc=sampled_at,
            limits=ControlConfig())
        operation_id = accepted.receipt.operation_record_id
        assert operation_id is not None
        operation = graph.view().get(operation_id)
        assert isinstance(operation, Feature)
        retained = capture_felt_state(self.store, moved, driver_ids=step.evidence_ids, now_utc=sampled_at)
        attach_inner_projection(operation, FrozenProjectionPayload(mental_state=moved, felt_state=retained))
        graph.revise(operation)
        graph.preserve_on_recovery()
        self.commit(graph)
        # A later accepted main state must not replace the retained sample.
        later = FullMentalState()
        later.state_emotions.sadness = 1.0
        main = self.store.begin_batch("later-main-state", new_tick=True, embed=False)
        main.depend_on([operation_id])
        main.set_gwt(GwtTurnRecord(tick_id=main.tick_id, slot_id="main", knoxel_ids=[operation_id],
                                  summary="A later accepted state."),
                     CycleState(tick_id=main.tick_id, mental_state=later))
        self.commit(main)
        restored = MemoryStore.load(self.path)
        history = inquiry_history(restored.snapshot(capture=False), "private-sketch")

        def commit_restored(projection: WriteBatch) -> None:
            """Persist the derived turn through the actual restored store."""
            restored.commit(projection)

        owner = PrivateInquiryOwner(restored, commit=commit_restored, cancel=Event(), limits=ControlConfig())
        owner.ensure_inner_turn(history[-1])
        after_repair = self.path.read_bytes()
        owner.ensure_inner_turn(history[-1])
        self.assertEqual(self.path.read_bytes(), after_repair)
        turn, cycle = self.inner_cycle(restored, operation_id)
        self.assertEqual(cycle.qualia_channels, retained.channels)
        self.assertEqual(cycle.timestamp, sampled_at)
        self.assertEqual(cycle.mental_state.state_emotions.joy, 1.0)
        qualia = self.qualia(restored)
        self.assertEqual(len(qualia), 1)
        self.assertAlmostEqual(qualia[0].velocity, 0.1)
        self.assertEqual(turn.knoxel_ids, [operation_id, qualia[0].id])

    def test_operation_accepted_before_sampling_projects_no_invented_channels(self) -> None:
        selected = self.begin()
        graph = self.store.begin_batch("graph-before-sampling", new_tick=True, embed=False)
        accepted = stage_inquiry_step(graph, parent_receipt_id=selected.record.id, step=self.generation(),
            grant=self.owner.execution_grant(selected), active_seconds=1.5, now_utc=self.now, limits=ControlConfig())
        operation_id = accepted.receipt.operation_record_id
        assert operation_id is not None
        operation = graph.view().get(operation_id)
        assert isinstance(operation, Feature)
        attach_inner_projection(operation, FrozenProjectionPayload(mental_state=FullMentalState()))
        graph.revise(operation)
        graph.preserve_on_recovery()
        self.commit(graph)
        self.owner.ensure_inner_turn(accepted)
        turn, cycle = self.inner_cycle(self.store, operation_id)
        self.assertEqual(cycle.qualia_channels, {})
        self.assertEqual(turn.knoxel_ids, [operation_id])

    def test_expectation_opened_by_an_outcome_is_not_citable_until_main_accepts_it(self) -> None:
        generated = self.advance(self.begin(), self.generation(), "generation")
        step = self.comparison(generated.receipt.state.node_ids, evidence_ids=(self.source.id,))
        grant = self.owner.execution_grant(generated)
        integration = IntegrationResult(expectation=ExpectationUpdate(action="open",
            anticipation="Jesse explains the unfinished sketch.", expected_outcome="A reason is given."))
        final = self.owner.accept_step(generated, step, grant=grant, commit_id="comparison", active_seconds=1.5,
            now_utc=self.now, mental_state=FullMentalState(), control_generation=0,
            outcome=capture_graph_outcome(grant, step, integration))
        finding_id = final.receipt.operation_record_id
        assert finding_id is not None
        opened = [record for record in self.store.snapshot(capture=False).all_knoxels.values()
                  if isinstance(record, ExpectationRecord)]
        self.assertEqual(len(opened), 1)
        expectation_id = opened[0].id
        # The finding is a recovery root. The expectation lives in the derived inner-turn commit.
        accepted = accepted_source_ids(self.store)
        self.assertIn(finding_id, accepted)
        self.assertNotIn(expectation_id, accepted)
        # This is why it must stay out of a private grant: the store refuses the next graph commit.
        unaccepted_grant = capture_inquiry_grant(self.store.snapshot(capture=False),
            observer_entity_id=self.observer.id, scope=RecordScope(), source_ids=[expectation_id], critical_epoch=0)
        with self.assertRaisesRegex(ValueError, "Recovery root depends on unaccepted transactions"):
            self.owner.accept_nominations(NominationBatch(nominations=(), reason="Nothing attracted attention."),
                                          grant=unaccepted_grant, batch_id="cites-unaccepted-expectation")
        # A main cycle that read the expectation accepts the commit that produced it.
        main = self.store.begin_batch("main-reads-expectation", new_tick=True, embed=False)
        main.depend_on([finding_id, expectation_id])
        main.set_gwt(GwtTurnRecord(tick_id=main.tick_id, slot_id="main", knoxel_ids=[finding_id],
                                  summary="Appraise the finding and its expectation."),
                     CycleState(tick_id=main.tick_id, mental_state=FullMentalState(),
                                stimulus_groups=[StimulusGroup(stimulus_ids=[finding_id])]))
        self.commit(main)
        self.assertIn(expectation_id, accepted_source_ids(self.store))
        self.owner.accept_nominations(NominationBatch(nominations=(), reason="Nothing attracted attention."),
                                      grant=unaccepted_grant, batch_id="cites-accepted-expectation")

    def test_comparison_input_keeps_original_branch_assumptions_and_judgments(self) -> None:
        generated = self.advance(self.begin(), self.generation(), "generation")
        request = graph_thought_request(self.store.snapshot(capture=False), InnerRequest(), generated,
                                       self.owner.execution_grant(generated))
        branches = [node.payload for node in request.nodes if isinstance(node.payload, InquiryAlternative)]
        self.assertEqual(tuple(branches), self.generation().alternatives)
        self.assertTrue(all(branch.assumptions and branch.judgments == self.judgments for branch in branches))

    def test_child_semantic_validation_rejects_unknown_citations_and_early_finding(self) -> None:
        selected = self.begin()
        request = graph_thought_request(self.store.snapshot(capture=False), InnerRequest(), selected,
                                       self.owner.execution_grant(selected))
        generation = self.generation()
        self.assertEqual(validate_graph_step(request, generation), generation)
        with self.assertRaises(ModelProposalError):
            validate_graph_step(request, generation.model_copy(update={"evidence_ids": (999999,)}))
        with self.assertRaises(ModelProposalError):
            validate_graph_step(request, generation.model_copy(update={"outcome": "finding", "next_operation": ""}))

    def test_discrimination_claim_names_the_wrong_operation_mode_separately_from_the_count(self) -> None:
        generation = self.generation().model_copy(update={"next_operation": "counterfactual"})
        generated = self.advance(self.begin(), generation, "generation")
        node_ids = generated.receipt.state.node_ids
        request = graph_thought_request(self.store.snapshot(capture=False), InnerRequest(), generated,
                                       self.owner.execution_grant(generated))
        instruction = graph_thought_instruction(request)
        self.assertIn("A generative operation cannot claim progress.kind='discrimination'", instruction)
        relations = tuple(ThoughtRelation(target_record_id=node_id, kind="notices", reason="Survey the earlier node.")
                          for node_id in node_ids)
        # The model compared three nodes, so a message about the count cannot help it.
        surveyed = InquiryStep(operation="counterfactual", thought="A different path stays imaginable.",
            evidence_ids=(self.source.id,), assumptions=(), reality="imagined", judgments=self.judgments,
            relations=relations, alternatives=(),
            progress=InquiryProgress(kind="discrimination", inspected_source_ids=(), compared_node_ids=node_ids,
                meaningful_difference=True, reason="The earlier nodes differ."),
            outcome="continue", reason="Compare next.", information_gap="Unknown motive.",
            next_operation="feasibility_check")
        with self.assertRaisesRegex(ModelProposalError,
                "counterfactual is a generative operation, so it cannot claim discrimination progress") as raised:
            validate_graph_step(request, surveyed)
        self.assertIn("progress.kind='none'", str(raised.exception))
        corrected = surveyed.model_copy(update={"progress": InquiryProgress(kind="none", inspected_source_ids=(),
            compared_node_ids=(), meaningful_difference=False, reason="This step only generates.")})
        self.assertEqual(validate_graph_step(request, corrected), corrected)

    def test_discriminatory_claim_must_compare_two_accepted_nodes(self) -> None:
        generated = self.advance(self.begin(), self.generation(), "generation")
        node_ids = generated.receipt.state.node_ids
        request = graph_thought_request(self.store.snapshot(capture=False), InnerRequest(), generated,
                                       self.owner.execution_grant(generated))
        self.assertIn("at least two accepted graph nodes", graph_thought_instruction(request))
        single = InquiryStep(operation="feasibility_check", thought="Only one reading was weighed.",
            evidence_ids=(self.source.id,), assumptions=(), reality="inferred", judgments=self.judgments,
            relations=(ThoughtRelation(target_record_id=node_ids[0], kind="notices", reason="Weigh it."),),
            alternatives=(), progress=InquiryProgress(kind="discrimination", inspected_source_ids=(),
                compared_node_ids=(node_ids[0],), meaningful_difference=True, reason="One node only."),
            outcome="continue", reason="Compare more.", information_gap="Unknown motive.",
            next_operation="future_projection")
        with self.assertRaisesRegex(ModelProposalError, "must compare at least two accepted nodes"):
            validate_graph_step(request, single)

    def test_earlier_thought_claimed_as_new_evidence_is_corrected_inside_the_loop(self) -> None:
        # B07 D1: the 2026-11-15 live crash. The model claimed new evidence for
        # an earlier thought of the same episode.
        generated = self.advance(self.begin(), self.generation(), "generation")
        earlier_thought = generated.receipt.state.node_ids[0]
        grant = self.owner.execution_grant(generated)
        request = graph_thought_request(self.store.snapshot(capture=False), InnerRequest(), generated, grant)
        self.assertIn(self.source.id, request.original_source_ids)
        self.assertNotIn(earlier_thought, request.original_source_ids)
        relations = tuple(ThoughtRelation(target_record_id=node_id, kind="merges", reason="Compare the causes.")
                          for node_id in generated.receipt.state.node_ids)
        claim = InquiryStep(operation="feasibility_check", thought="My earlier thought shows the cause.",
            evidence_ids=(self.source.id, earlier_thought), assumptions=(), reality="inferred",
            judgments=self.judgments, relations=relations, alternatives=(),
            progress=InquiryProgress(kind="new_evidence", inspected_source_ids=(earlier_thought,),
                compared_node_ids=(), meaningful_difference=False, reason="I read my earlier thought."),
            outcome="continue", reason="Look further.", information_gap="Unknown motive.",
            next_operation="future_projection")
        with self.assertRaisesRegex(ModelProposalError, rf"names \[{earlier_thought}\], which are earlier thoughts") as raised:
            validate_graph_step(request, claim)
        self.assertIn(f"The original source IDs of this step are {sorted(request.original_source_ids)}",
                      str(raised.exception))
        self.assertIn("progress.kind='none'", str(raised.exception))
        # The store owner keeps its own check, unchanged.
        batch = self.store.begin_batch("store-owner-direct", new_tick=True, embed=False)
        with self.assertRaisesRegex(ValueError, "Internal derivations cannot count as acquired original evidence"):
            stage_inquiry_step(batch, parent_receipt_id=generated.record.id, step=claim, grant=grant,
                active_seconds=1.0, now_utc=self.now, limits=ControlConfig())
        # The original source itself is accepted by both.
        original = claim.model_copy(update={"progress": InquiryProgress(kind="new_evidence",
            inspected_source_ids=(self.source.id,), compared_node_ids=(), meaningful_difference=False,
            reason="I read the message again.")})
        self.assertEqual(validate_graph_step(request, original), original)

    def test_forced_stall_bound_does_not_publish_generated_possibilities_as_finding(self) -> None:
        self.owner.limits = ControlConfig(stalled_steps=1)
        generated = self.advance(self.begin(), self.generation(), "generation-at-bound")
        self.assertEqual(generated.receipt.state.status, "paused")
        self.assertTrue(generated.receipt.state.pending_generation_ids)
        self.assertEqual(pending_findings(self.store), ())


class MetacognitionDeliveryTests(GraphHostTests):
    """A noticing is an intermediate result: delivered once, never an action close."""

    def begin_with(self, operation: str) -> InquiryRecord:
        """Persist a spontaneous nomination and attention for the given operation."""
        batch = self.store.begin_batch("nomination", embed=False)
        nomination = ThoughtNomination(text="The short reply keeps returning.",
            question="Why does it keep returning without new evidence?", evidence_ids=(self.source.id,),
            reality="inferred", judgments=self.judgments, suggested_operation=operation)
        records = stage_nominations(batch, [nomination], grant=self.grant, batch_id="nomination")
        batch.preserve_on_recovery()
        self.commit(batch)
        selected = self.owner.accept_attention([records[0].record.id],
            ThoughtAttentionDecision(nomination_id=records[0].record.id, disposition="select",
                reason="Inspect the returning reply."),
            commit_id="attention", episode_id="private-return", now_utc=self.now)
        assert selected.inquiry is not None
        return selected.inquiry

    def noticing_step(self, operation: str) -> InquiryStep:
        """One continuing step of the admitted operation, with its recorded activity."""
        return InquiryStep(operation=operation, thought="The reply returns without new material.",
            evidence_ids=(self.source.id,), assumptions=(), reality="inferred", judgments=self.judgments,
            relations=(), alternatives=(),
            progress=InquiryProgress(kind="none", inspected_source_ids=(), compared_node_ids=(),
                meaningful_difference=False, reason="No new original evidence yet."),
            outcome="continue", reason="The step read its recorded activity.", information_gap="",
            next_operation="future_projection")

    def activity(self) -> ThoughtActivityView:
        """Recorded activity facts: one window with one stalled step on the source."""
        step = RecentThoughtView(knoxel_id=100, text="The reply returned again.", revision=1,
            operation="emotional_debrief", source_ids=[self.source.id], loop="inquiry",
            inquiry=InquiryThoughtView(episode_id="private-return", outcome="continue", reason="No gain.",
                information_gap="", progress_kind="none"))
        return thought_activity([step])

    def test_rumination_notice_is_pending_and_keeps_its_activity(self) -> None:
        head = self.begin_with("rumination_notice")
        activity = self.activity()
        accepted = self.owner.accept_step(head, self.noticing_step("rumination_notice"),
            grant=self.owner.execution_grant(head), commit_id="step-notice", active_seconds=1.5,
            now_utc=self.now, mental_state=FullMentalState(), control_generation=0, activity=activity)
        assert accepted.receipt.operation_record_id is not None
        operation_id = accepted.receipt.operation_record_id
        restored = MemoryStore.load(self.path)
        self.assertIn(operation_id, {item.id for item in pending_findings(restored)})
        operation = restored.snapshot(capture=False).get(operation_id)
        self.assertEqual(operation.metadata["thought_activity"]["window_step_ids"], [100])
        self.assertIsNone(operation.metadata.get(MENTAL_ACTION_KEY))

    def test_future_projection_continue_is_not_pending(self) -> None:
        head = self.begin_with("future_projection")
        accepted = self.owner.accept_step(head, self.noticing_step("future_projection"),
            grant=self.owner.execution_grant(head), commit_id="step-projection", active_seconds=1.5,
            now_utc=self.now, mental_state=FullMentalState(), control_generation=0)
        assert accepted.receipt.operation_record_id is not None
        self.assertNotIn(accepted.receipt.operation_record_id, {item.id for item in pending_findings(self.store)})

    def test_noticing_inside_a_deliberate_episode_closes_no_action(self) -> None:
        batch = self.store.begin_batch("dispatch", embed=False)
        action = batch.create(ActionSchemaRecord(content="I want to notice this on my own.",
            action_type=ActionType.Loopback, provenance=Provenance(source_kind=SourceKind.Workflow),
            metadata={"action_id": "action-notice", "workflow_name": "theory_of_mind",
                      "dispatch_status": "pending", "dispatch_target": "theory_of_mind",
                      "dispatch_correlation": "corr-notice", "evidence_ids": [self.source.id]}))
        batch.preserve_on_recovery()
        self.commit(batch)
        view = self.store.snapshot(capture=False)
        grant = capture_inquiry_grant(view, observer_entity_id=self.observer.id, scope=RecordScope(),
            source_ids=[action.id, self.source.id], critical_epoch=0)
        admission = DeliberateAdmission(action_record_id=action.id,
            instruction="Notice whether the reply keeps returning.", evidence_ids=(self.source.id,),
            operation="rumination_notice", reason="Selected private mental work.",
            subject_entity_id=self.partner.id)
        head = self.owner.admit_deliberate_action(admission, grant=grant, now_utc=self.now)
        activity = self.activity()
        accepted = self.owner.accept_step(head, self.noticing_step("rumination_notice"),
            grant=self.owner.execution_grant(head), commit_id="step-deliberate-notice", active_seconds=1.5,
            now_utc=self.now, mental_state=FullMentalState(), control_generation=0, activity=activity)
        assert accepted.receipt.operation_record_id is not None
        operation = self.store.snapshot(capture=False).get(accepted.receipt.operation_record_id)
        self.assertIsNone(operation.metadata.get(MENTAL_ACTION_KEY))
        view = self.store.snapshot(capture=False)
        head_now = deliberate_inquiry(view, action.id)
        assert head_now is not None
        self.assertIsNone(mental_outcome(view, head_now))


if __name__ == "__main__":
    unittest.main()
