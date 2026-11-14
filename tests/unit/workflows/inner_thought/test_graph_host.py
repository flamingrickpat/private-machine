"""Exercise graph-host acceptance with real stores and durable checkpoints.

Boundary proposals are explicit test inputs. These checks do not substitute
for the separate generated workflow gate.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from pydantic import ValidationError

from pm.memory.snapshot import GhostCancelled, GhostConflict
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.enums import EntityClass, FeatureType, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Entity, Feature
from pm.records.mental_state import FullMentalState
from workflows.cognitive_control.schemas import ControlConfig
from workflows.inner_thought.graph_execution import (
    graph_thought_instruction, graph_thought_request, validate_graph_step,
)
from workflows.inner_thought.pipeline import ModelProposalError
from workflows.inner_thought.graph_host import (
    FrozenProjectionPayload, PrivateInquiryOwner, active_inquiry, attach_inner_projection, pending_findings,
)
from workflows.inner_thought.inquiry import InquiryRecord, inquiry_history, stage_inquiry_step
from workflows.inner_thought.nominations import capture_inquiry_grant, stage_nominations
from workflows.inner_thought.schemas import (
    InnerRequest, InquiryAlternative, InquiryProgress, InquiryStep, ThoughtAttentionDecision,
    ThoughtJudgments, ThoughtNomination, ThoughtRelation,
)


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

    def test_forced_stall_bound_does_not_publish_generated_possibilities_as_finding(self) -> None:
        self.owner.limits = ControlConfig(stalled_steps=1)
        generated = self.advance(self.begin(), self.generation(), "generation-at-bound")
        self.assertEqual(generated.receipt.state.status, "paused")
        self.assertTrue(generated.receipt.state.pending_generation_ids)
        self.assertEqual(pending_findings(self.store), ())


if __name__ == "__main__":
    unittest.main()
