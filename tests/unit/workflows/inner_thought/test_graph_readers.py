"""The private loop reads its own inquiry-graph thinking.

No model. These checks use the real inquiry graph, the real ``MemoryStore``
commit boundary, and the real ``PrivateThoughtLoop`` read path. A graph step
must reach the private loop's own readers with its operation, sources, outcome,
and progress, so the dwelling trigger and the prompt see what the loop actually
did. A linear thought keeps its legacy read.
"""
from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from pm.bus.bus import BusSystem
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.store import MemoryStore
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.enums import EntityClass, FeatureType, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Entity, Feature
from pm.records.mental_state import FullMentalState
from pm.runs.workflow_module import PROJECT_ROOT

from tests.support import ManualClock, refuse_private_inquiry
from workflows.cognitive_control.schemas import ControlConfig
from workflows.inner_thought.graph_host import PrivateInquiryOwner
from workflows.inner_thought.host import PrivateThoughtLoop, build_inner_thought_tools
from workflows.inner_thought.inquiry import InquiryRecord
from workflows.inner_thought.nominations import capture_inquiry_grant, stage_nominations
from workflows.inner_thought.schemas import (
    InnerRequest, InquiryAlternative, InquiryProgress, InquiryStep, ThoughtAttentionDecision,
    ThoughtJudgments, ThoughtNomination, ThoughtRelation,
)
from workflows.inner_thought.triggers import evaluate_triggers

MANIFEST = PROJECT_ROOT / "workflows" / "inner_thought" / "inner_thought.workflow.md"


class GraphReaderTests(unittest.TestCase):
    """Read executed graph steps through the private loop's own contracts."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.runtime_root = Path(directory.name) / "runtime"
        self.store: MemoryStore = MemoryStore(
            PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
            checkpoint=Path(directory.name) / "graph-readers.pickle")
        batch = self.store.begin_batch("anchors", embed=False)
        self.observer: Entity = batch.create(
            Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.partner: Entity = batch.create(
            Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.source: Feature = batch.create(Feature(content="I left the sketch unfinished.",
            feature_type=FeatureType.Dialogue, entity_id=self.partner.id,
            provenance=Provenance(source_kind=SourceKind.User)))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        # The stall bound is raised so three stalled steps can be recorded as
        # accepted history; the shipped bound would pause the inquiry at two.
        self.owner = PrivateInquiryOwner(self.store, commit=self._commit, cancel=Event(),
            limits=ControlConfig(stalled_steps=8, episode_steps=8))
        self.now: datetime = datetime(2026, 11, 12, 17, 0)
        self.judgments = ThoughtJudgments(interestingness=0.8, character_relevance=0.7,
            plausibility=0.3, consequence=0.4, desirability=0.1, action_suitability=0.2,
            reason="The sketch matters.")
        self.grant = capture_inquiry_grant(self.store.snapshot(capture=False),
            observer_entity_id=self.observer.id, scope=RecordScope(),
            source_ids=[self.source.id], governing_ids=[self.source.id], critical_epoch=0)

    def _commit(self, batch) -> None:
        self.store.commit(batch)

    def _branches(self) -> tuple[InquiryAlternative, ...]:
        return tuple(InquiryAlternative(text=text, evidence_ids=(self.source.id,),
            assumptions=(assumption,), reality="imagined", judgments=self.judgments,
            disposition="open", reason="A possible cause, not observation.")
            for text, assumption in (("Jesse revisits the sketch.", "Interest remains."),
                                     ("Jesse chooses another subject.", "A new interest matters more.")))

    def _step(self, thought: str, *, operation: str = "future_projection",
              next_operation: str = "feasibility_check",
              progress: InquiryProgress | None = None,
              relations: tuple[ThoughtRelation, ...] = (),
              alternatives: tuple[InquiryAlternative, ...] | None = None) -> InquiryStep:
        branches = self._branches() if alternatives is None else alternatives
        return InquiryStep(operation=operation, thought=thought, evidence_ids=(self.source.id,),
            assumptions=(), reality="inferred", judgments=self.judgments, relations=relations,
            alternatives=branches,
            progress=progress or InquiryProgress(kind="none", inspected_source_ids=(),
                compared_node_ids=(), meaningful_difference=False,
                reason="Possibilities provide no new outside evidence."),
            outcome="continue", reason="Compare the assumptions before concluding.",
            information_gap="Unknown motive.", next_operation=next_operation)

    def _stalled_progress(self) -> InquiryProgress:
        return InquiryProgress(kind="none", inspected_source_ids=(), compared_node_ids=(),
            meaningful_difference=False, reason="This step repeats earlier work.")

    def _accepted_progress(self, compared: tuple[int, ...]) -> InquiryProgress:
        return InquiryProgress(kind="discrimination", inspected_source_ids=(),
            compared_node_ids=compared, meaningful_difference=True,
            reason="Each possible cause predicts a different next choice.")

    def _select(self) -> InquiryRecord:
        batch = self.store.begin_batch("nomination", embed=False)
        nomination = ThoughtNomination(text="Jesse may return to the sketch or drop it.",
            question="Which explanation has evidence?", evidence_ids=(self.source.id,),
            reality="inferred", judgments=self.judgments, suggested_operation="future_projection")
        records = stage_nominations(batch, [nomination], grant=self.grant, batch_id="nomination")
        batch.preserve_on_recovery()
        self.store.commit(batch)
        selected = self.owner.accept_attention([records[0].record.id],
            ThoughtAttentionDecision(nomination_id=records[0].record.id, disposition="select",
                reason="This unresolved question holds private attention."),
            commit_id="attention", episode_id="private-sketch", now_utc=self.now)
        assert selected.inquiry is not None
        return selected.inquiry

    def _accept(self, parent: InquiryRecord, step: InquiryStep, commit_id: str) -> InquiryRecord:
        return self.owner.accept_step(parent, step, grant=self.owner.execution_grant(parent),
            commit_id=commit_id, active_seconds=1.5, now_utc=self.now,
            mental_state=FullMentalState(), control_generation=0)

    def _module(self) -> PrivateThoughtLoop:
        bus = BusSystem()
        errors: list[BaseException] = []

        def tools_factory(environment, batch):
            return build_inner_thought_tools(self.store, self.runtime_root, None,
                inquiry_provider=refuse_private_inquiry)(environment, batch)

        return PrivateThoughtLoop("inner_thought", bus, errors.append, ManualClock(), None,
            working_target=MemoryBashTarget(self.store, "read"), workflow=MANIFEST,
            tools_factory=tools_factory, runtime_root=self.runtime_root, ghost=self.store)

    def _commit_main_turn(self, stimulus_ids: list[int]) -> None:
        batch = self.store.begin_batch("main-1", new_tick=True)
        batch.set_gwt(
            GwtTurnRecord(tick_id=batch.tick_id, slot_id="main",
                stimulus_ids=list(stimulus_ids), knoxel_ids=list(stimulus_ids),
                summary="appraise the private finding"),
            CycleState(tick_id=batch.tick_id,
                stimulus_groups=[StimulusGroup(stimulus_ids=list(stimulus_ids))]))
        self.store.commit(batch)

    def test_a_graph_step_reaches_the_private_loop_with_its_operation_and_sources(self) -> None:
        selected = self._select()
        executed = self._accept(selected, self._step("Both explanations remain possible."), "generation")
        operation_id = executed.receipt.operation_record_id
        assert operation_id is not None
        thoughts = self._module()._recent_private_thoughts()
        found = next((item for item in thoughts if item.knoxel_id == operation_id), None)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertEqual(found.loop, "inquiry")
        self.assertEqual(found.operation, "future_projection")
        self.assertEqual(found.source_ids, [self.source.id])
        self.assertIsNotNone(found.inquiry)
        assert found.inquiry is not None
        self.assertEqual(found.inquiry.episode_id, "private-sketch")
        self.assertEqual(found.inquiry.outcome, "continue")
        self.assertEqual(found.inquiry.reason, "Compare the assumptions before concluding.")
        self.assertEqual(found.inquiry.information_gap, "Unknown motive.")
        self.assertEqual(found.inquiry.progress_kind, "none")

    def test_three_stalled_graph_steps_raise_the_repeated_concern_trigger(self) -> None:
        first = self._accept(self._select(), self._step("Both explanations remain possible."), "generation-1")
        operation_one = first.receipt.operation_record_id
        assert operation_one is not None
        branches = tuple(node_id for node_id in first.receipt.state.node_ids if node_id != operation_one)
        second = self._accept(first, self._step("Neither cause is excluded yet.",
            operation="feasibility_check", next_operation="future_projection",
            relations=tuple(ThoughtRelation(target_record_id=node_id, kind="merges",
                reason="Compare uncertain causes.") for node_id in branches),
            alternatives=()), "comparison-1")
        operation_two = second.receipt.operation_record_id
        assert operation_two is not None
        third = self._accept(second, self._step("The same two explanations still stand.",
            relations=tuple(ThoughtRelation(target_record_id=node_id, kind="merges",
                reason="Carry the unresolved graph forward.") for node_id in (operation_one, operation_two))),
            "generation-2")
        self.assertEqual(third.receipt.state.stalled_steps, 3)
        request = InnerRequest(recent_thoughts=self._module()._recent_private_thoughts())
        names = [candidate.name for candidate in evaluate_triggers(request)]
        self.assertIn("repeated_concern", names)

    def test_three_progressing_graph_steps_do_not_claim_repetition(self) -> None:
        first = self._accept(self._select(), self._step("Both explanations remain possible."), "generation-1")
        operation_one = first.receipt.operation_record_id
        assert operation_one is not None
        branches = tuple(node_id for node_id in first.receipt.state.node_ids if node_id != operation_one)
        second = self._accept(first, self._step("The two causes predict different choices.",
            operation="feasibility_check", progress=self._accepted_progress(branches),
            relations=tuple(ThoughtRelation(target_record_id=node_id, kind="merges",
                reason="Compare the two generated causes.") for node_id in branches),
            alternatives=()), "comparison-1")
        operation_two = second.receipt.operation_record_id
        assert operation_two is not None
        third = self._accept(second, self._step("The comparison narrows the remaining causes.",
            operation="feasibility_check", progress=self._accepted_progress((operation_one, operation_two)),
            relations=tuple(ThoughtRelation(target_record_id=node_id, kind="merges",
                reason="Compare the accepted operations.") for node_id in (operation_one, operation_two)),
            alternatives=()), "comparison-2")
        operation_three = third.receipt.operation_record_id
        assert operation_three is not None
        self._accept(third, self._step("The narrowed set still separates the causes.",
            operation="feasibility_check", progress=self._accepted_progress((operation_one, operation_three)),
            relations=tuple(ThoughtRelation(target_record_id=node_id, kind="merges",
                reason="Compare the accepted operations.") for node_id in (operation_one, operation_three)),
            alternatives=()), "comparison-3")
        request = InnerRequest(recent_thoughts=self._module()._recent_private_thoughts())
        names = [candidate.name for candidate in evaluate_triggers(request)]
        self.assertNotIn("repeated_concern", names)

    def test_a_main_frame_that_claims_only_a_graph_step_is_self_feedback(self) -> None:
        selected = self._select()
        executed = self._accept(selected, self._step("Both explanations remain possible."), "generation")
        operation_id = executed.receipt.operation_record_id
        assert operation_id is not None
        self._commit_main_turn([operation_id])
        payload = self._module().prepare_input([])
        self.assertTrue(payload["main_frame"]["self_feedback"])


if __name__ == "__main__":
    unittest.main()
