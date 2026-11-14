"""Check the focus-advice boundary of deliberate mental work without a model.

These are in-process checks of deterministic validation and prompt assembly,
plus one real store for source capture. The proposals are explicit inputs. No
check here shows that a model writes useful advice; the recorded
``deliberate-entry`` scenario exercises the real integrate stage.
"""
from __future__ import annotations

import unittest

from pm.memory.store import MemoryStore
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import EntityClass, FeatureType, SourceKind
from pm.records.memory_records import Entity, Feature
from workflows.cognitive_control.schemas import FocusProposal
from workflows.inner_thought.graph_execution import with_origin_evidence
from workflows.inner_thought.graph_outcomes import GraphMentalProposal, capture_graph_outcome
from workflows.inner_thought.nominations import capture_inquiry_grant
from workflows.inner_thought.pipeline import IntegrationError, validate_integration
from workflows.inner_thought.schemas import (
    FocusAdviceRequest, InnerRequest, InquiryGrant, InquiryOrigin, InquiryProgress, InquiryStep, IntegrationResult,
    ThoughtCandidate, ThoughtJudgments,
)
from workflows.inner_thought.stages.integrate.steps import build_messages, load_prompts


class FocusAdviceTests(unittest.TestCase):
    """Only deliberate work is asked for focus advice, and only over granted records."""

    def setUp(self) -> None:
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        batch = self.store.begin_batch("anchors", embed=False)
        self.observer: Entity = batch.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.source: Feature = batch.create(Feature(content="I stopped talking about the story.",
            feature_type=FeatureType.Dialogue, provenance=Provenance(source_kind=SourceKind.User)))
        self.other: Feature = batch.create(Feature(content="The tap is dripping again.",
            feature_type=FeatureType.Dialogue, provenance=Provenance(source_kind=SourceKind.User)))
        batch.preserve_on_recovery()
        self.store.commit(batch)
        self.grant: InquiryGrant = capture_inquiry_grant(self.store.snapshot(capture=False),
            observer_entity_id=self.observer.id, scope=RecordScope(), source_ids=[self.source.id], critical_epoch=0)
        judgments = ThoughtJudgments(interestingness=0.5, character_relevance=0.7, plausibility=0.2,
            consequence=0.4, desirability=0.5, action_suitability=0.3, reason="A withdrawn topic needs explaining.")
        self.step: InquiryStep = InquiryStep(operation="feasibility_check", thought="The report names no motive.",
            evidence_ids=(self.source.id,), assumptions=(), reality="inferred", judgments=judgments, relations=(),
            alternatives=(), progress=InquiryProgress(kind="none", inspected_source_ids=(), compared_node_ids=(),
                meaningful_difference=False, reason="Nothing outside was acquired."),
            outcome="information_gap", reason="The motive is unknown.", information_gap="Jesse's own reason.",
            next_operation="")

    def advice(self, *, promote_ids: list[int]) -> FocusProposal:
        """State one focus proposal over the named records."""
        return FocusProposal(instruction="Keep the unfinished story in view.", promote_ids=promote_ids,
                             strength=0.3, seconds=600, reason="Jesse dropped the topic without explaining.")

    def test_unrequested_focus_advice_is_rejected(self) -> None:
        integration = IntegrationResult(focus=self.advice(promote_ids=[self.source.id]))
        with self.assertRaisesRegex(IntegrationError, "not requested"):
            validate_integration(integration, {self.source.id}, InnerRequest())
        validate_integration(integration, {self.source.id}, InnerRequest(), focus_requested=True)
        validate_integration(IntegrationResult(), {self.source.id}, InnerRequest())

    def test_requested_advice_cannot_name_records_outside_the_frozen_view(self) -> None:
        integration = IntegrationResult(focus=self.advice(promote_ids=[self.source.id, self.other.id]))
        with self.assertRaisesRegex(IntegrationError, f"outside the frozen view: \\[{self.other.id}\\]"):
            validate_integration(integration, {self.source.id}, InnerRequest(), focus_requested=True)

    def test_outcome_keeps_the_advised_records_as_original_sources(self) -> None:
        integration = IntegrationResult(summary="The reason is still unknown.",
                                        focus=self.advice(promote_ids=[self.source.id]))
        outcome = capture_graph_outcome(self.grant, self.step, integration)
        self.assertIn(self.source.id, {source.record_id for source in outcome.sources})
        self.assertEqual(outcome.integration.focus, integration.focus)
        outside = IntegrationResult(focus=self.advice(promote_ids=[self.other.id]))
        with self.assertRaisesRegex(ValueError, "outside its original grant"):
            capture_graph_outcome(self.grant, self.step, outside)

    def test_proposal_carries_the_advice_for_main_attention(self) -> None:
        proposal = GraphMentalProposal(instruction="Notice the dropped topic.", summary="Unknown reason.",
            evidence_ids=(self.source.id,), goal_changes=(), focus=self.advice(promote_ids=[self.source.id]))
        assert proposal.focus is not None
        self.assertEqual(proposal.focus.promote_ids, [self.source.id])
        self.assertIsNone(GraphMentalProposal(instruction="Notice.", summary="Unknown.",
                                              evidence_ids=(), goal_changes=()).focus)

    def test_only_a_deliberate_integration_prompt_offers_the_focus_field(self) -> None:
        prompts = load_prompts()
        request = InnerRequest(companion_name="Spamuel").model_dump(mode="json")
        spontaneous = build_messages(dict(request), prompts)
        spontaneous_text = "\n".join(str(turn.content) for turn in spontaneous)
        self.assertNotIn("promote_ids", spontaneous_text)
        self.assertNotIn("deliberate mental work", spontaneous_text)
        deliberate_request = dict(request)
        deliberate_request["focus_advice"] = FocusAdviceRequest(instruction="Compare the two endings.",
            citable_ids=(self.source.id,)).model_dump(mode="json")
        deliberate = build_messages(deliberate_request, prompts)
        # The system turn and the start of the user turn stay identical.
        self.assertEqual(str(deliberate[0].content), str(spontaneous[0].content))
        deliberate_user = str(deliberate[1].content)
        self.assertTrue(deliberate_user.startswith(str(spontaneous[1].content)))
        added = deliberate_user[len(str(spontaneous[1].content)):]
        self.assertIn("Compare the two endings.", added)
        self.assertIn(f"may only use these IDs: [{self.source.id}]", added)
        self.assertIn('"promote_ids"', added)

    def test_deliberate_step_reads_the_text_of_the_sources_its_action_named(self) -> None:
        view = self.store.snapshot(capture=False)
        origin = InquiryOrigin(record_id=self.observer.id, kind="deliberate_action",
            text="Compare the two endings.", question="Compare the two endings.",
            evidence_ids=(self.source.id, self.other.id), judgments=None, suggested_operation="")
        request = InnerRequest(candidates=[ThoughtCandidate(knoxel_id=self.other.id, text="already supplied")])
        extended = with_origin_evidence(view, request, origin)
        by_id = {candidate.knoxel_id: candidate for candidate in extended.candidates}
        self.assertEqual(by_id[self.source.id].text, "I stopped talking about the story.")
        self.assertEqual(by_id[self.source.id].kind, "deliberate_evidence")
        self.assertEqual(by_id[self.source.id].revision, view.revisions[self.source.id])
        # A source the request already supplies is not replaced or duplicated.
        self.assertEqual(by_id[self.other.id].text, "already supplied")
        self.assertEqual(len(extended.candidates), 2)
        self.assertEqual(len(request.candidates), 1)


if __name__ == "__main__":
    unittest.main()
