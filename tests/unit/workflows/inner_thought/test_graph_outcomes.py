"""Apply actual-store inquiry outcomes without mutating accepted graph records."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.records.base import Provenance, RecordScope
from pm.records.cognitive_records import ExpectationRecord
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import EntityClass, FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import Entity, Feature
from pm.records.lineage import RecordLineage
from workflows.inner_thought.expectations import open_expectation
from workflows.inner_thought.graph_outcomes import (
    attach_graph_outcome, capture_graph_outcome, read_graph_outcome, stage_graph_outcome,
)
from workflows.inner_thought.nominations import capture_inquiry_grant, extend_inquiry_grant
from workflows.inner_thought.schemas import (
    DisclosureChoice, ExpectationUpdate, InquiryProgress, InquiryStep, IntegrationResult, ThoughtJudgments,
)


class GraphOutcomeTests(unittest.TestCase):
    """Retained integration supplies original guards to a later GWT projection."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "outcome.pickle"
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=self.path)
        batch = self.store.begin_batch("evidence", embed=False)
        self.observer = batch.create(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.partner = batch.create(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        self.source = batch.create(Feature(content="Yes, let's finish the story tomorrow.",
            feature_type=FeatureType.Dialogue, causal=True, entity_id=self.partner.id,
            addressee_entity_id=self.observer.id, interlocus=InterlocusType.Public,
            provenance=Provenance(source_kind=SourceKind.User)))
        self.expectation = open_expectation(batch, anticipation="Jesse agrees to finish the story.",
            expected_outcome="Jesse says yes.", expected_valence=0.6, uncertainty=0.4,
            goal_ids=[], evidence_ids=[self.source.id])
        batch.preserve_on_recovery()
        self.store.commit(batch)
        self.grant = capture_inquiry_grant(self.store.snapshot(), observer_entity_id=self.observer.id,
            scope=RecordScope(), source_ids=[self.source.id, self.expectation.id],
            governing_ids=[self.expectation.id], critical_epoch=0)
        judgments = ThoughtJudgments(interestingness=0.5, character_relevance=0.8, plausibility=0.8,
            consequence=0.5, desirability=0.8, action_suitability=0.5, reason="The actual reply supports the anticipation.")
        self.step = InquiryStep(operation="expectation_review", thought="Jesse explicitly agreed.",
            evidence_ids=(self.source.id,), assumptions=(), reality="inferred", judgments=judgments,
            relations=(), alternatives=(), progress=InquiryProgress(kind="none", inspected_source_ids=(),
                compared_node_ids=(), meaningful_difference=False, reason="The supplied reply was already available."),
            outcome="finding", reason="The anticipation has a cited outcome.", information_gap="", next_operation="")

    def retain(self, integration: IntegrationResult) -> Feature:
        """Persist a semantic boundary input through the production store."""
        outcome = capture_graph_outcome(self.grant, self.step, integration)
        batch = self.store.begin_batch("accepted-graph", embed=False)
        operation = batch.create(Feature(content=self.step.thought, feature_type=FeatureType.InnerMonologue,
            entity_id=self.observer.id, provenance=Provenance(source_kind=SourceKind.Self),
            interlocus=InterlocusType.PrivateInternal))
        attach_graph_outcome(operation, self.step, outcome)
        batch.revise(operation)
        batch.preserve_on_recovery()
        self.store.commit(batch)
        return operation

    def resolution(self) -> IntegrationResult:
        """Propose comparison with the actual supplied utterance."""
        return IntegrationResult(summary="Jesse agreed; the original anticipation is supported.",
            expectation=ExpectationUpdate(action="resolve", expectation_id=self.expectation.id,
                observation_ids=[self.source.id], status="supported", reason="The reply explicitly agrees."),
            disclosure=DisclosureChoice(mode="candid", shareable_summary="I am glad we will finish it together."))

    def test_resolution_repair_uses_persisted_integration_without_mutating_graph(self) -> None:
        operation = self.retain(self.resolution())
        reloaded = MemoryStore.from_checkpoint(self.path)
        view = reloaded.snapshot()
        retained = view.get(operation.id)
        assert isinstance(retained, Feature)
        before = retained.model_dump_json()
        outcome = read_graph_outcome(retained)
        assert outcome is not None
        batch = reloaded.begin_batch("derived-turn", embed=False)
        stage_graph_outcome(batch, view, retained, self.step, outcome)
        reloaded.commit(batch)
        accepted = reloaded.snapshot()
        current = next(record for record in accepted.all_knoxels.values() if isinstance(record, ExpectationRecord))
        self.assertEqual(current.status, "supported")
        self.assertEqual(current.observation_ids, [self.source.id])
        self.assertEqual(accepted.get(operation.id).model_dump_json(), before)
        original_expectation = accepted.get(self.expectation.id)
        assert isinstance(original_expectation, ExpectationRecord)
        self.assertEqual(original_expectation.status, "pending")

    def test_changed_original_target_rejects_projection_without_refresh(self) -> None:
        operation = self.retain(self.resolution())
        original = self.store.snapshot().get(self.expectation.id)
        assert isinstance(original, ExpectationRecord)
        changed = original.copy_for_edit()
        assert isinstance(changed, ExpectationRecord)
        changed.expected_outcome = "A different outcome, changed after the model read."
        correction = self.store.begin_batch("later-correction", embed=False)
        correction.replace(changed)
        self.store.commit(correction)
        view = self.store.snapshot()
        outcome = read_graph_outcome(operation)
        assert outcome is not None
        batch = self.store.begin_batch("stale-projection", embed=False)
        stage_graph_outcome(batch, view, operation, self.step, outcome)
        with self.assertRaises(GhostConflict):
            self.store.commit(batch)
        self.assertEqual(self.store.snapshot().get(operation.id).metadata["graph_outcome"], outcome.model_dump(mode="json"))

    def test_opening_outcome_preserves_scope_and_proposes_goals_without_changing_them(self) -> None:
        operation = self.retain(IntegrationResult(summary="A new anticipation remains uncertain.",
            expectation=ExpectationUpdate(action="open", anticipation="The story session feels collaborative.",
                expected_outcome="A shared session", uncertainty=0.6)))
        outcome = read_graph_outcome(operation)
        assert outcome is not None
        batch = self.store.begin_batch("open-at-projection", embed=False)
        stage_graph_outcome(batch, self.store.snapshot(), operation, self.step, outcome)
        self.store.commit(batch)
        new = [record for record in self.store.all_knoxels.values()
               if isinstance(record, ExpectationRecord) and record.id != self.expectation.id]
        self.assertEqual(len(new), 1)
        self.assertEqual(new[0].scope, operation.scope)
        self.assertEqual(new[0].evidence_ids, [self.source.id])
        self.assertEqual(new[0].status, "pending")

    def test_capture_rejects_outside_observations_and_historical_resolution_authority(self) -> None:
        integration = self.resolution()
        integration.expectation.observation_ids = [999]
        with self.assertRaisesRegex(ValueError, "outside its original grant"):
            capture_graph_outcome(self.grant, self.step, integration)
        historical = self.grant.model_copy(update={"sources": tuple(source.model_copy(update={"current_head": False})
            for source in self.grant.sources)})
        with self.assertRaisesRegex(ValueError, "captured current expectation authority"):
            capture_graph_outcome(historical, self.step, self.resolution())

    def test_generated_content_cannot_be_an_observed_expectation_outcome(self) -> None:
        imagined = self.store.begin_batch("imagined-option", embed=False)
        thought = imagined.create(Feature(content="Perhaps Jesse will enjoy the session.",
            feature_type=FeatureType.Thought, entity_id=self.observer.id,
            provenance=Provenance(source_kind=SourceKind.Self), interlocus=InterlocusType.PrivateInternal,
            lineage=RecordLineage(episode_id="earlier", operation="future_projection",
                source_event_ids=[self.source.id], reality="imagined")))
        imagined.preserve_on_recovery()
        self.store.commit(imagined)
        self.grant = extend_inquiry_grant(self.store.snapshot(), self.grant, [thought.id])
        integration = self.resolution()
        integration.expectation.observation_ids = [thought.id]
        operation = self.retain(integration)
        outcome = read_graph_outcome(operation)
        assert outcome is not None
        batch = self.store.begin_batch("invalid-imagined-resolution", embed=False)
        with self.assertRaisesRegex(ValueError, "Generated or imagined content"):
            stage_graph_outcome(batch, self.store.snapshot(), operation, self.step, outcome)


if __name__ == "__main__":
    unittest.main()
