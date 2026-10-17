"""Exercise real checkpoint commits, stale owners, and experiment recovery."""
from datetime import datetime, timedelta
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from uuid import uuid4

from pm.memory.snapshot import GhostCancelled, GhostConflict
from pm.memory.store import MemoryStore
from pm.records.cognitive_records import BeliefRecord
from pm.records.enums import FeatureType, SourceKind
from pm.records.memory_records import new_feature
from tests.unit.test_design_contracts import small_ghost
from workflows.cognitive_control.attention import active_focus, stage_focus
from workflows.reflection.experiments import context, effective_inputs, stage_decision
from workflows.reflection.experiments import validate_decision
from workflows.reflection.schemas import ExperimentPlan, Intervention, ReflectionDecision


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.store = small_ghost()
        self.source = new_feature(self.store, "Please leave the ending open.", FeatureType.Dialogue,
                                  source="Jesse", source_kind=SourceKind.User)
        self.episode = new_feature(self.store, "I chose a closed ending.", FeatureType.StateUpdate,
                                   source="Spamuel", metadata={"operation": "action_selection"})
        self.anchor = new_feature(self.store, "The choice missed the preference.", FeatureType.InnerMonologue,
                                  source="Spamuel", metadata={"loop": "inner", "evidence_ids": [self.episode.id]})
        self.plan = ExperimentPlan(triggering_episode_id=self.anchor.id,
            finding="The choice missed a stated preference.", diagnosis="missing_evidence",
            hypotheses=[{"explanation": "The original preference was absent from context.",
                         "supporting_ids": [self.source.id, self.episode.id],
                         "uncertainty": "A different preference can supersede it."},
                        {"explanation": "The closed ending was a deliberate creative choice.",
                         "opposing_ids": [self.source.id], "uncertainty": "The recorded choice does not explain why."}],
            intervention={"owner": "attention", "focus": {"instruction": "Consider the original ending preference.",
                "promote_ids": [self.source.id], "strength": 0.3, "seconds": 900}},
            predicted_effect="The next relevant choice keeps the ending open.",
            discriminating_observation="Inspect the next selected ending.",
            observation_event="The next story choice.", scope="This shared story.")

    def step(self, action, experiment_id=None, *, cancel=None, **kwargs):
        view = self.store.snapshot()
        batch = self.store.begin_batch(uuid4().hex, new_tick=True, cancel=cancel)
        record = stage_decision(batch, view, ReflectionDecision(action=action,
            experiment_id=experiment_id, reason="Inspect the recorded consequence.", **kwargs),
            context(view), companion_name="Spamuel")
        self.store.commit(batch)
        return self.store.get_knoxel_by_id(record.id) if record is not None else None

    def proposed(self):
        return self.step("propose", plan=self.plan)

    def applied(self):
        proposal = self.proposed()
        return self.step("apply", proposal.id)

    def observation(self, experiment, text="I chose the open ending."):
        batch = self.store.begin_batch(uuid4().hex, new_tick=True)
        record = new_feature(self.store, text, FeatureType.StateUpdate, source="Spamuel", changes=batch,
            metadata={"operation": "action_selection", "reflection_versions": {
                str(experiment.id): experiment.metadata["effective_record_id"]}})
        self.store.commit(batch)
        return record

    def test_prediction_commits_before_application_and_changes_no_advice(self):
        proposal = self.proposed()
        self.assertEqual(proposal.metadata["state"], "proposed")
        self.assertEqual(active_focus(self.store), {})
        applied = self.step("apply", proposal.id)
        self.assertEqual(active_focus(self.store)["control_record_id"], applied.metadata["effective_record_id"])
        self.assertLess(proposal.tick_id, applied.metadata["applied_tick"])
        self.assertEqual(applied.metadata["history"][0]["action"], "proposed")

    def test_stale_source_cannot_apply(self):
        proposal = self.proposed()
        batch = self.store.begin_batch("source-correction")
        batch.replace(self.source.model_copy(update={"content": "The user corrected the preference."}))
        self.store.commit(batch)
        with self.assertRaises(GhostConflict):
            self.step("apply", proposal.id)
        self.assertEqual(active_focus(self.store), {})
        self.assertEqual(self.store.get_knoxel_by_id(proposal.id).metadata["state"], "proposed")

    def test_concurrent_first_advice_rejects_empty_family_guard(self):
        view = self.store.snapshot()
        first = self.store.begin_batch("first")
        second = self.store.begin_batch("second")
        for batch in (first, second):
            stage_focus(batch, view, instruction="Consider the story.", focus={"accept": True},
                        companion_name="Spamuel", expires_at=(datetime.utcnow() + timedelta(seconds=30)).isoformat())
        self.store.commit(first)
        with self.assertRaises(GhostConflict):
            self.store.commit(second)

    def test_new_advice_blocks_application_and_reversal(self):
        proposal = self.proposed()
        view = self.store.snapshot()
        batch = self.store.begin_batch("new-instruction")
        stage_focus(batch, view, instruction="Consider the newer instruction.", focus={"accept": True},
                    companion_name="Spamuel", expires_at=(datetime.utcnow() + timedelta(seconds=30)).isoformat())
        self.store.commit(batch)
        with self.assertRaises(GhostConflict):
            self.step("apply", proposal.id)
        self.plan.intervention.expected_record_id = max(active_focus(self.store)["control_record_id"], 0)
        applied = self.applied()
        batch = self.store.begin_batch("still-newer")
        stage_focus(batch, self.store.snapshot(), instruction="An urgent new focus.", focus={"accept": True},
                    companion_name="Spamuel", expires_at=(datetime.utcnow() + timedelta(seconds=30)).isoformat())
        self.store.commit(batch)
        with self.assertRaises(GhostConflict):
            self.step("reverse", applied.id)

    def test_cancelled_application_preserves_proposal_and_checkpoint(self):
        with TemporaryDirectory() as directory:
            self.store.checkpoint = Path(directory) / "ghost.json"
            proposal = self.proposed()
            before = self.store.checkpoint.read_bytes()
            view = self.store.snapshot()
            cancel = Event()
            batch = self.store.begin_batch("cancel-at-commit", new_tick=True, cancel=cancel)
            stage_decision(batch, view, ReflectionDecision(action="apply", experiment_id=proposal.id,
                reason="Try the prediction."), context(view), companion_name="Spamuel")
            cancel.set()
            with self.assertRaises(GhostCancelled):
                self.store.commit(batch)
            self.assertEqual(self.store.checkpoint.read_bytes(), before)
            reopened = MemoryStore.from_checkpoint(self.store.checkpoint)
            self.assertEqual(reopened.get_knoxel_by_id(proposal.id).metadata["state"], "proposed")
            self.assertEqual(active_focus(reopened), {})

    def test_no_observation_remains_unresolved(self):
        applied = self.applied()
        evaluated = self.step("evaluate", applied.id, result="unresolved")
        self.assertEqual(evaluated.metadata["result"], "unresolved")
        with self.assertRaises(ValueError):
            self.step("retain", applied.id)
        with self.assertRaises(ValueError):
            self.step("evaluate", applied.id, result="supported")

    def test_generated_worry_and_preapplication_evidence_cannot_prove_an_outcome(self):
        applied = self.applied()
        self.observation(applied)
        thought = new_feature(self.store, "Perhaps this worked.", FeatureType.InnerMonologue, source="Spamuel")
        for key in (thought.id, self.source.id):
            view = self.store.snapshot()
            payload = context(view)
            from workflows.reflection.experiments import source_payload
            payload["records"].append(source_payload(view, view.get(key)))
            with self.assertRaises(ValueError):
                stage_decision(self.store.begin_batch(uuid4().hex), view,
                    ReflectionDecision(action="evaluate", experiment_id=applied.id,
                        result="supported", observation_ids=[key], reason="A proposed reading."),
                    payload, companion_name="Spamuel")

    def test_two_later_choices_can_retain_scoped_strategy(self):
        applied = self.applied()
        first = self.observation(applied)
        self.step("evaluate", applied.id, result="supported", observation_ids=[first.id])
        with self.assertRaises(ValueError):
            self.step("retain", applied.id)
        second = self.observation(applied, "The next chapter also leaves an open ending.")
        self.step("evaluate", applied.id, result="supported", observation_ids=[second.id])
        retained = self.step("retain", applied.id)
        self.assertEqual(retained.metadata["plan"]["scope"], "This shared story.")
        self.assertEqual(retained.metadata["state"], "retained")

    def test_failed_prediction_and_reversal_survive_reopen(self):
        with TemporaryDirectory() as directory:
            self.store.checkpoint = Path(directory) / "ghost.json"
            applied = self.applied()
            observed = self.observation(applied, "I still chose a closed ending.")
            self.step("evaluate", applied.id, result="contradicted", observation_ids=[observed.id])
            reversed_record = self.step("reverse", applied.id)
            reopened = MemoryStore.from_checkpoint(self.store.checkpoint)
            result = reopened.get_knoxel_by_id(applied.id)
            self.assertEqual(result.metadata, reversed_record.metadata)
            self.assertEqual(result.metadata["result"], "contradicted")
            self.assertEqual(result.metadata["history"][-2]["action"], "evaluated")
            self.assertEqual(active_focus(reopened), {})
            self.assertGreater(result.metadata["reversal_record_id"], result.metadata["effective_record_id"])

    def test_belief_revision_keeps_original_report_and_reversal_is_a_new_belief(self):
        belief = self.store.store_knoxel(BeliefRecord(content="Jesse does not care about endings.",
            belief_key="relationship:story", source="self:Spamuel", supporting_knoxel_ids=[self.episode.id]))
        self.plan.intervention = Intervention(owner="belief", expected_record_id=belief.id,
                                             belief="Jesse cares about an open ending in our story.")
        applied = self.applied()
        effective = self.store.get_knoxel_by_id(applied.metadata["effective_record_id"])
        self.assertFalse(self.store.get_knoxel_by_id(belief.id).active)
        self.assertIn(self.source.id, effective.supporting_knoxel_ids)
        inputs = effective_inputs(self.store.snapshot(), {})
        self.assertEqual(inputs["versions"][str(applied.id)], effective.id)
        self.assertEqual(self.store.get_knoxel_by_id(self.source.id).content, self.source.content)
        reversed_record = self.step("reverse", applied.id)
        restored = self.store.get_knoxel_by_id(reversed_record.metadata["reversal_record_id"])
        self.assertEqual(restored.content, belief.content)
        self.assertNotEqual(restored.id, belief.id)

    def test_cannot_rewrite_a_user_report_as_a_belief(self):
        belief = self.store.store_knoxel(BeliefRecord(content="Jesse wants an open ending.",
            belief_key="ending", belief_kind="known", source="user_message"))
        self.plan.intervention = Intervention(owner="belief", expected_record_id=belief.id, belief="Jesse wants the opposite.")
        with self.assertRaises(ValueError):
            self.proposed()

    def test_interruption_retains_known_applied_state(self):
        applied = self.applied()
        interrupted = self.step("interrupt", applied.id)
        self.assertEqual(interrupted.metadata["state"], "interrupted")
        self.assertEqual(interrupted.metadata["effective_record_id"], active_focus(self.store)["control_record_id"])

    def test_unrelated_later_user_message_cannot_establish_an_outcome(self):
        applied = self.applied()
        self.observation(applied)
        report = new_feature(self.store, "The unrelated timer worked.", FeatureType.Dialogue,
                             source="Jesse", source_kind=SourceKind.User)
        with self.assertRaisesRegex(ValueError, "explicit link"):
            self.step("evaluate", applied.id, result="supported", observation_ids=[report.id])

    def test_reported_user_belief_stays_immutable(self):
        belief = self.store.store_knoxel(BeliefRecord(content="I can choose later.",
            belief_key="user-claim", belief_kind="reported_by_user", source="user:Jesse"))
        self.plan.intervention = Intervention(owner="belief", expected_record_id=belief.id, belief="The opposite claim.")
        with self.assertRaisesRegex(ValueError, "subjective"):
            self.proposed()

    def test_reversal_version_is_visible_to_later_main_cognition(self):
        belief = self.store.store_knoxel(BeliefRecord(content="The old interpretation.", belief_key="old", source="self:Spamuel"))
        self.plan.intervention = Intervention(owner="belief", expected_record_id=belief.id, belief="A revised interpretation.")
        applied = self.applied()
        restored = self.step("reverse", applied.id)
        inputs = effective_inputs(self.store.snapshot(), {})
        self.assertEqual(inputs["versions"][str(applied.id)], restored.metadata["reversal_record_id"])

    def test_hypothesis_revision_keeps_failed_prediction_and_new_proposal_separate(self):
        applied = self.applied()
        observed = self.observation(applied, "The correction did not help.")
        self.step("evaluate", applied.id, result="contradicted", observation_ids=[observed.id])
        restored = self.step("reverse", applied.id)
        corrected = self.plan.model_copy(deep=True)
        corrected.hypotheses[0].explanation = "The proposal needs a different focus."
        corrected.hypotheses[0].supporting_ids.append(observed.id)
        corrected.intervention.expected_record_id = restored.metadata["reversal_record_id"]
        revised = self.step("revise", applied.id, plan=corrected)
        replacement = self.store.get_knoxel_by_id(revised.metadata["replacement_experiment_id"])
        self.assertEqual(revised.metadata["result"], "contradicted")
        self.assertEqual(revised.metadata["plan"]["predicted_effect"], self.plan.predicted_effect)
        self.assertEqual(replacement.metadata["state"], "proposed")
        self.assertEqual(replacement.metadata["revised_from_experiment_id"], applied.id)
        self.assertIsNone(replacement.metadata["effective_record_id"])


class CapturedOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.capture = json.loads(Path("test_data/captures/alpha/reflection-outcome-rejection.json").read_text(encoding="utf-8"))

    def test_captured_invalid_evaluation_is_rejected_before_host_commit(self):
        with self.assertRaisesRegex(ValueError, "Allowed observation IDs"):
            validate_decision(ReflectionDecision.model_validate(self.capture["proposal"]), self.capture["payload"])

    def test_consuming_decisions_do_not_force_a_successful_result(self):
        decision = ReflectionDecision.model_validate({**self.capture["proposal"], "observation_ids": [11226, 11247]})
        validate_decision(decision, self.capture["payload"])
        self.assertEqual(decision.result, "unresolved")
