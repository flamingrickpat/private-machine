"""Check durable reasons and the fixed vector positions of game-choice affect."""
from tempfile import TemporaryDirectory
from pathlib import Path
import json
import unittest

from pm.memory.store import MemoryStore
from pm.records.mental_state import FullMentalState
from tests.unit.test_design_contracts import small_ghost
from workflows.main_cognition.affect_dynamics import project_affect
from workflows.main_cognition.schemas import AppraisalResult, GoalChange
from workflows.main_cognition.stage_commits import apply_goal_changes


class GameChoiceTests(unittest.TestCase):
    def test_real_quiet_choice_is_accepted_without_starting_a_workflow(self):
        from workflows.main_cognition.pipeline import action_dispatches
        from workflows.main_cognition.schemas import ActionSelectionResult
        from workflows.main_cognition.stages.action_selection.steps import validate_actions
        original = json.loads(Path("test_data/captures/alpha/game-quiet-choice.json").read_text(encoding="utf-8"))["result"]
        result = ActionSelectionResult.model_validate(original)
        validate_actions(result, [{"name": "minecraft"}])
        self.assertEqual(action_dispatches(original["actions"]), [])
        self.assertEqual(result.actions[0].content, original["actions"][0]["content"])

    def test_social_appraisal_changes_relationship_but_world_events_retain_it(self):
        social = AppraisalResult(feeling="I distrust that treatment.", appraisal={
            "norm_violation": 0.9, "contempt": 0.8, "fairness": 0.0, "agency_other": 1.0},
            emotion_delta={"anger": 0.6})
        accepted = project_affect(None, social, relationship_entity_id=3)
        self.assertEqual(accepted.state.state_relationship.entity_id, 3)
        self.assertLess(accepted.state.state_relationship.trust, 0)
        self.assertGreater(accepted.state.state_relationship.conflict_tension, 0)
        world = project_affect(accepted.state.to_list(), social)
        self.assertEqual(world.state.state_relationship.model_dump(), accepted.state.state_relationship.model_dump())
        self.assertEqual(world.trace["relationship_change"], {})

    def test_negative_affect_round_trips_without_changing_vector_size(self):
        baseline = FullMentalState().to_list()
        appraisal = AppraisalResult(feeling="I want different work.", emotion_delta={
            "anger": 0.3, "resentment": 0.4, "jealousy": 0.5, "boredom": 0.6})
        projection = project_affect(baseline, appraisal)
        accepted = FullMentalState.init_from_list(projection.state.to_list())
        self.assertEqual(len(accepted.to_list()), len(baseline))
        for name, value in appraisal.emotion_delta.items():
            self.assertAlmostEqual(getattr(accepted.state_emotions, name), value)
        self.assertEqual(accepted.to_list()[406:409], [0.4, 0.5, 0.6])

    def test_refusal_replaces_current_reason_and_retains_the_original_choice(self):
        ghost = small_ghost()
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "checkpoint.json"
        ghost.checkpoint = path
        first = ghost.begin_batch("choose-house")
        added = apply_goal_changes(first, first.view(), [GoalChange(change="add", goal="build house",
                    reason="shared project", priority=0.8)], evidence_ids=[], appraisal_id=10)
        ghost.commit(first)
        second = ghost.begin_batch("refuse-house")
        apply_goal_changes(second, second.view(), [GoalChange(change="refuse", goal="build house",
                    reason="resentment about dismissed work", priority=0.8)], evidence_ids=[], appraisal_id=11)
        ghost.commit(second)
        retained = MemoryStore.from_checkpoint(path).get_knoxel_by_id(added[0]["intention_id"])
        self.assertEqual(retained.status, "refused")
        self.assertFalse(retained.active)
        self.assertEqual(retained.metadata["appraisal_id"], 11)
        self.assertEqual(retained.metadata["reason"], "resentment about dismissed work")
        self.assertEqual([choice["appraisal_id"] for choice in retained.metadata["choice_history"]], [10, 11])
