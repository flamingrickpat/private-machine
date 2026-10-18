"""Check durable reasons and the fixed vector positions of game-choice affect."""
from tempfile import TemporaryDirectory
from pathlib import Path
import json
import unittest

from pm.memory.store import MemoryStore
from pm.records.mental_state import FullMentalState
from tests.unit.test_design_contracts import small_ghost
from workflows.main_cognition.affect import appraise_state
from workflows.main_cognition.schemas import AppraisalResult, GoalChange
from workflows.main_cognition.stage_commits import apply_goal_changes


class GameChoiceTests(unittest.TestCase):
    def test_real_quiet_choice_is_accepted_without_starting_a_workflow(self):
        from workflows.main_cognition.phases.decide import validate_actions
        from workflows.main_cognition.phases.route import action_dispatches
        from workflows.main_cognition.schemas import ActionSelectionResult, WorkflowOption
        original = json.loads(Path("test_data/captures/alpha/game-quiet-choice.json").read_text(encoding="utf-8"))["result"]
        result = ActionSelectionResult.model_validate(original)
        validate_actions(result, [WorkflowOption(name="minecraft")])
        self.assertEqual(action_dispatches(result.actions), [])
        self.assertEqual(result.actions[0].content, original["actions"][0]["content"])

    def test_negative_affect_round_trips_without_changing_vector_size(self):
        baseline = FullMentalState()
        appraisal = AppraisalResult(feeling="I want different work.", emotion_delta={
            "anger": 0.3, "resentment": 0.4, "jealousy": 0.5, "boredom": 0.6})
        state, delta = appraise_state(baseline, appraisal, entity_id=None)
        accepted = FullMentalState.init_from_list(state.to_list())
        self.assertEqual(len(accepted.to_list()), len(baseline.to_list()))
        self.assertEqual(len(delta), len(baseline.to_list()))
        for name in ("resentment", "jealousy", "boredom"):
            self.assertAlmostEqual(getattr(accepted.state_emotions, name), appraisal.emotion_delta[name])
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
