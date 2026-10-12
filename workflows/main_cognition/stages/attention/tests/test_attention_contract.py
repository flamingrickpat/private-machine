"""Direct Attention tests.

No model endpoint, module thread, or bus. The kernel test uses the real kernel
and the real Python role with a deterministic ``llm_complete`` tool.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path

from pm_workflows.runtime import WorkflowRuntime
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.main_cognition.schemas import AttentionNomination, AttentionResult
from workflows.main_cognition.stage_support import StageOutputError
from workflows.main_cognition.stages.attention import steps
from pm.memory.bash_view import GhostBashEnvironment

MANIFEST = Path(steps.__file__).resolve().parent / "attention.workflow.md"


def example_request() -> dict:
    return {
        "companion_name": "Spamuel",
        "user_name": "Jesse",
        "state_summary": "smug but curious",
        "active_goals": ["keep the bit going"],
        "triage": {"items": [{"knoxel_id": 11, "category": "Critical"}]},
        "appraisal": {"feeling": "alarmed", "reasoning": "fire"},
        "context": {
            "memories": [],
            "active_goals": ["keep the bit going"],
            "capabilities": [],
            "candidate_ids": [11, 12],
        },
        "candidates": [
            {"knoxel_id": 11, "kind": "telemetry", "text": "the kitchen is on fire"},
            {"knoxel_id": 12, "kind": "vision", "text": "the carpet is beige"},
        ],
    }


class MessageTests(unittest.TestCase):
    def test_messages_are_system_then_user_with_every_candidate(self):
        messages = steps.build_messages(example_request(), steps.load_prompts())
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Spamuel", messages[0]["content"])
        self.assertIn("the kitchen is on fire", messages[1]["content"])
        self.assertIn("the carpet is beige", messages[1]["content"])
        self.assertIn("id=11", messages[1]["content"])


class ValidationGuardTests(unittest.TestCase):
    def test_selected_id_outside_candidates_is_rejected(self):
        result = AttentionResult(selected_ids=[999], reasoning="bad")
        with self.assertRaises(StageOutputError):
            steps.ensure_candidate_ids(result, {11, 12})

    def test_target_id_outside_candidates_is_rejected(self):
        result = AttentionResult(
            nominations=[AttentionNomination(target_ids=[999], reasoning="bad")],
            reasoning="bad",
        )
        with self.assertRaises(StageOutputError):
            steps.ensure_candidate_ids(result, {11, 12})

    def test_supplied_ids_pass(self):
        result = AttentionResult(
            nominations=[AttentionNomination(target_ids=[11])],
            selected_ids=[11],
            reasoning="fine",
        )
        steps.ensure_candidate_ids(result, {11, 12})


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(".runtime/w5-direct-tests")
        self.root.mkdir(parents=True, exist_ok=True)

    def _run(self, completion: str):
        captured: dict = {}

        def llm_complete(messages, **generation):
            captured["messages"] = messages
            captured["generation"] = generation
            return completion

        environment = GhostBashEnvironment({}, {}, "read")
        runtime = WorkflowRuntime(
            environment=environment, stop_event=threading.Event(),
            tools={"llm_complete": llm_complete}, propagate_python_errors=True,
        )
        runner = WorkflowRunner(MANIFEST, self.root, namespace="attention", base_dir=PROJECT_ROOT)
        result = runner.invoke(WorkflowRequest("attention-1", example_request()), runtime)
        return result, captured

    def test_real_role_parses_a_valid_nomination(self):
        payload = (
            '{"nominations": ['
            '{"target_ids": [11], "attention_kind": "danger", "score": 0.95,'
            ' "novelty": 0.6, "urgency": 0.95, "goal_relevance": 0.8,'
            ' "reportability": 0.9, "reasoning": "the fire demands action"},'
            '{"target_ids": [12], "attention_kind": "detail", "score": 0.1,'
            ' "novelty": 0.0, "urgency": 0.0, "goal_relevance": 0.0,'
            ' "reportability": 0.1, "reasoning": "just a color"}],'
            ' "selected_ids": [11], "reasoning": "the fire matters"}'
        )
        result, captured = self._run(payload)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        report = AttentionResult.model_validate(result.report)
        self.assertEqual(report.selected_ids, [11])
        self.assertIn(11, report.selected_ids)
        self.assertNotIn(12, report.selected_ids)
        scores = {nomination.target_ids[0]: nomination.score for nomination in report.nominations}
        self.assertGreater(scores[11], scores[12])
        self.assertEqual(captured["messages"][0]["role"], "system")

    def test_invalid_model_output_raises_instead_of_a_successful_empty_answer(self):
        with self.assertRaises(StageOutputError):
            self._run("this is not json at all")


if __name__ == "__main__":
    unittest.main()
