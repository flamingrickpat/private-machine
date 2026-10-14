"""Direct Appraisal tests.

No model endpoint, module thread, or bus. The kernel test uses the real kernel
and the real Python role with a deterministic ``llm_complete`` tool.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path

from pm_workflows.runtime import WorkflowRuntime
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.main_cognition.schemas import AppraisalResult
from workflows.main_cognition.stage_support import StageOutputError
from workflows.main_cognition.stages.appraisal import steps
from pm.memory.bash_view import MemoryBashEnvironment

MANIFEST = Path(steps.__file__).resolve().parent / "appraisal.workflow.md"


def example_request() -> dict:
    return {
        "companion_name": "Spamuel",
        "user_name": "Jesse",
        "state_summary": "smug but curious",
        "active_goals": ["keep the bit going"],
        "stimuli": [
            {"knoxel_id": 11, "source": "Jesse", "kind": "user.input",
             "text": "the kitchen is on fire"},
            {"knoxel_id": 12, "source": "Jesse", "kind": "user.input",
             "text": "also the carpet is beige"},
        ],
        "triage": {
            "items": [
                {"knoxel_id": 11, "category": "Critical", "input_kind": "message",
                 "urgency": 0.95, "relevance": 0.9, "reasoning": "the kitchen is on fire",
                 "evidence_ids": [11]},
                {"knoxel_id": 12, "category": "Insignificant", "input_kind": "message",
                 "urgency": 0.0, "relevance": 0.1, "reasoning": "color trivia",
                 "evidence_ids": [12]},
            ],
        },
    }


class MessageTests(unittest.TestCase):
    def test_messages_are_system_then_user_with_companion_and_triage(self):
        messages = steps.build_messages(example_request(), steps.load_prompts())
        self.assertEqual([message.role.value for message in messages], ["system", "user"])
        self.assertIn("Spamuel", messages[0].content.text)
        self.assertIn("Spamuel", messages[1].content.text)
        self.assertIn("the kitchen is on fire", messages[1].content.text)
        self.assertIn("color trivia", messages[1].content.text)
        self.assertIn("id=11", messages[1].content.text)


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

        environment = MemoryBashEnvironment({}, {}, "read")
        runtime = WorkflowRuntime(
            environment=environment, stop_event=threading.Event(),
            tools={"llm_complete": llm_complete}, propagate_python_errors=True,
        )
        runner = WorkflowRunner(MANIFEST, self.root, namespace="appraisal", base_dir=PROJECT_ROOT)
        result = runner.invoke(WorkflowRequest("appraisal-1", example_request()), runtime)
        return result, captured

    def test_real_role_parses_a_valid_appraisal(self):
        payload = (
            '{"feeling": "The kitchen is burning and I cannot look away.",'
            ' "emotion_delta": {"valence": -0.7, "anxiety": 0.8},'
            ' "need_delta": {"connection": 0.3, "autonomy": -0.2},'
            ' "reasoning": "A fire is urgent and personal.",'
            ' "evidence_ids": [11]}'
        )
        result, captured = self._run(payload)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        report = AppraisalResult.model_validate(result.report)
        self.assertIn("kitchen", report.feeling)
        self.assertAlmostEqual(report.emotion_delta["anxiety"], 0.8)
        self.assertAlmostEqual(report.emotion_delta["valence"], -0.7)
        self.assertAlmostEqual(report.need_delta["connection"], 0.3)
        self.assertEqual(report.evidence_ids, [11])
        self.assertEqual(captured["messages"][0].role.value, "system")
        self.assertAlmostEqual(captured["generation"]["temperature"], 0.2)

    def test_invalid_model_output_raises_instead_of_a_successful_empty_answer(self):
        with self.assertRaises(StageOutputError):
            self._run("this is not json at all")


if __name__ == "__main__":
    unittest.main()
