"""Direct Triage tests.

No model endpoint, module thread, or bus. The kernel test uses the real kernel
and the real Python role with a deterministic ``llm_complete`` tool.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path

from pm_workflows.runtime import WorkflowRuntime
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.main_cognition.schemas import TriageResult
from workflows.main_cognition.stage_support import StageOutputError
from workflows.main_cognition.stages.triage import steps
from pm.memory.bash_view import MemoryBashEnvironment

MANIFEST = Path(steps.__file__).resolve().parent / "triage.workflow.md"


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
    }


class MessageTests(unittest.TestCase):
    def test_messages_are_system_then_user_with_every_stimulus(self):
        messages = steps.build_messages(example_request(), steps.load_prompts())
        self.assertEqual([message.role.value for message in messages], ["system", "user"])
        self.assertIn("Spamuel", messages[0].content.text)
        self.assertIn("the kitchen is on fire", messages[1].content.text)
        self.assertIn("also the carpet is beige", messages[1].content.text)
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
        runner = WorkflowRunner(MANIFEST, self.root, namespace="triage", base_dir=PROJECT_ROOT)
        result = runner.invoke(WorkflowRequest("triage-1", example_request()), runtime)
        return result, captured

    def test_real_role_parses_a_valid_classification(self):
        payload = (
            '{"items": ['
            '{"knoxel_id": 11, "category": "Critical", "input_kind": "message",'
            ' "urgency": 0.95, "relevance": 0.9, "reasoning": "fire", "evidence_ids": [11]},'
            '{"knoxel_id": 12, "category": "Insignificant", "input_kind": "message",'
            ' "urgency": 0.0, "relevance": 0.1, "reasoning": "color", "evidence_ids": [12]}]}'
        )
        result, captured = self._run(payload)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        report = TriageResult.model_validate(result.report)
        self.assertEqual([item.category for item in report.items],
                         ["Critical", "Insignificant"])
        self.assertEqual(captured["messages"][0].role.value, "system")

    def test_invalid_model_output_raises_instead_of_a_successful_empty_answer(self):
        with self.assertRaises(StageOutputError):
            self._run("this is not json at all")


if __name__ == "__main__":
    unittest.main()
