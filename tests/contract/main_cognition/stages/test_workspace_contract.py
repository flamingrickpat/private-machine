"""Direct Workspace tests.

No model endpoint, module thread, or bus. The kernel test uses the real kernel
and the real Python role with a deterministic ``llm_complete`` tool.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path

from pm_workflows.runtime import WorkflowRuntime
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.main_cognition.schemas import WorkspaceResult
from workflows.main_cognition.stage_support import StageOutputError
from workflows.main_cognition.stages.workspace import steps
from pm.memory.bash_view import MemoryBashEnvironment

MANIFEST = Path(steps.__file__).resolve().parent / "workspace.workflow.md"


def example_request() -> dict:
    return {
        "companion_name": "Spamuel",
        "user_name": "Jesse",
        "state_summary": "smug but curious",
        "active_goals": ["keep the bit going"],
        "attention": {
            "nominations": [
                {"target_ids": [11], "attention_kind": "danger", "score": 0.95},
            ],
            "selected_ids": [11],
            "reasoning": "the fire matters",
        },
        "goals": {"changes": [{"change": "retain", "goal": "keep the bit going"}]},
        "candidate_texts": [
            {"knoxel_id": 11, "text": "the kitchen is on fire"},
            {"knoxel_id": 12, "text": "the carpet is beige"},
        ],
    }


class MessageTests(unittest.TestCase):
    def test_messages_are_system_then_user_with_candidate_text(self):
        messages = steps.build_messages(example_request(), steps.load_prompts())
        self.assertEqual([message.role.value for message in messages], ["system", "user"])
        self.assertIn("Spamuel", messages[0].content.text)
        self.assertIn("the kitchen is on fire", messages[1].content.text)
        self.assertIn("the carpet is beige", messages[1].content.text)
        self.assertIn("id=11", messages[1].content.text)


class ValidationGuardTests(unittest.TestCase):
    def test_focus_id_outside_candidates_is_rejected(self):
        result = WorkspaceResult(focus_ids=[999], control_mode="reactive")
        with self.assertRaises(StageOutputError):
            steps.ensure_valid_result(result, {11, 12})

    def test_bad_control_mode_is_rejected(self):
        result = WorkspaceResult(focus_ids=[11], control_mode="autopilot")
        with self.assertRaises(StageOutputError):
            steps.ensure_valid_result(result, {11, 12})

    def test_supplied_ids_and_mode_pass(self):
        result = WorkspaceResult(focus_ids=[11], control_mode="reflective")
        steps.ensure_valid_result(result, {11, 12})

    def test_attention_selection_narrows_allowed_ids(self):
        request = example_request()
        self.assertEqual(steps.allowed_focus_ids(request), {11})


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
        runner = WorkflowRunner(MANIFEST, self.root, namespace="workspace", base_dir=PROJECT_ROOT)
        result = runner.invoke(WorkflowRequest("workspace-1", example_request()), runtime)
        return result, captured

    def test_real_role_parses_a_valid_frame(self):
        payload = (
            '{"frame_reason": "the fire dominates", "focus_ids": [11],'
            ' "control_mode": "reflective",'
            ' "subjective_experience": "The smoke fills my attention and everything else goes quiet."}'
        )
        result, captured = self._run(payload)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        report = WorkspaceResult.model_validate(result.report)
        self.assertEqual(report.focus_ids, [11])
        self.assertEqual(report.control_mode, "reflective")
        self.assertIn("smoke", report.subjective_experience)
        self.assertEqual(captured["messages"][0].role.value, "system")

    def test_invalid_model_output_raises_instead_of_a_successful_empty_answer(self):
        with self.assertRaises(StageOutputError):
            self._run("this is not json at all")


if __name__ == "__main__":
    unittest.main()
