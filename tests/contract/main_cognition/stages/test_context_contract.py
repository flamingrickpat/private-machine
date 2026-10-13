"""Direct Context tests.

No model endpoint, module thread, or bus. The kernel test uses the real kernel
and the real Python role with a deterministic ``llm_complete`` tool.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path

from pm_workflows.runtime import WorkflowRuntime
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.main_cognition.schemas import ContextResult
from workflows.main_cognition.stage_support import StageOutputError
from workflows.main_cognition.stages.context import steps
from pm.memory.bash_view import MemoryBashEnvironment

MANIFEST = Path(steps.__file__).resolve().parent / "context.workflow.md"

CANDIDATES = [
    {"knoxel_id": 501, "text": "Jesse's kitchen caught fire last week"},
    {"knoxel_id": 502, "text": "Jesse dislikes the color beige"},
]


def example_request() -> dict:
    return {
        "companion_name": "Spamuel",
        "user_name": "Jesse",
        "state_summary": "smug but curious",
        "triage": {"items": [{"knoxel_id": 11, "category": "Critical"}]},
        "appraisal": {"feeling": "alarmed", "evidence_ids": [11]},
        "active_goals": ["keep the bit going", "avoid fires"],
        "capabilities": ["can read files", "can send replies"],
        "memory_candidates": CANDIDATES,
    }


class MessageTests(unittest.TestCase):
    def test_messages_are_system_then_user_with_candidates_and_ids(self):
        messages = steps.build_messages(example_request(), steps.load_prompts())
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Context", messages[0]["content"])
        self.assertIn("Jesse's kitchen caught fire last week", messages[1]["content"])
        self.assertIn("knoxel_id=501", messages[1]["content"])
        self.assertIn("keep the bit going", messages[1]["content"])


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
        runner = WorkflowRunner(MANIFEST, self.root, namespace="context", base_dir=PROJECT_ROOT)
        result = runner.invoke(WorkflowRequest("context-1", example_request()), runtime)
        return result, captured

    def test_real_role_parses_a_valid_context(self):
        payload = (
            '{"memories": ["Jesse\'s kitchen caught fire last week"],'
            ' "active_goals": ["avoid fires"],'
            ' "capabilities": ["can send replies"],'
            ' "candidate_ids": [501]}'
        )
        result, captured = self._run(payload)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        report = ContextResult.model_validate(result.report)
        self.assertEqual(report.memories, ["Jesse's kitchen caught fire last week"])
        self.assertEqual(report.candidate_ids, [501])
        self.assertEqual(report.active_goals, ["avoid fires"])
        self.assertEqual(captured["messages"][0]["role"], "system")

    def test_fabricated_candidate_id_raises(self):
        payload = (
            '{"memories": ["a made up memory"], "active_goals": [],'
            ' "capabilities": [], "candidate_ids": [999]}'
        )
        with self.assertRaises(StageOutputError):
            self._run(payload)

    def test_invalid_model_output_raises(self):
        with self.assertRaises(StageOutputError):
            self._run("this is not json at all")


class SelectionTests(unittest.TestCase):
    def test_validate_selection_rejects_ids_outside_candidates(self):
        result = ContextResult(candidate_ids=[999])
        with self.assertRaises(StageOutputError):
            steps.validate_selection(result, CANDIDATES)

    def test_validate_selection_accepts_supplied_ids(self):
        result = ContextResult(candidate_ids=[502])
        steps.validate_selection(result, CANDIDATES)


if __name__ == "__main__":
    unittest.main()
