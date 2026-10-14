"""Direct Goals tests.

No model endpoint, module thread, or bus. The kernel test uses the real kernel
and the real Python role with a deterministic ``llm_complete`` tool.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path

from pm_workflows.runtime import WorkflowRuntime
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.main_cognition.schemas import GoalsResult
from workflows.main_cognition.stage_support import StageOutputError
from workflows.main_cognition.stages.goals import steps
from pm.memory.bash_view import MemoryBashEnvironment

MANIFEST = Path(steps.__file__).resolve().parent / "goals.workflow.md"


def example_request() -> dict:
    return {
        "companion_name": "Spamuel",
        "user_name": "Jesse",
        "state_summary": "smug but curious",
        "active_goals": ["keep the bit going"],
        "current_goals": ["keep the bit going", "put out the fire"],
        "appraisal": {"feeling": "alarmed", "reasoning": "the kitchen is on fire"},
        "context": {
            "memories": [],
            "active_goals": ["keep the bit going"],
            "capabilities": [],
            "candidate_ids": [11],
        },
    }


class MessageTests(unittest.TestCase):
    def test_messages_are_system_then_user_with_the_current_goals(self):
        messages = steps.build_messages(example_request(), steps.load_prompts())
        self.assertEqual([message.role.value for message in messages], ["system", "user"])
        self.assertIn("Spamuel", messages[0].content.text)
        self.assertIn("keep the bit going", messages[1].content.text)
        self.assertIn("put out the fire", messages[1].content.text)
        self.assertIn("alarmed", messages[1].content.text)

    def test_accepted_inner_proposals_render_as_motive_evidence(self):
        payload = example_request()
        payload["inner_proposals"] = [
            {"kind": "action_tendency", "reality": "inferred",
             "text": "I want to set a limit about the teasing.", "target_ids": [11]},
            {"kind": "belief", "reality": "inferred",
             "text": "I keep my promises even when it is hard.", "target_ids": []},
        ]
        user = steps.build_messages(payload, steps.load_prompts())[1].content.text
        self.assertIn("proposals, not commands", user)
        self.assertIn("(action_tendency; inferred) I want to set a limit", user)
        self.assertIn("(belief; inferred) I keep my promises", user)

    def test_missing_inner_proposals_render_as_none(self):
        user = steps.build_messages(example_request(), steps.load_prompts())[1].content.text
        self.assertIn("(none)", user)


class ValidationGuardTests(unittest.TestCase):
    def test_unsupported_change_value_is_rejected(self):
        result = GoalsResult.model_validate({"changes": [
            {"change": "edit_prompt", "goal": "tweak prompts",
             "reason": "not allowed", "priority": 0.5},
        ]})
        with self.assertRaises(StageOutputError):
            steps.validate_changes(result)

    def test_every_supported_change_value_passes(self):
        result = GoalsResult.model_validate({"changes": [
            {"change": "retain", "goal": "keep the bit going",
             "reason": "still fits", "priority": 0.5},
            {"change": "add", "goal": "put out the fire",
             "reason": "the kitchen is on fire", "priority": 0.9},
            {"change": "complete", "goal": "water the plants",
             "reason": "already done", "priority": 0.1},
            {"change": "reprioritize", "goal": "keep the bit going",
             "reason": "the fire outranks the bit", "priority": 0.4},
            {"change": "defer", "goal": "explore the cave",
             "reason": "it is night", "priority": 0.2},
            {"change": "refuse", "goal": "give away my diamonds",
             "reason": "they are mine", "priority": 0.1},
        ]})
        steps.validate_changes(result)


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
        runner = WorkflowRunner(MANIFEST, self.root, namespace="goals", base_dir=PROJECT_ROOT)
        result = runner.invoke(WorkflowRequest("goals-1", example_request()), runtime)
        return result, captured

    def test_real_role_parses_a_valid_change_list(self):
        payload = (
            '{"changes": ['
            '{"change": "add", "goal": "put out the fire",'
            ' "reason": "the kitchen is on fire", "priority": 0.9},'
            '{"change": "reprioritize", "goal": "keep the bit going",'
            ' "reason": "the fire outranks the bit", "priority": 0.4},'
            '{"change": "complete", "goal": "water the plants",'
            ' "reason": "already done", "priority": 0.1}]}'
        )
        result, captured = self._run(payload)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        report = GoalsResult.model_validate(result.report)
        self.assertEqual(
            [change.change for change in report.changes],
            ["add", "reprioritize", "complete"],
        )
        self.assertEqual(captured["messages"][0].role.value, "system")

    def test_unsupported_change_from_the_model_raises(self):
        payload = (
            '{"changes": ['
            '{"change": "edit_prompt", "goal": "tweak prompts",'
            ' "reason": "not allowed", "priority": 0.5}]}'
        )
        with self.assertRaises(StageOutputError):
            self._run(payload)

    def test_invalid_model_output_raises_instead_of_a_successful_empty_answer(self):
        with self.assertRaises(StageOutputError):
            self._run("this is not json at all")


if __name__ == "__main__":
    unittest.main()
