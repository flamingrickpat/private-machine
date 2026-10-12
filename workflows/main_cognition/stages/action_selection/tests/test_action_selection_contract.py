"""Direct Action Selection tests.

No model endpoint, module thread, or bus. The kernel test uses the real kernel
and the real Python role with a deterministic ``llm_complete`` tool.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path

from pm_workflows.runtime import WorkflowRuntime
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.main_cognition.schemas import ActionIntent, ActionSelectionResult
from workflows.main_cognition.stage_support import StageOutputError
from workflows.main_cognition.stages.action_selection import steps
from pm.memory.bash_view import GhostBashEnvironment

MANIFEST = Path(steps.__file__).resolve().parent / "action_selection.workflow.md"


def example_request() -> dict:
    return {
        "companion_name": "Spamuel",
        "user_name": "Jesse",
        "state_summary": "smug but worried",
        "active_goals": ["keep the bit going", "put out the fire"],
        "workspace": {
            "frame_reason": "the kitchen is on fire",
            "focus_ids": [11],
            "control_mode": "reactive",
            "subjective_experience": "Everything is loud.",
        },
        "goals": {"changes": [{"change": "add", "goal": "put out the fire"}]},
        "capabilities": ["I can communicate via text.", "I can log events."],
        "available_workflows": [
            {"name": "dialogue_story", "description": "Generate one in-character Story reply."},
            {"name": "self_management", "description": "Log an internal event."},
        ],
    }


class MessageTests(unittest.TestCase):
    def test_messages_are_system_then_user_with_every_workflow(self):
        messages = steps.build_messages(example_request(), steps.load_prompts())
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Spamuel", messages[0]["content"])
        self.assertIn("dialogue_story", messages[1]["content"])
        self.assertIn("self_management", messages[1]["content"])
        self.assertIn("I can communicate via text.", messages[1]["content"])
        self.assertIn("the kitchen is on fire", messages[1]["content"])


class ValidationGuardTests(unittest.TestCase):
    def test_unavailable_workflow_name_is_rejected(self):
        result = ActionSelectionResult(actions=[
            ActionIntent(action_type="Reply", workflow_name="teleport",
                         content="vanish from the room"),
        ])
        with self.assertRaises(StageOutputError):
            steps.validate_actions(result, example_request()["available_workflows"])

    def test_unsupported_action_type_is_rejected(self):
        result = ActionSelectionResult(actions=[
            ActionIntent(action_type="SendEmail", workflow_name="dialogue_story",
                         content="send mail"),
        ])
        with self.assertRaises(StageOutputError):
            steps.validate_actions(result, example_request()["available_workflows"])

    def test_capability_claim_outside_the_list_is_rejected(self):
        result = ActionSelectionResult(actions=[
            ActionIntent(action_type="Reply", workflow_name="dialogue_story",
                         content="I can see the camera"),
        ])
        with self.assertRaises(StageOutputError):
            steps.check_capabilities(result, ["I can communicate via text."])

    def test_declared_capability_passes(self):
        result = ActionSelectionResult(actions=[
            ActionIntent(action_type="Reply", workflow_name="dialogue_story",
                         content="I can see the camera"),
        ])
        steps.check_capabilities(result, ["I can see through the camera."])

    def test_capability_refusal_is_not_a_claim(self):
        result = ActionSelectionResult(actions=[
            ActionIntent(action_type="Reply", workflow_name="dialogue_story",
                         content="I can't walk to the store, but I'll keep you company"),
        ])
        steps.check_capabilities(result, ["I can communicate via text."])

    def test_vision_refusal_is_not_a_claim(self):
        result = ActionSelectionResult(actions=[
            ActionIntent(action_type="Reply", workflow_name="dialogue_story",
                         content="I cannot see your kitchen from here. Describe what is burning."),
        ])
        steps.check_capabilities(result, ["I can communicate via text."])

    def test_affirmative_mobility_claim_is_still_rejected(self):
        result = ActionSelectionResult(actions=[
            ActionIntent(action_type="Reply", workflow_name="dialogue_story",
                         content="I will walk to the store and grab a drink"),
        ])
        with self.assertRaises(StageOutputError):
            steps.check_capabilities(result, ["I can communicate via text."])


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
        runner = WorkflowRunner(
            MANIFEST, self.root, namespace="action_selection", base_dir=PROJECT_ROOT,
        )
        result = runner.invoke(WorkflowRequest("action_selection-1", example_request()), runtime)
        return result, captured

    def test_real_role_parses_two_action_intents(self):
        payload = (
            '{"actions": ['
            '{"action_type": "Reply", "workflow_name": "dialogue_story",'
            ' "content": "reassure Jesse that the fire is handled",'
            ' "arguments": {}, "depends_on": []},'
            '{"action_type": "ToolCall", "workflow_name": "self_management",'
            ' "content": "log the fire alarm event",'
            ' "arguments": {"reason": "fire"}, "depends_on": ["dialogue_story"]}],'
            ' "reasoning": "speak first, then record the event"}'
        )
        result, captured = self._run(payload)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        selection = ActionSelectionResult.model_validate(result.report)
        self.assertEqual(len(selection.actions), 2)
        self.assertEqual([action.action_type for action in selection.actions],
                         ["Reply", "ToolCall"])
        self.assertEqual([action.workflow_name for action in selection.actions],
                         ["dialogue_story", "self_management"])
        # depends_on is preserved exactly as the model wrote it. The parent
        # assigns the durable action ids.
        self.assertEqual(selection.actions[1].depends_on, ["dialogue_story"])
        for action in selection.actions:
            self.assertTrue(action.content.strip(), "intent content must not be empty")
        self.assertTrue(selection.reasoning.strip())
        self.assertEqual(captured["messages"][0]["role"], "system")

    def test_invalid_model_output_raises_instead_of_a_successful_empty_answer(self):
        with self.assertRaises(StageOutputError):
            self._run("this is not json at all")


if __name__ == "__main__":
    unittest.main()
