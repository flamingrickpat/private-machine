"""Retain action ids from captured real cycles."""
import json
from pathlib import Path
import unittest

from workflows.main_cognition.dispatch import resolve_same_cycle
from workflows.main_cognition.phases.route import stable_action_id
from workflows.main_cognition.schemas import ActionDispatch


class ActionDependencyTests(unittest.TestCase):
    def test_real_model_workflow_labels_resolve_to_parent_action_ids(self):
        capture = json.loads(Path("test_data/captures/alpha/action-workflow-dependencies.json").read_text(encoding="utf-8"))
        dispatches = [ActionDispatch.model_validate({**item, "action_id": stable_action_id(ActionDispatch.model_validate(item), index)})
                      for index, item in enumerate(capture["actions"])]
        resolved = resolve_same_cycle(dispatches)
        self.assertEqual([item.status for item in resolved], ["dispatched", "dispatched"])
        self.assertEqual(resolved[1].depends_on, [resolved[0].action_id])

    def test_repeated_workflow_labels_require_an_explicit_action_id(self):
        first = ActionDispatch(action_id="first", workflow_name="note_task", action_type="ToolCall", content="first note")
        second = first.model_copy(update={"action_id": "second", "content": "second note"})
        reply = ActionDispatch(action_id="reply", workflow_name="dialogue_assistant", action_type="Reply",
                               content="report", depends_on=["note_task"])
        self.assertEqual(resolve_same_cycle([first, second, reply])[-1].status, "rejected")
        explicit = reply.model_copy(update={"depends_on": ["second"]})
        self.assertEqual(resolve_same_cycle([first, second, explicit])[-1].status, "dispatched")
