"""Separate inspection of completed work from authorization for new work."""
import unittest
import json
from pathlib import Path

from workflows.dialogue_assistant.tools import reply_tools
from workflows.dialogue_assistant.schemas import AssistantRequest
from workflows.dialogue_assistant.steps import build_messages, load_prompts, review_conversation


class ReplyToolTests(unittest.TestCase):
    def test_terminal_reply_can_inspect_but_cannot_repeat_execution(self):
        offered = reply_tools(terminal_receipt=True)
        names = {tool.name for tool in offered}
        self.assertIn("managed_status", names)
        self.assertIn("minecraft_observe", names)
        self.assertTrue(all(tool.read_only for tool in offered))
        self.assertTrue({"managed_start", "managed_update", "managed_terminate", "managed_answer"}.isdisjoint(names))

    def test_a_new_action_keeps_management_tools(self):
        self.assertIn("managed_start", {tool.name for tool in reply_tools()})

    def test_captured_task_history_reaches_review_with_its_current_request(self):
        capture = json.loads(Path("test_data/captures/alpha/assistant-task-conversation.json").read_text(encoding="utf-8"))
        request = AssistantRequest.model_validate(capture["request"])
        turns = review_conversation(request)
        self.assertEqual(turns[-1].text, request.current_user_message)
        self.assertEqual(turns[-1].speaker, request.user_name)
        self.assertTrue(any("Run only local_task" in turn.text for turn in turns[:-1]))
        self.assertEqual(build_messages(request, load_prompts())[-1].content, request.current_user_message)
