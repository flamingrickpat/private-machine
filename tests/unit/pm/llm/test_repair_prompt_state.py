"""Deterministic repair-state checks; these do not certify model behavior."""
from threading import Event
import unittest

from pydantic import BaseModel, Field

from pm.context_render.turns import Turn
from pm.llm.prompt_encoding import normalize_turns
from pm.llm.structured import RepairConversation, RepeatedRejectionError


class PositiveAnswer(BaseModel):
    """A small validation contract for the repair state machine."""

    value: int = Field(gt=0)


class RepairPromptStateTests(unittest.TestCase):
    """Failed answers cannot accumulate at the expense of original evidence."""

    def test_rejections_retain_original_task_and_only_latest_answer(self) -> None:
        original = Turn("user", "Jesse said Monday. Sunday is a rejected interpretation.")
        state = RepairConversation([original], PositiveAnswer, None, None, None, Event())
        for value in range(6):
            raw = f'{{"value":{-value}}}'
            self.assertIsNone(state.accept(raw))
            turns = normalize_turns(state.turns)
            self.assertEqual(len(turns), 3)
            self.assertEqual(turns[0], original)
            self.assertEqual(turns[1].content.text, raw)
            self.assertIn("Constraints reported", turns[2].content.text)
        self.assertEqual(len(state.problems), 6)

    def test_repeated_alternating_errors_are_a_failed_repair_cycle(self) -> None:
        original = Turn("user", "Return a positive integer.")
        first = RepairConversation([original], PositiveAnswer, None, None, None, Event())
        second = RepairConversation([original], PositiveAnswer, None, None, None, Event())
        for state in (first, second):
            for raw in ('{"value":-1}', '{"value":"wrong"}'):
                self.assertIsNone(state.accept(raw))
        signature = first.check_repeated_rejections(None)
        with self.assertRaises(RepeatedRejectionError):
            second.check_repeated_rejections(signature)
