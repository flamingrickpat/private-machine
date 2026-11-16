"""Deterministic checks for the simulation history budget helpers."""
import unittest

from simulation.actors import ConversationTurn
from simulation.history import bound_conversation, conversation_tokens


def turn(index: int, text: str) -> ConversationTurn:
    return ConversationTurn(speaker="Jesse", at=f"2026-06-02T22:{index:02d}:00Z", text=text)


class HistoryBudgetTests(unittest.TestCase):
    def test_bound_conversation_keeps_the_most_recent_turns_that_fit(self) -> None:
        turns = [turn(index, "a short line " * 20) for index in range(10)]
        bounded = bound_conversation(turns, conversation_tokens(turns) // 2)
        self.assertLess(len(bounded), len(turns))
        self.assertEqual(bounded, turns[-len(bounded):])

    def test_bound_conversation_keeps_everything_when_it_fits(self) -> None:
        turns = [turn(index, "hi") for index in range(3)]
        self.assertEqual(bound_conversation(turns, 10_000), turns)

    def test_conversation_tokens_grows_with_text(self) -> None:
        self.assertLess(conversation_tokens([turn(0, "hi")]),
                        conversation_tokens([turn(0, "hi " * 200)]))
