"""Exercise real rendered evidence selection and conservative token planning."""
import unittest

from pm.context_render import Content, Role, Turn
from pm.context_render.budget import ContextBudget, ContextGroup, RequiredContextTooLarge, select_context


class ContextBudgetTests(unittest.TestCase):
    def test_required_sources_stay_together_and_optional_groups_use_workflow_salience(self) -> None:
        budget = ContextBudget(context_window=1800)
        groups = [
            ContextGroup("Required original report", Content("Source 17: the user reported a broken window."), 0, True, (17,)),
            ContextGroup("Low relevance", Content("unrelated " * 700), 0.1, source_ids=(3,)),
            ContextGroup("High relevance", Content("Source 21 directly contradicts source 17. " * 20), 0.9, source_ids=(17, 21)),
        ]
        selection = select_context([Turn(Role.System, "Inspect the evidence.")], groups, budget)
        self.assertIn("Required original report", selection.included)
        self.assertIn("High relevance", selection.included)
        self.assertIn("Low relevance", selection.omitted)
        self.assertLessEqual(selection.estimated_tokens, budget.input_ceiling)
        self.assertIn("Source 17", selection.turns[-1].content.text)
        self.assertIn("Omitted optional context groups: Low relevance", selection.turns[-1].content.text)

    def test_required_overflow_fails_before_any_inference(self) -> None:
        with self.assertRaises(RequiredContextTooLarge):
            select_context([Turn(Role.System, "Instructions " * 1000)], [], ContextBudget(context_window=300))

    def test_unused_capacity_does_not_add_unrequested_context(self) -> None:
        selection = select_context([Turn(Role.System, "One small decision.")], [], ContextBudget(context_window=131072))
        self.assertLess(selection.estimated_tokens, 1000)
        self.assertEqual(selection.included, ())

    def test_measurement_preserves_original_text_and_uses_a_safety_margin(self) -> None:
        turn = Turn(Role.User, "A message with IDs 1002, 1003 and an exact value 0.0012345.")
        bare = ContextBudget(context_window=1000, estimate_factor=1)
        safe = ContextBudget(context_window=1000)
        self.assertGreater(safe.estimate([turn]), bare.estimate([turn]))
        self.assertIn("0.0012345", turn.content.text)
