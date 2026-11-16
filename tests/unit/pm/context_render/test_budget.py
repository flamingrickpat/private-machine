"""Exercise real rendered evidence selection and conservative token planning."""
import unittest
from random import Random

from pm.context_render import Content, Role, Turn
from pm.context_render.budget import ContextBudget, ContextGroup, RequiredContextTooLarge, batch_prompt, fit_turns, optional_text, rank_context, select_context
from pm.context_render.turns import ContextTag


class ContextBudgetTests(unittest.TestCase):
    def test_batches_cover_every_task_once_under_identical_context(self) -> None:
        context = [Turn(Role.System, "Rate every candidate against this same moment."),
                   Turn(Role.User, "Required current event." + optional_text(" historical" * 2000, ContextTag("history", 1)))]
        groups = [ContextGroup(str(index), Content(f"Candidate {index}: " + "evidence " * 240), index,
                               source_ids=(index,), recency=index) for index in range(8)]
        def instruction(chosen: list[ContextGroup] | tuple[ContextGroup, ...]) -> Content:
            return Content("Rate exactly these candidates:\n" + "\n".join(group.content.text for group in chosen))
        budget = ContextBudget(context_window=1800)
        batches = batch_prompt(context, groups, budget, lambda chosen: instruction(list(chosen)))
        self.assertGreater(len(batches), 1)
        self.assertEqual(sorted(name for batch in batches for name in batch.included), sorted(group.name for group in groups))
        for batch in batches:
            self.assertEqual(batch.turns[:-1], batches[0].turns[:-1])
            self.assertIn("Required current event.", batch.turns[1].content.text)
            self.assertLessEqual(batch.estimated_tokens, budget.input_ceiling)

    def test_an_indivisible_oversized_task_is_not_clipped(self) -> None:
        groups = [ContextGroup("whole source", Content("exact source " * 2000), 1)]
        with self.assertRaises(RequiredContextTooLarge):
            batch_prompt([Turn(Role.System, "Keep exact evidence.")], groups, ContextBudget(context_window=500),
                         lambda chosen: chosen[0].content)

    def test_ranking_balances_old_salient_and_recent_context_with_random_variety(self) -> None:
        groups = [ContextGroup("old important", Content(""), 1, recency=0),
                  ContextGroup("recent", Content(""), 0, recency=1),
                  ContextGroup("old unimportant", Content(""), 0, recency=0)]
        orders: set[tuple[str, ...]] = set()
        for seed in range(10):
            names = tuple(group.name for group in rank_context(groups, random=Random(seed)))
            self.assertEqual(names[-1], "old unimportant")
            orders.add(names)
        self.assertGreater(len(orders), 1)

    def test_complete_prompt_preserves_support_and_repairs_when_optional_evidence_is_removed(self) -> None:
        source = optional_text("\nOriginal source 17.", ContextTag("source", .1, record_id=17))
        claim = optional_text("\nClaim 21.", ContextTag("claim", 2, record_id=21, support_ids=(17,)))
        unrelated = optional_text(" unrelated" * 3000, ContextTag("unrelated", 0, record_id=99))
        turns = [Turn(Role.System, "Inspect exact evidence."), Turn(Role.User, source + claim + unrelated),
                 Turn(Role.Assistant, "rejected answer"), Turn(Role.User, "Repair this specific error."),
                 Turn(Role.Assistant, "prefill:")]
        budget = ContextBudget(context_window=900)
        selected = fit_turns(turns, budget)
        text = "\n".join(turn.content.text for turn in selected.turns)
        self.assertIn("Original source 17.", text)
        self.assertIn("Claim 21.", text)
        self.assertIn("Repair this specific error.", text)
        self.assertNotIn(" unrelated", text)
        self.assertEqual(selected.turns[-1].content.text, "prefill:")
        self.assertLessEqual(selected.estimated_tokens, budget.input_ceiling)
        self.assertIn(" unrelated", turns[1].content.text)

    def test_fitting_keeps_exact_turns_when_everything_fits(self) -> None:
        turns = [Turn(Role.System, "Instructions."), Turn(Role.User,
            "Context:" + optional_text("\nExact bytes.", ContextTag("record", 1)))]
        self.assertEqual(tuple(turns), fit_turns(turns, ContextBudget(context_window=1000)).turns)

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
