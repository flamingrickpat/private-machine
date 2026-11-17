"""Deterministic annotation and span checks; live inference has a separate gate."""
from __future__ import annotations

import unittest

from pm.context_render.budget import ContextBudget
from pm.context_render.turns import Content, ContextTag, Text, Turn
from workflows.main_cognition.context_reduction import evidence_spans, evidence_text


class EvidenceSpanTests(unittest.TestCase):
    """Review partitions retain all characters and existing selection policy."""

    def test_long_single_line_and_unicode_are_complete_ordered_spans(self) -> None:
        original = "Jesse corrected the date: Monday, not Sunday. ☕\n" + "uninterrupted" * 3000
        budget = ContextBudget(context_window=32768, input_fraction=.85)
        spans = evidence_spans(original, budget, 1200)
        self.assertGreater(len(spans), 1)
        self.assertEqual("".join(spans), original)
        for span in spans:
            self.assertLessEqual(budget.estimate([Turn("user", span)]), 1200)

    def test_annotations_preserve_normal_bytes_and_existing_optional_policy(self) -> None:
        optional = ContextTag("optional", .7, record_id=123)
        original = Content("Required evidence\n") + Content(Text("Optional evidence", optional))
        marked = evidence_text(original, "required", [456])
        self.assertEqual(marked.text, original.text)
        self.assertEqual(marked.parts[1], original.parts[1])
        first = marked.parts[0]
        self.assertIsInstance(first, Text)
        assert isinstance(first, Text) and first.context is not None
        self.assertTrue(first.context.required)
        self.assertTrue(first.context.reducible)
        self.assertEqual(first.context.support_ids, (456,))
