"""Check exact-source typography repair with captured reviewer quotations.

These inputs check quotation mechanics, not a model's judgment or accuracy.
"""
from pathlib import Path
import json
import unittest

from dev.work_items.a18b.review import exact_quote_span

ROOT = Path(__file__).resolve().parents[3]


class CognitiveQuotationTests(unittest.TestCase):
    def test_captured_typography_maps_to_exact_original_spans(self):
        fixture = ROOT / "test_data/captures/alpha/cognitive-quotation-boundaries.json"
        capture = json.loads(fixture.read_text(encoding="utf-8"))
        for item in capture["quotations"]:
            with self.subTest(index=item["index"]):
                span = exact_quote_span(item["source"], item["attempted_quote"])
                self.assertIsNotNone(span)
                self.assertIn(span, item["source"])

    def test_words_and_order_cannot_change(self):
        for source, attempted in (("not ready", "never ready"),
                                  ("I am ready.", "I am very ready"), ("red green", "green red")):
            with self.subTest(attempted=attempted):
                self.assertIsNone(exact_quote_span(source, attempted))

    def test_excerpt_case_is_restored_from_the_original_source(self):
        self.assertEqual(exact_quote_span("I am ready.", "i am ready"), "I am ready")
        self.assertEqual(exact_quote_span("The bee is lost.", "the bee is lost"), "The bee is lost")
        self.assertIsNone(exact_quote_span("Polish paint; polish paint.", "POLISH PAINT!"))

    def test_repair_rejects_ambiguous_normalized_spans(self):
        self.assertIsNone(exact_quote_span("ready, now. ready - now.", "ready; now!"))
        self.assertIsNone(exact_quote_span("ready", "!!!"))
        self.assertIsNone(exact_quote_span("ready", ""))

    def test_verbatim_quote_is_unchanged(self):
        quote = "ready, now."
        self.assertEqual(exact_quote_span("ready, now. ready, now.", quote), quote)
