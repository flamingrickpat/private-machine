"""Classify the actual A10 parser rejection captured from local chat.

These checks establish HTTP error classification. Live generation verifies
the repaired inventory transcript separately.
"""
import json
from pathlib import Path
import unittest

import httpx

from runtime.model_io import rejected_message_protocol
from workflows.dialogue_assistant.steps import validate_public_speech
from workflows.public_claims import ClaimReview, validate_claim_review
from workflows.public_context import PublicContext


class MessageProtocolTests(unittest.TestCase):
    def test_captured_parser_rejection_is_permanent(self):
        captured = json.loads((Path(__file__).resolve().parents[1] / "test_data/captures/alpha/a10-message-protocol-error.json").read_text())
        response = httpx.Response(captured["status"], json=captured["body"])
        self.assertTrue(rejected_message_protocol(response))

    def test_other_server_failures_remain_transient(self):
        self.assertFalse(rejected_message_protocol(httpx.Response(500, content=b"service unavailable")))
        self.assertFalse(rejected_message_protocol(httpx.Response(503, json={"error": {"message": "busy"}})))
        self.assertFalse(rejected_message_protocol(httpx.Response(200, json={"choices": []})))

    def test_captured_tool_markup_cannot_be_published_as_speech(self):
        captured = (Path(__file__).resolve().parents[1] / "test_data/captures/alpha/a10-assistant-tool-markup.txt").read_text()
        with self.assertRaisesRegex(ValueError, "tool protocol"):
            validate_public_speech(captured)

    def _inventory_citations(self):
        """Remap the captured supported clauses to their actual invocation ID.

        The original model invented numeric source IDs. These checks isolate
        citation validation from the separately required live interpretation.
        """
        captured = json.loads((Path(__file__).resolve().parents[1] / "test_data/captures/alpha/a10-inventory-citation.json").read_text(encoding="utf-8"))
        review = ClaimReview.model_validate(captured["claim_review"])
        review.accepted = True
        for claim in review.claims[:2]:
            claim.status = "supported"
            claim.evidence_ids = []
            claim.tool_receipt_ids = [captured["tool_receipts"][0]["tool_call_id"]]
        return captured, review, PublicContext(companion_name="Spamuel", user_name="Jesse")

    def test_actual_observation_can_support_its_current_inventory(self):
        captured, review, context = self._inventory_citations()
        result = validate_claim_review(review, captured["candidate"], context, receipts=captured["tool_receipts"])
        self.assertTrue(result.accepted)

    def test_failed_observation_cannot_support_current_inventory(self):
        captured, review, context = self._inventory_citations()
        captured["tool_receipts"][0]["result"]["ok"] = False
        self.assertFalse(validate_claim_review(review, captured["candidate"], context,
                                              receipts=captured["tool_receipts"]).accepted)

    def test_tool_citation_cannot_excuse_an_unknown_numeric_id(self):
        captured, review, context = self._inventory_citations()
        review.claims[0].evidence_ids = [11203, 11204]
        self.assertFalse(validate_claim_review(review, captured["candidate"], context,
                                              receipts=captured["tool_receipts"]).accepted)

    def test_observation_does_not_replace_a_historical_source(self):
        captured, review, context = self._inventory_citations()
        review.claims[0].domain = "history"
        self.assertFalse(validate_claim_review(review, captured["candidate"], context,
                                              receipts=captured["tool_receipts"]).accepted)
