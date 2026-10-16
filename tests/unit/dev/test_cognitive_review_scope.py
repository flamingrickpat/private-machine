"""Check bounded review groups without fixed completions or model inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from jsonschema import Draft202012Validator

from dev.work_items.a18b.constants import MAX_TEST_INPUT_TOKENS, REVIEW_CRITERIA_PER_CALL
from dev.work_items.a18b.identity import save_review_origin
from dev.work_items.a18b.review import input_budget, review_parts, review_turns, validate_provider_budget, review_schema
from dev.work_items.a18b.runner import load_suite, make_packets, public_messages
from pm.context_render.turns import Role, Turn


class CognitiveReviewScopeTests(unittest.TestCase):
    def test_criterion_groups_preserve_all_evidence_and_exact_coverage(self):
        suite = load_suite()
        messages = public_messages([{"at": datetime.now(timezone.utc), "speaker": "Spamuel", "text": "Boundary input."}])
        packet = make_packets(suite.scenarios[0], suite, messages, (), review_from_message="m0001")["editor"]
        parts = review_parts(packet)
        self.assertGreater(len(parts), 1)
        self.assertEqual(tuple(item for part in parts for item in part.criteria), packet.criteria)
        for part in parts:
            self.assertLessEqual(len(part.criteria), REVIEW_CRITERIA_PER_CALL)
            self.assertEqual(part.conversation, packet.conversation)
            self.assertEqual(part.review_from_message, packet.review_from_message)
            self.assertEqual(part.premise, packet.premise)
            self.assertEqual(part.facts, packet.facts)
            supplied = json.loads(review_turns(part)[1].content.text)
            self.assertNotIn("previous_review", supplied)
            self.assertNotIn("summary", supplied)
            self.assertNotIn("gate", supplied)

    def test_shared_test_budget_is_thirty_thousand_tokens(self):
        self.assertEqual(MAX_TEST_INPUT_TOKENS, 30_000)
        tokens, measurement = input_budget("qwen/qwen3.8-flash", [Turn(Role.User, "A small input.")])
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, MAX_TEST_INPUT_TOKENS)
        self.assertTrue(measurement.startswith("estimated."))

    def test_oversized_text_fails_without_truncation_or_inference(self):
        turns = [Turn(Role.User, "word " * (MAX_TEST_INPUT_TOKENS * 2))]
        with self.assertRaisesRegex(ValueError, "Nothing was truncated"):
            input_budget("qwen/qwen3.8-flash", turns)
        self.assertEqual(len(turns[0].content.text), 5 * MAX_TEST_INPUT_TOKENS * 2)

    def test_response_schema_counts_toward_the_same_input_limit(self):
        with self.assertRaisesRegex(ValueError, "estimated-token allowance"):
            input_budget("qwen/qwen3.8-flash", [Turn(Role.User, "Small text.")],
                         schema={"description": "word " * (MAX_TEST_INPUT_TOKENS * 2)})

    def test_wire_schema_requires_citations_for_pass_and_fail(self):
        validator = Draft202012Validator(review_schema())
        for verdict in ("pass", "fail"):
            with self.subTest(verdict=verdict):
                judgment = {"criterion_id": "voice", "verdict": verdict, "reason": "Schema boundary input."}
                payload = {"judgments": [judgment], "summary": "Schema check only."}
                self.assertFalse(validator.is_valid(payload))
                judgment["citations"] = [{"source_id": "m0001", "quote": "Boundary input."}]
                self.assertTrue(validator.is_valid(payload))

    def test_wire_branches_are_complete_objects_for_provider_compilers(self):
        for variant in review_schema()["$defs"]["Judgment"]["anyOf"]:
            self.assertEqual(variant["type"], "object")
            self.assertTrue({"criterion_id", "verdict", "reason"} <= set(variant["required"]))
            self.assertTrue({"criterion_id", "verdict", "reason", "citations"} <= variant["properties"].keys())

    def test_single_criterion_wire_schema_fixes_its_id_and_count(self):
        suite = load_suite()
        packet = make_packets(suite.scenarios[0], suite, public_messages([{
            "at": datetime.now(timezone.utc), "speaker": "Spamuel", "text": "Boundary input."}]), ())["editor"]
        part = review_parts(packet)[0]
        self.assertEqual(len(part.criteria), 1)
        validator = Draft202012Validator(review_schema(part))
        judgment = {"criterion_id": part.criteria[0].id, "verdict": "uncertain", "reason": "Schema boundary."}
        self.assertTrue(validator.is_valid({"judgments": [judgment], "summary": "Schema check."}))
        self.assertFalse(validator.is_valid({"judgments": [], "summary": "Schema check."}))
        self.assertFalse(validator.is_valid({"judgments": [judgment, judgment], "summary": "Schema check."}))
        self.assertFalse(validator.is_valid({"judgments": [{**judgment, "criterion_id": "not_supplied"}], "summary": "Schema check."}))

    def test_wire_schema_keeps_uncertainty_without_an_invented_quote(self):
        validator = Draft202012Validator(review_schema())
        self.assertTrue(validator.is_valid({"judgments": [{"criterion_id": "voice", "verdict": "uncertain",
            "reason": "No positive evidence is established."}], "summary": "Schema check only."}))

    def test_provider_usage_cannot_override_the_input_guard(self):
        validate_provider_budget({"prompt_tokens": MAX_TEST_INPUT_TOKENS})
        with self.assertRaisesRegex(ValueError, "cannot establish acceptance"):
            validate_provider_budget({"prompt_tokens": MAX_TEST_INPUT_TOKENS + 1})

    def test_review_source_identity_retains_context_constants(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            save_review_origin(root)
            origin = json.loads((root / "review-origin.json").read_text(encoding="utf-8"))
            self.assertIn("constants.py", origin["sources_sha256"])
            for name, digest in origin["sources_sha256"].items():
                data = (root / "review-source" / name).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)
