"""Agentic review parsing and the single-generation Story path.

No model endpoint. A deterministic completion stands in for the reviewer and
the generator. These tests fix the A09R rules: one review returns a judgment
and a reason, advisory findings never block, and only a false task result
blocks. They do not show that a real model finds real hallucinations; the
live speech check does that.
"""
from __future__ import annotations

import asyncio
import json
import unittest

from workflows.dialogue_story.schemas import DialogueTurn, StoryRequest
from workflows.dialogue_story.steps import generate_reviewed, load_prompts, review_context
from workflows.public_speech.public_context import PublicContext
from workflows.public_speech.speech_review import (
    build_review_messages, is_blocked, load_prompts as load_review_prompts, parse_review,
    review_speech,
)

REPLY = "Oh, fantastic. Four months of silence and you open with a guilt trip?"


class ParseReviewTests(unittest.TestCase):
    def test_minor_and_equivalent_judgments_pass_without_erasing_the_finding(self):
        for judgment in ("good_enough", "equivalent_or_better"):
            with self.subTest(judgment=judgment):
                review = parse_review({"judgment": judgment, "reason": "The discrepancy does not affect the requested outcome.",
                    "false_task_result": False, "findings": [{"kind": "minor_discrepancy", "quote": "63 logs"}]})
                self.assertFalse(is_blocked(review))
                self.assertEqual(review.judgment, judgment)
                self.assertEqual(review.findings[0].quote, "63 logs")

    def test_a_passing_label_cannot_hide_a_material_false_result(self):
        for judgment in ("accept", "good_enough", "equivalent_or_better"):
            with self.subTest(judgment=judgment), self.assertRaises(ValueError):
                parse_review({"judgment": judgment, "reason": "The diamonds do not exist.", "false_task_result": True})
        with self.assertRaises(ValueError):
            parse_review({"judgment": "almost_fine", "reason": "An unknown label.", "false_task_result": False})

    def test_findings_do_not_block_and_the_reason_is_kept(self):
        review = parse_review({
            "judgment": "accept", "reason": "in character",
            "false_task_result": False,
            "findings": [{"kind": "invented_ability", "quote": "I saw", "why": "no eyes"}],
        })
        self.assertEqual(review.judgment, "accept")
        self.assertEqual(review.reason, "in character")
        self.assertFalse(is_blocked(review))
        self.assertEqual(review.findings[0].kind, "invented_ability")

    def test_a_false_task_result_is_the_only_block(self):
        review = parse_review({
            "judgment": "block", "reason": "claimed a saved note",
            "false_task_result": True, "findings": [],
        })
        self.assertTrue(is_blocked(review))

    def test_a_string_finding_is_accepted_as_an_advisory_note(self):
        review = parse_review({"reason": "unsupported label", "false_task_result": False,
                               "findings": ["unsupported_label"]})
        self.assertFalse(is_blocked(review))
        self.assertEqual(review.findings[0].kind, "unsupported_label")

    def test_a_non_json_review_is_repeated_once_without_changing_the_reply(self):
        calls: list[str] = []

        async def complete(messages, **generation):
            calls.append(messages[1].content.text)
            return "this is not json" if len(calls) == 1 else '{"judgment": "accept", "reason": "ok", "false_task_result": false}'

        complete.measure_context = lambda *args, **kwargs: {"input_tokens": 100, "input_budget_tokens": 1000}
        context = PublicContext(companion_name="Spamuel", user_name="Jesse")
        review = asyncio.run(review_speech(complete, "hello", context,
                                           prompts=load_review_prompts(), attempts=2))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(review.reason, "ok")

    def test_string_boolean_and_missing_reason_are_invalid(self):
        for payload in ({"reason": "ok", "false_task_result": "false"},
                        {"reason": "", "false_task_result": False}, {}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_review(payload)

    def test_invalid_findings_are_format_errors(self):
        for findings in (4, "unsupported_label", {}, [4], [None]):
            with self.subTest(findings=findings), self.assertRaises(ValueError):
                parse_review({"reason": "ok", "false_task_result": False, "findings": findings})

    def test_invalid_speech_extraction_is_repaired_with_the_same_draft(self):
        calls = []
        draft = "Jesse asks an honest question.\n\nI can talk to you."

        async def complete(messages, **generation):
            calls.append(messages)
            return json.dumps({"reason": "ok", "false_task_result": False,
                "spoken_text": "Invented words." if len(calls) == 1 else "I can talk to you."})

        complete.measure_context = lambda *args, **kwargs: {"input_tokens": 100, "input_budget_tokens": 1000}
        result = asyncio.run(review_speech(complete, draft,
            PublicContext(companion_name="Spamuel", user_name="Jesse"), prompts=load_review_prompts()))
        self.assertEqual(result.spoken_text, "I can talk to you.")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][:2], calls[0])
        self.assertIn("nonempty verbatim span", calls[1][-1]["content"])


class ReviewMessageTests(unittest.TestCase):
    def test_review_input_is_plain_text_without_evidence_ids(self):
        context = PublicContext(companion_name="Spamuel", user_name="Jesse", evidence=[{
            "knoxel_id": 42, "memory_kind": "utterance", "evidence_role": "user_report",
            "timestamp": "2026-05-30T21:03:00",
            "text": "[utterance; source=Jesse; evidence=42; recorded_at=2026-05-30T21:03:00+00:00] see you tomorrow"}],
            character="Spamuel is brash.", feelings="mild guilt",
            capabilities={"configured": ["I can talk in text."], "unavailable": [], "notes": [], "human_body": False},
            ontology=[{"id": 7, "term": "guilt", "term_kind": "mood_word",
                       "definition": "Responsibility discomfort.", "usage_rule": "Use with state.",
                       "stability": 1.0},
                      {"id": 8, "term_id": 7, "active": True, "fit_score": 0.5, "explanation": "accepted guilt 0.50"}])
        messages = build_review_messages(REPLY, context, load_review_prompts())
        self.assertEqual([item.role.value for item in messages], ["system", "user"])
        self.assertIn("see you tomorrow", messages[1].content.text)
        self.assertNotIn("evidence=42", messages[1].content.text)
        self.assertIn("Spamuel is brash.", messages[1].content.text)
        self.assertIn("- guilt (mood_word)", messages[1].content.text)
        self.assertIn("guilt fits right now", messages[1].content.text)
        self.assertIn("judge it the way a person checks their own words", messages[0].content.text)

    def test_a_deception_intent_is_given_to_the_review(self):
        from workflows.public_speech.public_context import Presentation
        context = PublicContext(companion_name="Spamuel", user_name="Jesse",
                                presentation=Presentation(mode="deceive", purpose="tease",
                                                          known_fact_ids=[4]))
        messages = build_review_messages("The codeword is wrong.", context, load_review_prompts())
        self.assertIn("chose to deceive on purpose", messages[1].content.text)
        self.assertIn("Known facts kept: [4]", messages[1].content.text)


class StorySingleGenerationTests(unittest.TestCase):
    def request(self) -> StoryRequest:
        return StoryRequest(companion_name="Spamuel", partner_name="Jesse", character_card_story="Brash.",
                            prior_dialogue=[DialogueTurn(speaker="Jesse", text="hi")],
                            current_user_message="where were you?",
                            public_context=PublicContext(companion_name="Spamuel", user_name="Jesse"))

    def run_once(self, draft: str, review_payload: dict):
        review_calls = []

        async def generate(messages, **generation):
            return draft

        async def review(messages, **generation):
            review_calls.append(messages)
            return json.dumps({"false_task_result": False, **review_payload})

        attempts, chosen = generate_reviewed(self.request(), load_prompts(), generate, review,
                                             review_prompts=load_review_prompts())
        return attempts, chosen, review_calls

    def test_one_draft_generates_exactly_one_review(self):
        attempts, chosen, calls = self.run_once('I was sulking, obviously."',
                                                {"judgment": "accept", "reason": "in character"})
        self.assertEqual(len(attempts), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(chosen.candidate, "I was sulking, obviously.")
        self.assertEqual(chosen.review.reason, "in character")
        self.assertFalse(chosen.blocked)

    def test_a_transcript_echo_without_new_speech_never_reaches_review(self):
        from workflows.public_speech.speech_text import SpeechOutputError
        calls = []

        async def generate(messages, **generation):
            calls.append(messages)
            # Echo the assistant turn content (the dialogue block).
            assistant = [m for m in messages if m.role.value == "assistant"]
            return assistant[-1].content.text if assistant else ""

        async def review(messages, **generation):
            self.fail("An unchanged transcript is not a draft reply.")

        with self.assertRaisesRegex(SpeechOutputError, "no new speech"):
            generate_reviewed(self.request(), load_prompts(), generate, review)
        self.assertEqual(len(calls), 2)

    def test_empty_prefill_recovers_once_without_dropping_context(self):
        from pm.llm.http_client import EmptyCompletionError
        for first in ("", " \n", "*(20 October 2026)*", "Spamuel thinks: *quiet*", "<tool_call></tool_call>", None):
            with self.subTest(first=first):
                calls, reviews = [], []

                async def generate(messages, **generation):
                    calls.append(messages)
                    if len(calls) == 1:
                        if first is None:
                            raise EmptyCompletionError("No model text.")
                        return first
                    return "I was sulking, obviously."

                async def review(messages, **generation):
                    reviews.append(messages)
                    return '{"reason": "ok", "false_task_result": false}'

                attempts, chosen = generate_reviewed(self.request(), load_prompts(), generate, review)
                self.assertEqual(len(attempts), 2)
                self.assertTrue(attempts[0].generation_error)
                self.assertEqual(calls[1][:len(calls[0])], calls[0])
                self.assertEqual(chosen.message_roles, ["system", "user", "assistant", "user", "user"])
                self.assertEqual(chosen.candidate, "I was sulking, obviously.")
                self.assertEqual(len(reviews), 1)

    def test_logic_transport_and_recording_errors_do_not_receive_recovery(self):
        from pm.llm.http_client import ModelError, ProviderCompletionError
        for error in (TypeError("bug"), KeyError("bug"), ValueError("bug"),
                      OSError("disk error"), ModelError("HTTP 401"), ProviderCompletionError("provider failure"),
                      asyncio.CancelledError()):
            with self.subTest(error=type(error).__name__):
                calls = []

                async def generate(*args, **kwargs):
                    calls.append(1)
                    raise error

                with self.assertRaises(type(error)) as caught:
                    generate_reviewed(self.request(), load_prompts(), generate, generate)
                self.assertIs(caught.exception, error)
                self.assertEqual(len(calls), 1)

    def test_an_invented_ability_finding_does_not_block(self):
        attempts, chosen, _ = self.run_once(
            'I watched you through the camera."',
            {"judgment": "accept", "reason": "reported",
             "findings": [{"kind": "invented_ability", "why": "no eyes"}]})
        self.assertEqual(len(attempts), 1)
        self.assertFalse(chosen.blocked)

    def test_a_false_task_result_blocks(self):
        attempts, chosen, _ = self.run_once(
            'I saved the note already."',
            {"judgment": "block", "reason": "no note receipt", "false_task_result": True})
        self.assertTrue(chosen.blocked)


class NoContextReviewTests(unittest.TestCase):
    def test_standalone_adapter_supplies_measurement_for_review_repair(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch
        from pm.llm.activation import ModelActivation
        from pm.llm.config import Connection
        from workflows.dialogue_story.standalone import build_story_tools

        connection = Connection(base_url="http://127.0.0.1:1/v1", model="fixture")
        owner = ModelActivation()
        owner._ready[connection.base_url] = connection.model
        client = SimpleNamespace(connection=connection, activation=owner, complete=AsyncMock(side_effect=[
            SimpleNamespace(text="bad JSON"),
            SimpleNamespace(text='{"reason":"ok","false_task_result":false}'),
        ]))
        complete = build_story_tools(client)["public_complete"]
        with patch("pm.llm.context_window.measure_context", new_callable=AsyncMock,
                   return_value={"input_tokens": 100, "input_budget_tokens": 1000}) as measure:
            result = asyncio.run(review_speech(complete, "hello",
                PublicContext(companion_name="Spamuel", user_name="Jesse")))
        self.assertEqual(result.reason, "ok")
        self.assertEqual(client.complete.await_count, 2)
        self.assertEqual(measure.await_count, 1)
        self.assertIs(measure.call_args.args[0], connection)

    def test_request_context_keeps_abilities_and_memories(self):
        request = StoryRequest(companion_name="Spamuel", partner_name="Jesse", character_card_story="Brash.",
                               supported_capabilities=["I can talk in text."],
                               unsupported_capabilities=["I cannot see."],
                               current_user_message="hi", memories=["Jesse has a sink."])
        context = review_context(request)
        self.assertIn("I can talk in text.", context.capabilities["configured"])
        self.assertIn("Cannot: I cannot see.", context.capabilities["configured"])
        self.assertEqual(context.memories, ["Jesse has a sink."])


if __name__ == "__main__":
    unittest.main()
