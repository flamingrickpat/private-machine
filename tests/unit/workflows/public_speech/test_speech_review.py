"""Hallucination review weighting and the Story regeneration loop.

No model endpoint. A deterministic completion stands in for the reviewer and
the generator. These tests fix the policy rules: which reported violations
count, how much they weigh, and when Story generates again. They do not show
that a real model finds real hallucinations; the live speech check does that.
"""
from __future__ import annotations

import json
import unittest

from workflows.dialogue_story.schemas import DialogueTurn, StoryRequest
from workflows.dialogue_story.steps import generate_reviewed, load_prompts
from workflows.public_speech.public_context import PublicContext, memory_lines
from workflows.public_speech.speech_review import (
    build_review_messages, is_blocked, load_policy, should_regenerate, weigh,
)

REPLY = "Oh, fantastic. Four months of silence and you open with a guilt trip?"


class WeighTests(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy()

    def test_known_kinds_weigh_and_invented_kinds_are_ignored(self):
        review = weigh({"violations": [
            {"kind": "invented_event", "quote": "Four months of silence", "why": "no record"},
            {"kind": "exaggeration", "quote": "Oh, fantastic", "why": "sarcasm"},
        ]}, REPLY, self.policy)
        self.assertEqual([item.kind for item in review.violations], ["invented_event"])
        self.assertEqual([item.kind for item in review.ignored], ["exaggeration"])
        self.assertEqual(review.score, 2)
        self.assertFalse(should_regenerate(review, self.policy))

    def test_a_quote_that_is_not_in_the_reply_cannot_count(self):
        review = weigh({"violations": [{"kind": "private_leak", "quote": "system log at 10:09", "why": "x"}]},
                       REPLY, self.policy)
        self.assertEqual(review.score, 0)
        self.assertEqual(len(review.ignored), 1)

    def test_duplicates_count_once_and_heavy_kinds_trigger_regeneration(self):
        item = {"kind": "private_leak", "quote": "guilt trip", "why": "x"}
        review = weigh({"violations": [item, dict(item)]}, REPLY, self.policy)
        self.assertEqual(review.score, 3)
        self.assertTrue(should_regenerate(review, self.policy))
        self.assertFalse(is_blocked(review, self.policy))

    def test_a_false_task_result_blocks_publication(self):
        review = weigh({"violations": [{"kind": "false_task_result", "quote": "guilt trip", "why": "x"}]},
                       REPLY, self.policy)
        self.assertTrue(is_blocked(review, self.policy))


class ReviewMessageTests(unittest.TestCase):
    def test_review_input_is_plain_text_without_evidence_ids(self):
        context = PublicContext(companion_name="Spamuel", user_name="Jesse", evidence=[{
            "knoxel_id": 42, "memory_kind": "utterance", "evidence_role": "user_report", "timestamp": "2026-05-30T21:03:00",
            "text": "[utterance; source=Jesse; evidence=42; recorded_at=2026-05-30T21:03:00+00:00] see you tomorrow"}])
        messages = build_review_messages(REPLY, context, load_policy())
        self.assertEqual([item["role"] for item in messages], ["system", "user"])
        self.assertIn("30 May 2026, 21:03, Jesse said: see you tomorrow", messages[1]["content"])
        self.assertNotIn("evidence=42", messages[1]["content"])
        self.assertIn("lie, tease, exaggerate", messages[0]["content"])

    def test_internal_state_records_never_render_as_memories(self):
        context = PublicContext(companion_name="Spamuel", user_name="Jesse", evidence=[{
            "knoxel_id": 7, "memory_kind": "internal_state", "text": "core affect valence -0.19"}])
        self.assertEqual(memory_lines(context), [])


class RegenerationLoopTests(unittest.TestCase):
    def request(self) -> StoryRequest:
        return StoryRequest(companion_name="Spamuel", user_name="Jesse", character_card_story="Brash.",
                            prior_dialogue=[DialogueTurn(speaker="Jesse", text="hi")],
                            current_user_message="where were you?",
                            public_context=PublicContext(companion_name="Spamuel", user_name="Jesse"))

    def run_loop(self, drafts: list[str], reviews: list[list[dict]]):
        notes: list[str] = []

        async def generate(messages, **generation):
            notes.append(messages[1]["content"])
            return drafts.pop(0)

        async def review(messages, **generation):
            return json.dumps({"violations": reviews.pop(0)})

        attempts, chosen = generate_reviewed(self.request(), load_prompts(), generate, review)
        return attempts, chosen, notes

    def test_a_clean_first_draft_is_used_without_regeneration(self):
        attempts, chosen, _ = self.run_loop(['I was sulking, obviously."'], [[]])
        self.assertEqual(len(attempts), 1)
        self.assertEqual(chosen.candidate, "I was sulking, obviously.")

    def test_a_heavy_draft_is_regenerated_with_an_in_character_note(self):
        attempts, chosen, notes = self.run_loop(
            ['*sighs* I was offline."', 'I was sulking, obviously."'],
            [[{"kind": "private_leak", "quote": "*sighs*", "why": "stage direction"}], []])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(chosen.candidate, "I was sulking, obviously.")
        self.assertIn("*sighs*", notes[1])
        self.assertIn("stay fully in character", notes[1])

    def test_after_all_attempts_the_lightest_draft_is_chosen(self):
        heavy = [{"kind": "private_leak", "quote": "logs", "why": "x"}]
        attempts, chosen, _ = self.run_loop(
            ['my logs say so."', 'my logs, my logs."', 'the logs agree."'],
            [heavy + [{"kind": "invented_event", "quote": "say so", "why": "x"}], heavy, heavy])
        self.assertEqual(len(attempts), 3)
        self.assertEqual(chosen.candidate, "my logs, my logs.")
        self.assertFalse(chosen.blocked)


if __name__ == "__main__":
    unittest.main()
