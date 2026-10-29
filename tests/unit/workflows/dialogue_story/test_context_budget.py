"""Fit real Story prompts without a model or substitute workflow collaborators."""
from __future__ import annotations

import unittest

from pm.context_render.budget import RequiredContextTooLarge, TokenEstimator
from pm.context_render.turns import Role, Turn
from workflows.dialogue_story.context import select_story_prompt
from workflows.dialogue_story.schemas import StoryRequest
from workflows.dialogue_story.steps import build_story_messages, load_prompts
from workflows.public_speech.public_context import DialogueTurn, EvidenceItem, PublicContext
from workflows.public_speech.speech_review import build_review_messages, load_prompts as load_review_prompts


class StoryContextBudgetTests(unittest.TestCase):
    def request(self, evidence: list[EvidenceItem]) -> StoryRequest:
        return StoryRequest(
            companion_name="Spamuel", partner_name="Jesse", character_card_story="Brash and theatrical.",
            current_user_message="I found a cat and I am keeping her.", current_message_id=20,
            prior_dialogue=[DialogueTurn(speaker="Jesse", text="This earlier line must remain.", knoxel_id=19)],
            workspace="The current cat adoption decision.",
            public_context=PublicContext(companion_name="Spamuel", user_name="Jesse", partner_name="Jesse",
                                         shard_story="The accepted shard state.", evidence=evidence),
        )

    def test_optional_memory_competes_after_required_story_context(self) -> None:
        required = EvidenceItem(knoxel_id=1, text="A required shard observation.", lane="shard:Jesse")
        strong = EvidenceItem(knoxel_id=2, text="Relevant cat history. " * 30, lane="facts", score=0.9)
        weak = EvidenceItem(knoxel_id=3, text="Unrelated background. " * 2000, lane="facts", score=0.1)
        request = self.request([required, strong, weak])
        prompts = load_prompts()
        baseline = self.request([required])
        request.input_budget_tokens = TokenEstimator().estimate(build_story_messages(baseline, prompts)) + 500
        fitted = select_story_prompt(request, prompts)
        assert request.public_context is not None
        assert fitted.request.public_context is not None
        self.assertEqual([item.knoxel_id for item in fitted.request.public_context.evidence], [1, 2])
        self.assertIn("memory:3", fitted.selection.omitted)
        text = "\n".join(turn.content.text for turn in fitted.selection.turns)
        self.assertIn(request.current_user_message, text)
        self.assertIn("This earlier line must remain.", text)
        self.assertIn("The accepted shard state.", text)
        self.assertIn("Omitted optional context groups: memory:3", text)
        self.assertLessEqual(fitted.selection.estimated_tokens, request.input_budget_tokens)
        self.assertEqual(len(request.public_context.evidence), 3)

    def test_required_overflow_remains_an_error(self) -> None:
        request = self.request([])
        request.input_budget_tokens = 10
        with self.assertRaises(RequiredContextTooLarge):
            select_story_prompt(request, load_prompts())

    def test_repair_feedback_is_required_and_can_displace_optional_memory(self) -> None:
        optional = EvidenceItem(knoxel_id=2, text="Optional memory. " * 300, lane="facts", score=0.9)
        request = self.request([optional])
        prompts = load_prompts()
        request.input_budget_tokens = TokenEstimator().estimate(build_story_messages(request, prompts)) + 50
        initial = select_story_prompt(request, prompts)
        self.assertIn("memory:2", initial.selection.included)
        repair = Turn(Role.User, "Correct the unsupported claim. " * 150)
        fitted = select_story_prompt(request, prompts, [repair])
        self.assertIn("memory:2", fitted.selection.omitted)
        self.assertEqual(fitted.selection.turns[-1], repair)
        self.assertLessEqual(fitted.selection.estimated_tokens, request.input_budget_tokens)

    def test_selected_record_keeps_its_supplied_source(self) -> None:
        source = EvidenceItem(knoxel_id=1, text="Original reported cat event.", lane="facts", score=0.0)
        derived = EvidenceItem(knoxel_id=2, text="Derived cat fact.", source_ids=[1], lane="facts", score=1.0)
        request = self.request([source, derived])
        request.input_budget_tokens = 20000
        fitted = select_story_prompt(request, load_prompts())
        assert fitted.request.public_context is not None
        self.assertEqual([item.knoxel_id for item in fitted.request.public_context.evidence], [1, 2])
        self.assertIn(1, fitted.selection.source_ids)

    def test_review_layout_keeps_the_complete_draft_and_required_context(self) -> None:
        required = EvidenceItem(knoxel_id=1, text="Required observation.", lane="shard:Jesse")
        optional = EvidenceItem(knoxel_id=2, text="Optional history. " * 1000, lane="facts", score=0.1)
        request = self.request([required, optional])
        draft = "The full candidate reply must reach the reviewer."
        review_prompts = load_review_prompts()

        def render(selected: StoryRequest) -> list[Turn]:
            assert selected.public_context is not None
            return build_review_messages(draft, selected.public_context, review_prompts)

        request.input_budget_tokens = TokenEstimator().estimate(render(self.request([required]))) + 300
        fitted = select_story_prompt(request, load_prompts(), render=render)
        text = "\n".join(turn.content.text for turn in fitted.selection.turns)
        self.assertIn(draft, text)
        self.assertIn(required.text, text)
        self.assertIn("memory:2", fitted.selection.omitted)
        self.assertNotIn(optional.text, text)
        self.assertLessEqual(fitted.selection.estimated_tokens, request.input_budget_tokens)
