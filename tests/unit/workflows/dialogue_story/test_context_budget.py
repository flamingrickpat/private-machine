"""Fit real Story prompts without a model or substitute workflow collaborators."""
from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from pm.memory.temporal_context import TemporalHistoryInput, TemporalHistoryItem
from pm.records.enums import FeatureType

from pm.context_render.budget import RequiredContextTooLarge, TokenEstimator
from pm.context_render.prompt import RenderSettings
from pm.context_render.turns import Role, Turn
from workflows.dialogue_story.context import select_story_prompt
from workflows.dialogue_story.schemas import StoryRequest
from workflows.dialogue_story.steps import build_story_messages, load_prompts
from workflows.public_speech.public_context import DialogueTurn, EvidenceItem, PublicContext
from workflows.public_speech.speech_review import SpeechReviewRenderInput, render_speech_review, load_prompts as load_review_prompts


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
        request.required_evidence_ids = (1,)
        prompts = load_prompts()
        baseline = self.request([required])
        request.input_budget_tokens = TokenEstimator().estimate(build_story_messages(baseline, prompts)) + 3000
        fitted = select_story_prompt(request, prompts)
        assert request.public_context is not None
        assert fitted.request.public_context is not None
        self.assertEqual([item.knoxel_id for item in fitted.request.public_context.evidence], [1, 2])
        self.assertIn("memory:3", fitted.selection.omitted)
        text = "\n".join(turn.content.text for turn in fitted.selection.turns)
        self.assertIn(request.current_user_message, text)
        self.assertIn("This earlier line must remain.", text)
        self.assertIn("The accepted shard state.", text)
        self.assertNotIn(weak.text, text)
        self.assertLessEqual(fitted.selection.estimated_tokens, request.input_budget_tokens)
        self.assertEqual(len(request.public_context.evidence), 3)

    def test_required_overflow_remains_an_error(self) -> None:
        request = self.request([])
        request.input_budget_tokens = 10
        with self.assertRaises(RequiredContextTooLarge):
            select_story_prompt(request, load_prompts())

    def test_repair_feedback_is_required_and_can_displace_optional_memory(self) -> None:
        optional = EvidenceItem(knoxel_id=2, text="Optional memory. " * 30, lane="facts", score=0.9)
        request = self.request([optional])
        prompts = load_prompts()
        request.input_budget_tokens = TokenEstimator().estimate(build_story_messages(request, prompts)) + 2000
        initial = select_story_prompt(request, prompts)
        self.assertIn("memory:2", initial.selection.included)
        repair = Turn(Role.User, "Correct the unsupported claim. " * 230)
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
        draft = "I remember that observation."
        review_prompts = load_review_prompts()

        assert request.public_context is not None
        settings = RenderSettings(context_window=32000, input_fraction=0.85, reserved_output_tokens=8192)
        fitted = render_speech_review(SpeechReviewRenderInput(draft, request.public_context, review_prompts), settings)
        self.assertIn(draft, fitted.raw_text)
        self.assertIn(required.text, fitted.raw_text)
        self.assertNotIn(optional.text, fitted.raw_text)
        self.assertLessEqual(fitted.estimated_tokens, settings.input_ceiling)

    def test_required_older_dialogue_and_missing_record(self) -> None:
        request = self.request([])
        request.input_budget_tokens = 20000
        request.prior_dialogue[:0] = [DialogueTurn(speaker="Jesse", text="The original promise.", knoxel_id=11),
                                      DialogueTurn(speaker="Spamuel", text="I remember.", knoxel_id=12)]
        request.required_evidence_ids = (11,)
        fitted = select_story_prompt(request, load_prompts())
        self.assertIn(11, fitted.selection.source_ids)
        self.assertIn("The original promise.", "\n".join(turn.content.text for turn in fitted.selection.turns))
        request.required_evidence_ids = (404,)
        with self.assertRaisesRegex(ValueError, "404"):
            select_story_prompt(request, load_prompts())

    def test_fact_survives_competing_lifetime_history(self) -> None:
        fact = EvidenceItem(knoxel_id=1, text="Jesse already named the cat Juniper.", lane="facts", score=1.0)
        request = self.request([fact])
        prompts = load_prompts()
        request.input_budget_tokens = TokenEstimator().estimate(build_story_messages(request, prompts)) + 900
        origin = datetime(2026, 1, 1, tzinfo=UTC)
        history: list[TemporalHistoryItem] = []
        for day in range(12):
            moment = origin + timedelta(days=day)
            history.append(TemporalHistoryItem(record_id=100 + day, started_at=moment, ended_at=moment,
                kind="feature", level=0, content=f"Day {day}: " + "a remembered conversation " * 25,
                source="Jesse", feature_type=FeatureType.Dialogue, relevance=0.5, token_cost=90))
        assert request.public_context is not None
        request.public_context.temporal_history = TemporalHistoryInput(items=tuple(history))
        fitted = select_story_prompt(request, prompts)
        self.assertIn(fact.knoxel_id, fitted.selection.source_ids)
        self.assertIn(fact.text, "\n".join(turn.content.text for turn in fitted.selection.turns))
        assert fitted.history is not None
        self.assertTrue(fitted.history.items)
        self.assertTrue(fitted.history.omitted_feature_ids)
        self.assertLessEqual(fitted.selection.estimated_tokens, request.input_budget_tokens)
