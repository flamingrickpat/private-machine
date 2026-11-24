"""Fit real Story prompts without a model or substitute workflow collaborators."""
from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from pm.memory.temporal_context import TemporalHistoryInput, TemporalHistoryItem
from pm.records.base import RecordScope
from pm.records.enums import EntityClass, FeatureType

from pm.context_render.budget import RequiredContextTooLarge, TokenEstimator
from pm.context_render.prompt import RenderSettings
from pm.context_render.turns import Role, Turn
from workflows.dialogue_story.context import StoryRenderInput, render_story, select_story_prompt
from workflows.dialogue_story.schemas import StoryRequest
from workflows.dialogue_story.steps import build_story_messages, load_prompts
from workflows.public_speech.public_context import DialogueTurn, EvidenceItem, PublicContext, ToolEvidence
from workflows.public_speech.situational_guidance import EntitySituationEvidence, SituationSource, SituationalGuidance
from workflows.public_speech.speech_review import (
    SpeechReviewRenderInput, build_review_messages, load_prompts as load_review_prompts, render_speech_review,
)
from workflows.public_speech.task_receipts import TaskReceipt


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

    def test_trimmed_pins_leave_the_review_contract_at_supported_windows(self) -> None:
        large_pin = EvidenceItem(knoxel_id=91, text="older stone observation " * 22000, lane="facts")
        small_pin = EvidenceItem(knoxel_id=92, text="The block Jesse requested was blue.", lane="facts")
        story_request = self.request([large_pin, small_pin])
        story_request.required_evidence_ids = (91, 92)
        prompts = load_prompts()
        for window in (32000, 86000):
            with self.subTest(window=window):
                settings = RenderSettings(context_window=window, input_fraction=0.8, reserved_output_tokens=8192)
                packet = StoryRenderInput(story_request, prompts, problem="Return spoken words only.")
                prompt = render_story(packet, settings)
                self.assertEqual(prompt.required_source_ids, (92,))
                self.assertIn(20, prompt.visible_source_ids)
                self.assertNotIn(91, prompt.visible_source_ids)
                self.assertLessEqual(prompt.estimated_tokens, prompt.input_ceiling)
                self.assertEqual(render_story(packet, settings), prompt)
                self.assertEqual(story_request.required_evidence_ids, (91, 92))

    def test_required_overflow_remains_an_error(self) -> None:
        request = self.request([])
        request.input_budget_tokens = 10
        with self.assertRaises(RequiredContextTooLarge):
            select_story_prompt(request, load_prompts())

    def test_oversized_pins_compete_with_other_evidence_without_losing_current_facts(self) -> None:
        long_history = EvidenceItem(knoxel_id=1, text="Long unrelated history. " * 7000, lane="facts", score=0.1)
        source = EvidenceItem(knoxel_id=2, text="Jesse reported the cat is called Juniper.", lane="facts", score=0.2)
        related = EvidenceItem(knoxel_id=3, text="The cat's name is Juniper.", source_ids=[2], lane="facts", score=1.0)
        request = self.request([long_history, source, related])
        request.current_user_message = "I found Juniper in the garden and I am keeping her."
        request.input_budget_tokens = 22784
        request.required_evidence_ids = (1, 3)
        request.prior_dialogue.append(DialogueTurn(speaker="Jesse", text="Juniper likes hiding.", knoxel_id=21))
        assert request.public_context is not None
        request.public_context.task_receipt = TaskReceipt(workflow_name="minecraft", state="completed",
            summary="Inventory was checked and contained eight oak logs.", task_verified=True)

        fitted = select_story_prompt(request, load_prompts())

        text = "\n".join(turn.content.text for turn in fitted.selection.turns)
        self.assertIn("story evidence pins relaxed", fitted.selection.included)
        self.assertIn(request.current_user_message, text)
        self.assertIn("Juniper likes hiding.", text)
        self.assertIn("eight oak logs", text)
        self.assertNotIn("Long unrelated history.", text)
        self.assertTrue({2, 3, 20, 21}.issubset(fitted.selection.source_ids))
        self.assertLessEqual(fitted.selection.estimated_tokens, request.input_budget_tokens)
        self.assertEqual(request.required_evidence_ids, (1, 3))

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

    def test_review_fits_an_oversize_situation_without_losing_task_evidence(self) -> None:
        moment = datetime(2026, 6, 8, 8, 0)
        request = "Tell me how many oak logs you found."
        context = PublicContext(companion_name="Spamuel", user_name="Jesse", partner_name="Jesse",
                                selected_action="Spamuel chose to report the result.")
        source = SituationSource(record_id=17, revision=1, occurred_at_utc=moment, source="Jesse",
                                 kind="dialogue", text=request, scope=RecordScope())
        context.situational_guidance = SituationalGuidance(
            as_of_utc=moment, companion_name="Spamuel", partner=EntitySituationEvidence(
                entity_id=2, revision=1, name="Jesse", entity_class=EntityClass.Human,
                recent_dialogue=[source], context=[source.model_copy(update={
                    "text": "Oversized advisory partner context. " * 4500,
                })]))
        draft = "There are eight oak logs in my inventory."
        receipt = ToolEvidence(name="inventory", executed=True, result_text="oak_log: 8", source="Minecraft")
        prompts = load_review_prompts()
        settings = RenderSettings(context_window=32000, input_fraction=0.85, reserved_output_tokens=8192)
        full_prompt = build_review_messages(draft, context, prompts, receipts=(receipt,))
        self.assertGreater(TokenEstimator().estimate(full_prompt), settings.input_ceiling)

        fitted = render_speech_review(SpeechReviewRenderInput(draft, context, prompts, receipts=(receipt,)), settings)

        self.assertLessEqual(fitted.estimated_tokens, settings.input_ceiling)
        self.assertIn(draft, fitted.raw_text)
        self.assertIn(f'Jesse said: "{request}"', fitted.raw_text)
        self.assertIn("inventory succeeded: oak_log: 8", fitted.raw_text)
        self.assertIn(context.selected_action, fitted.raw_text)
        self.assertNotIn("Oversized advisory partner context", fitted.raw_text)
        self.assertIsNotNone(context.situational_guidance)

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

    def test_summary_can_cover_a_verbatim_required_event(self) -> None:
        request = self.request([])
        prompts = load_prompts()
        moment = datetime(2026, 1, 1, tzinfo=UTC)
        pinned = TemporalHistoryItem(record_id=51, started_at=moment, ended_at=moment, kind="feature",
            level=0, content="My promise to Jesse.", source="Spamuel", feature_type=FeatureType.Dialogue,
            relevance=1.0, token_cost=20)
        later = pinned.model_copy(update={"record_id": 52, "started_at": moment + timedelta(days=1),
            "ended_at": moment + timedelta(days=1), "content": "A much longer background memory. " * 500})
        summary = TemporalHistoryItem(record_id=53, started_at=moment, ended_at=later.ended_at, kind="summary",
            level=1, content="A promise and its aftermath.", source="history", feature_type=None,
            relevance=0.5, token_cost=30)
        assert request.public_context is not None
        request.public_context.temporal_history = TemporalHistoryInput(items=(pinned, later, summary))
        request.required_evidence_ids = (51,)
        request.input_budget_tokens = TokenEstimator().estimate(build_story_messages(request, prompts)) + 500
        fitted = select_story_prompt(request, prompts)
        assert fitted.history is not None
        self.assertEqual(fitted.history.covered_feature_ids, (51, 52))
        self.assertIn(51, fitted.selection.source_ids)
        self.assertIn(53, fitted.selection.source_ids)
        self.assertNotIn(52, fitted.selection.source_ids)
