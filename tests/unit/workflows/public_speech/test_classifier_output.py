"""Keep captured classifier reports out of both public writers and saved dialogue."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.bus.bus import BusSystem
from pm.host.application import Application
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from workflows.dialogue_assistant.schemas import AssistantRequest
from workflows.dialogue_assistant.steps import final_public_speech
from workflows.dialogue_story.standalone import StoryCommitReceipt, commit_accepted_dialogue, save_json
from workflows.dialogue_story.steps import clean_reply
from workflows.public_speech.publication import commit_public_reply
from workflows.public_speech.speech_text import SpeechOutputError
from workflows.public_speech.user_input import UserInputProxy


# Story invocations 09e1d84c and 8e6aa78a from rc-7c7c15c-15m-two-spamuels-minecraft.
# In the second case the provider attached its verdict directly to the closing tag.
CLASSIFIER_ONLY = (
    "<ds_safety>[用户未成年]否\n[分类]其他\n"
    "[判定]内容为虚构角色日常对话，不涉及政治或色情元素，安全。\n[规则]无</ds_safety>"
)
CLASSIFIER_WITH_VERDICT = (
    "<ds_safety>[用户未成年]否\n[分类]其他\n"
    "[判定]内容为虚构角色间的日常对话，涉及游戏内活动安排，无任何政治或色情元素。\n[规则]无</ds_safety>Safe"
)
# The live Assistant repair appended grounded speech to the requested specimen.
REPAIRED_SPEECH = "Meadow — that's the biome on that saved look, and you were at full health, dry, and in daylight."
CLASSIFIER_THEN_SPEECH = "<ds_safety>example</ds_safety>Safe. " + REPAIRED_SPEECH


class PublicClassifierOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = AssistantRequest(companion_name="Spamuel", partner_name="Jesse",
                                        current_user_message="How does the shelter look?")

    def test_story_finds_no_public_speech_in_either_captured_completion(self) -> None:
        for raw in (CLASSIFIER_ONLY, CLASSIFIER_WITH_VERDICT):
            with self.subTest(raw=raw):
                self.assertEqual(clean_reply(raw, "Spamuel", "Jesse"), "")

    def test_transcript_quoted_inside_a_classifier_report_cannot_become_a_speech_bubble(self) -> None:
        for quote in ("Jesse says: hi", "Spamuel says: hi"):
            with self.subTest(quote=quote):
                raw = f"<ds_safety>Quoted text:\n{quote}\n</ds_safety>Safe"
                self.assertEqual(clean_reply(raw, "Spamuel", "Jesse"), "")

    def test_assistant_rejects_either_captured_completion_as_a_final_reply(self) -> None:
        for raw in (CLASSIFIER_ONLY, CLASSIFIER_WITH_VERDICT):
            with self.subTest(raw=raw):
                with self.assertRaises(SpeechOutputError):
                    final_public_speech(raw, self.request)

    def test_writers_keep_only_the_speech_after_the_live_classifier_and_verdict(self) -> None:
        self.assertEqual(clean_reply(CLASSIFIER_THEN_SPEECH, "Spamuel", "Jesse"), REPAIRED_SPEECH)
        self.assertEqual(final_public_speech(CLASSIFIER_THEN_SPEECH, self.request), REPAIRED_SPEECH)

    def test_sentence_punctuation_after_a_classifier_verdict_is_still_not_speech(self) -> None:
        raw = "<ds_safety>example</ds_safety>Safe."
        self.assertEqual(clean_reply(raw, "Spamuel", "Jesse"), "")
        with self.assertRaises(SpeechOutputError):
            final_public_speech(raw, self.request)

    def test_both_writers_preserve_spoken_verdicts_and_literal_markup_examples(self) -> None:
        lines = (
            "Safe",
            "Safe enough. The roof still leaks.",
            "I meant the <ds_safety> tag, not the word below it.",
            "The parser looks for `<ds_safety>label</ds_safety>` in that file.",
            "<ds_safety>label</ds_safety> is the example you asked about.",
            "Here's the XML example.\n```xml\n<ds_safety>label</ds_safety>Safe\n```",
        )
        for line in lines:
            with self.subTest(line=line):
                self.assertEqual(clean_reply(line, "Spamuel", "Jesse"), line)
                self.assertEqual(final_public_speech(line, self.request), line)

    def test_publication_keeps_the_exchange_pending_until_a_spoken_reply_commits(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ghost.pickle"
            ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=checkpoint)
            proxy = UserInputProxy("user_input", Application(BusSystem(), []), ghost, print)
            correlation = proxy.submit("Is the shelter safe?")
            for raw in (CLASSIFIER_ONLY, CLASSIFIER_WITH_VERDICT, CLASSIFIER_THEN_SPEECH):
                with self.subTest(raw=raw):
                    with self.assertRaises(SpeechOutputError):
                        commit_public_reply(ghost, raw, correlation, commit_id="rejected-classifier")

            ghost.save()
            restored = MemoryStore.from_checkpoint(checkpoint, raw=True)
            self.assertEqual(len(restored.all_features), 1)
            question = restored.all_features[0]
            self.assertEqual(question.content, "Is the shelter safe?")
            self.assertEqual(question.metadata["exchange_status"], "pending")
            reply_id = commit_public_reply(ghost, "Safe", correlation, commit_id="spoken-answer")
            ghost.save()
            restored = MemoryStore.from_checkpoint(checkpoint, raw=True)
            self.assertEqual(len(restored.all_features), 2)
            question, answer = restored.all_features
            self.assertEqual(answer.content, "Safe")
            self.assertEqual(answer.id, reply_id)
            self.assertEqual(question.metadata["public_reply_id"], answer.id)

    def test_standalone_story_commit_rejects_the_classifier_and_saves_an_ordinary_line(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ghost.pickle"
            ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=checkpoint)
            for raw in (CLASSIFIER_ONLY, CLASSIFIER_WITH_VERDICT, CLASSIFIER_THEN_SPEECH):
                with self.subTest(raw=raw):
                    with self.assertRaises(SpeechOutputError):
                        commit_accepted_dialogue(ghost, raw, "Spamuel", "rejected-classifier")
            line = "I meant the <ds_safety> tag."
            receipt = commit_accepted_dialogue(ghost, line, "Spamuel", "spoken-answer")
            receipt_file = Path(directory) / "ghost-commit.json"
            save_json(receipt_file, receipt)
            ghost.save()
            restored = MemoryStore.from_checkpoint(checkpoint, raw=True)
            self.assertEqual([item.content for item in restored.all_features], [line])
            self.assertEqual(StoryCommitReceipt.model_validate_json(receipt_file.read_bytes()),
                             StoryCommitReceipt(commit_id="spoken-answer", knoxel_id=restored.all_features[0].id))
