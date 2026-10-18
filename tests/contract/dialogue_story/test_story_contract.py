"""Direct Story dialogue tests.

No model endpoint, module thread, or bus is used. The kernel test uses the real
kernel and the real Python role with a deterministic tool function. The
message-role contract is tested before any prose quality test.
"""
from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path

from pm.memory.store import MemoryStore
from pm_workflows.runtime import WorkflowRuntime
from pm.runs.workflow_runner import WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.dialogue_story import standalone as engine
from workflows.dialogue_story.context import distinctive_terms
from workflows.dialogue_story.schemas import DialogueTurn, StoryRequest
from workflows.dialogue_story.steps import (
    build_story_messages,
    clean_reply,
    load_prompts,
    render_prior_dialogue,
    review_context,
)
from workflows.public_speech.speech_review import load_prompts as load_review_prompts
from pm.memory.bash_view import MemoryBashEnvironment


def example_request(**overrides) -> StoryRequest:
    base = dict(
        companion_name="Spamuel",
        partner_name="Jesse",
        character_card_story="Spamuel is a brash, theatrical companion.",
        supported_capabilities=["I can communicate via text."],
        unsupported_capabilities=["I cannot directly act in the physical world."],
        capability_notes=["Never claim capabilities beyond the implementation."],
        now="2026-09-29T12:00:00",
        prior_dialogue=[
            DialogueTurn(speaker="Jesse", text="my sink is judging me", timestamp="2026-09-28T20:00:00"),
            DialogueTurn(speaker="Spamuel", text="then you owe it a dish", timestamp="2026-09-28T20:01:00"),
        ],
        current_user_message="ok dish is done",
        feelings="clear pride and mild tenderness",
        memories=["Jesse has a sink."],
        workspace="No attention frame is recorded yet.",
    )
    base.update(overrides)
    return StoryRequest(**base)


class MessageContractTests(unittest.TestCase):
    def test_three_roles_and_the_open_prefill_ends_the_story(self):
        messages = build_story_messages(example_request(), load_prompts())
        self.assertEqual([message.role.value for message in messages], ["system", "user", "assistant"])
        self.assertTrue(messages[-1].content.text.endswith('Spamuel says: "'))
        self.assertIn("story writer", messages[0].content.text)

    def test_verbatim_speech_lives_only_in_the_assistant_turn(self):
        messages = build_story_messages(example_request(), load_prompts())
        story = messages[2].content.text
        self.assertEqual(story.splitlines()[:4], [
            "*(28 September 2026)*",
            'Jesse says: "my sink is judging me"',
            'Spamuel says: "then you owe it a dish"',
            'Jesse says: "ok dish is done"',
        ])
        for line in ("my sink is judging me", "then you owe it a dish", "ok dish is done"):
            self.assertNotIn(line, messages[0].content.text + messages[1].content.text)

    def test_thoughts_and_day_changes_render_as_story_elements(self):
        rendered = render_prior_dialogue([
            DialogueTurn(speaker="Jesse", text="a", timestamp="2026-05-21T20:00:00"),
            DialogueTurn(speaker="Spamuel", text="he is back", kind="thought", timestamp="2026-05-21T20:01:00"),
            DialogueTurn(speaker="Spamuel", text="b", timestamp="2026-05-22T09:00:00"),
        ], load_prompts())
        self.assertEqual(rendered.splitlines(), [
            "*(21 May 2026)*", 'Jesse says: "a"', "Spamuel thinks: *he is back*",
            "*(22 May 2026)*", 'Spamuel says: "b"'])
        self.assertEqual(clean_reply('Fine."\n*(23 May 2026)*\nJesse says: "x"', "Spamuel", "Jesse"), "Fine.")

    def test_derived_context_lives_only_in_the_user_turn(self):
        messages = build_story_messages(example_request(), load_prompts())
        context = messages[1].content.text
        for text in ("Spamuel is a brash, theatrical companion.", "I cannot directly act in the physical world.",
                     "Jesse has a sink.", "clear pride and mild tenderness", "29 September 2026, 12:00",
                     "about 16 hours ago"):
            self.assertIn(text, context)
            self.assertNotIn(text, messages[2].content.text)

class CleanReplyTests(unittest.TestCase):
    def test_echoed_prefix_is_removed(self):
        self.assertEqual(clean_reply('Spamuel says: "Pathetic."', "Spamuel", "Jesse"), "Pathetic.")

    def test_invented_next_user_turn_is_cut(self):
        raw = 'Spamuel says: "Pathetic."\nJesse says: "rude"\nSpamuel says: "whatever"'
        self.assertEqual(clean_reply(raw, "Spamuel", "Jesse"), "Pathetic.")

    def test_plain_continuation_is_kept(self):
        self.assertEqual(clean_reply("Bed. Now.", "Spamuel", "Jesse"), "Bed. Now.")

    def test_trailing_unmatched_quote_is_removed(self):
        self.assertEqual(clean_reply('Pathetic. Bed."', "Spamuel", "Jesse"), "Pathetic. Bed.")


class ReviewContextTests(unittest.TestCase):
    def test_no_context_review_keeps_abilities_and_memories(self):
        context = review_context(example_request())
        self.assertIn("I can communicate via text.", context.capabilities["configured"])
        self.assertIn("Cannot: I cannot directly act in the physical world.", context.capabilities["configured"])
        self.assertIn("Jesse has a sink.", context.memories)


class ContextSelectionTests(unittest.TestCase):
    def test_distinctive_terms_drop_stopwords_and_short_words(self):
        terms = distinctive_terms("ok dish is done. i am going to sleep now. no victory speech.")
        self.assertIn("dish", terms)
        self.assertIn("sleep", terms)
        self.assertIn("victory", terms)
        self.assertNotIn("the", terms)
        self.assertNotIn("now", terms)

    def test_render_prior_dialogue_includes_both_speakers(self):
        rendered = render_prior_dialogue([
            DialogueTurn(speaker="Jesse", text="a"),
            DialogueTurn(speaker="Spamuel", text="b"),
        ])
        self.assertEqual(rendered, 'Jesse says: "a"\nSpamuel says: "b"')


class KernelContractTests(unittest.TestCase):
    def setUp(self):
        root = Path(".runtime/w4-direct-tests")
        for stale in root.rglob("*"):
            if stale.is_file():
                stale.unlink()
        self.root = root

    def test_kernel_runs_the_real_role_with_a_deterministic_tool(self):
        captured: dict = {}

        async def story_complete(messages, **generation):
            captured["roles"] = [message.role.value for message in messages]
            captured["story"] = messages[-1].content.text
            return 'Spamuel says: "Pathetic. Bed."\nJesse says: "rude"'

        async def review_complete(messages, **generation):
            return json.dumps({"judgment": "accept", "reason": "in character",
                               "false_task_result": False, "findings": []})

        def fake_complete(messages, **generation):
            return story_complete(messages, **generation)

        environment = MemoryBashEnvironment({}, {}, "read")
        runtime = WorkflowRuntime(
            environment=environment, stop_event=threading.Event(),
            tools={"story_complete": fake_complete, "public_complete": review_complete}, propagate_python_errors=True,
        )
        runner = WorkflowRunner(
            engine.MANIFEST, self.root, namespace="dialogue_story",
            base_dir=engine.PROJECT_ROOT,
        )
        result = runner.invoke(
            WorkflowRequest("direct-1", example_request().model_dump(mode="json")), runtime,
        )
        self.assertEqual(result.status, WorkflowStatus.Completed)
        report = engine.StoryReport.model_validate(result.report)
        self.assertEqual(captured["roles"], ["system", "user", "assistant"])
        self.assertTrue(captured["story"].endswith('Spamuel says: "'))
        self.assertEqual(report.candidate, "Pathetic. Bed.")
        self.assertTrue(report.accepted)
        self.assertEqual(report.review_reason, "in character")

    def test_kernel_report_reports_an_invented_ability_without_blocking(self):
        async def story_complete(messages, **generation):
            return "I can see you through the camera, Jesse."

        async def review_complete(messages, **generation):
            return json.dumps({"judgment": "accept", "reason": "reported for review",
                               "false_task_result": False,
                               "findings": [{"kind": "invented_ability", "why": "no eyes"}]})

        environment = MemoryBashEnvironment({}, {}, "read")
        runtime = WorkflowRuntime(
            environment=environment, stop_event=threading.Event(),
            tools={"story_complete": story_complete, "public_complete": review_complete}, propagate_python_errors=True,
        )
        runner = WorkflowRunner(
            engine.MANIFEST, self.root, namespace="dialogue_story",
            base_dir=engine.PROJECT_ROOT,
        )
        result = runner.invoke(
            WorkflowRequest("direct-2", example_request().model_dump(mode="json")), runtime,
        )
        self.assertEqual(result.status, WorkflowStatus.Completed)
        report = engine.StoryReport.model_validate(result.report)
        self.assertTrue(report.accepted)
        self.assertFalse(report.speech_review.false_task_result)
        self.assertEqual(report.speech_review.findings[0].kind, "invented_ability")


if __name__ == "__main__":
    unittest.main()
