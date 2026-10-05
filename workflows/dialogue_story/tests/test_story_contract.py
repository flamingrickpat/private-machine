"""Direct Story dialogue tests.

No model endpoint, module thread, or bus is used. The kernel test uses the real
kernel and the real Python role with a deterministic tool function. The
message-role contract is tested before any prose quality test.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path

from pm_core.ghost.service import Ghost
from pm_workflows.runtime import WorkflowRuntime
from runtime.workflow_runner import WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.dialogue_story import engine
from workflows.dialogue_story.context import distinctive_terms
from workflows.dialogue_story.schemas import DialogueTurn, StoryRequest
from workflows.dialogue_story.steps import (
    build_story_messages,
    check_ontology,
    clean_reply,
    load_ontology,
    load_prompts,
    render_prior_dialogue,
)
from working_target import GhostBashEnvironment


def example_request(**overrides) -> StoryRequest:
    base = dict(
        companion_name="Spamuel",
        user_name="Jesse",
        character_card_story="Spamuel is a brash, theatrical companion.",
        supported_capabilities=["I can communicate via text."],
        unsupported_capabilities=["I cannot directly act in the physical world."],
        capability_notes=["Never claim capabilities beyond the implementation."],
        temporal_context="Current causal datetime: 2026-09-29 12:00 UTC",
        prior_dialogue=[
            DialogueTurn(speaker="Jesse", text="my sink is judging me"),
            DialogueTurn(speaker="Spamuel", text="then you owe it a dish"),
        ],
        current_user_message="ok dish is done",
        feelings=["Spamuel thinks: *she finally did it*"],
        memories=["*(Fact: Jesse has a sink.)*"],
        workspace="No attention frame is recorded yet.",
    )
    base.update(overrides)
    return StoryRequest(**base)


class MessageContractTests(unittest.TestCase):
    def test_exact_message_roles_and_prefix(self):
        messages = build_story_messages(example_request(), load_prompts())
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "user", "assistant", "user", "assistant"],
        )
        self.assertEqual(messages[-1]["content"], 'Spamuel says: "')

    def test_prior_dialogue_is_one_assistant_story_with_both_speakers(self):
        messages = build_story_messages(example_request(), load_prompts())
        story = messages[2]["content"]
        self.assertIn('Jesse says: "my sink is judging me"', story)
        self.assertIn('Spamuel says: "then you owe it a dish"', story)
        # No prior turn may leak into the user messages as a separate turn.
        for message in (messages[1], messages[3]):
            self.assertNotIn('says: "my sink is judging me"', message["content"])

    def test_character_background_and_temporal_context_are_message_two(self):
        messages = build_story_messages(example_request(), load_prompts())
        first_user = messages[1]["content"]
        self.assertIn("Spamuel is a brash, theatrical companion.", first_user)
        self.assertIn("2026-09-29 12:00 UTC", first_user)
        self.assertIn("I cannot directly act in the physical world.", first_user)

    def test_current_message_is_the_fourth_message(self):
        messages = build_story_messages(example_request(), load_prompts())
        fourth = messages[3]["content"]
        self.assertIn('Jesse says: "ok dish is done"', fourth)
        self.assertIn("she finally did it", fourth)


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


class OntologyTests(unittest.TestCase):
    def setUp(self):
        self.ontology = load_ontology()

    def test_unsupported_visual_claim_is_rejected(self):
        verdict = check_ontology(
            "I can see you through the camera, Jesse.", self.ontology, example_request()
        )
        self.assertFalse(verdict.accepted)
        self.assertEqual([violation.rule_id for violation in verdict.violations], ["visual_perception"])

    def test_unsupported_physical_action_is_rejected(self):
        verdict = check_ontology(
            "I walked over and opened the door for you.", self.ontology, example_request()
        )
        self.assertFalse(verdict.accepted)
        self.assertEqual([violation.rule_id for violation in verdict.violations], ["physical_action"])

    def test_established_voice_and_rudeness_are_accepted(self):
        for candidate in (
            "You're pathetic. Bed. Now, grease stain.",
            "Mockery is a love language and you are fluent in silence.",
            "The soup lost, Jesse. Sit down before the sink wins again.",
        ):
            with self.subTest(candidate=candidate):
                self.assertTrue(check_ontology(candidate, self.ontology, example_request()).accepted)


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
            captured["roles"] = [message["role"] for message in messages]
            captured["prefix"] = messages[-1]["content"]
            return 'Spamuel says: "Pathetic. Bed."\nJesse says: "rude"'

        def fake_complete(messages, **generation):
            return story_complete(messages, **generation)

        environment = GhostBashEnvironment({}, {}, "read")
        runtime = WorkflowRuntime(
            environment=environment, stop_event=threading.Event(),
            tools={"story_complete": fake_complete, "public_complete": fake_complete}, propagate_python_errors=True,
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
        self.assertEqual(captured["roles"], ["system", "user", "assistant", "user", "assistant"])
        self.assertEqual(captured["prefix"], 'Spamuel says: "')
        self.assertEqual(report.candidate, "Pathetic. Bed.")
        self.assertTrue(report.accepted)

    def test_kernel_report_rejects_an_unsupported_claim(self):
        async def story_complete(messages, **generation):
            return "I can see you through the camera, Jesse."

        environment = GhostBashEnvironment({}, {}, "read")
        runtime = WorkflowRuntime(
            environment=environment, stop_event=threading.Event(),
            tools={"story_complete": story_complete, "public_complete": story_complete}, propagate_python_errors=True,
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
        self.assertFalse(report.accepted)
        self.assertEqual(report.ontology.violations[0].rule_id, "visual_perception")


if __name__ == "__main__":
    unittest.main()
