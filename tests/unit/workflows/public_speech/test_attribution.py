"""Keep mixed stimuli and overlapping speech attributed to their real sources."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.bus.bus import BusSystem
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ActionType, FeatureType, SourceKind
from pm.records.memory_records import new_feature
from workflows.dialogue_story.host import StoryDialogue
from workflows.dialogue_story.schemas import StoryRequest
from workflows.dialogue_story.steps import build_story_messages, load_prompts
from workflows.public_speech.public_context import build_public_context
from workflows.public_speech.publication import commit_public_reply, exchange_record, acknowledge_reply
from workflows.public_speech.user_input import UserInputProxy
from workflows.public_speech.speech_review import SpeechReview, select_spoken_text
from workflows.public_speech.speech_text import strip_narrator_setup


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        self.bus = BusSystem()
        self.proxy = UserInputProxy("user_input", self.bus, self.fail, lambda: 0, self.ghost, print)

    def test_private_and_lifecycle_sources_never_become_current_user_speech(self):
        from pm.memory.bash_view import MemoryBashTarget
        from workflows.dialogue_story.standalone import MANIFEST
        first = self.proxy.submit("hey. I'm back.")
        user = exchange_record(self.ghost, first)
        thought = new_feature(self.ghost, "I notice the same worry returning: was Jesse's line a boundary?",
                              FeatureType.InnerMonologue, source="Spamuel", source_kind=SourceKind.Self)
        lifecycle = new_feature(self.ghost, "The previous stop was not clean.", FeatureType.SystemMessage,
                                source="system", source_kind=SourceKind.System)
        payload = {"stimuli": [{"knoxel_id": item.id, "text": str(item.content)}
                               for item in (user, thought, lifecycle)]}
        with TemporaryDirectory() as tmp:
            module = StoryDialogue("dialogue_story", self.bus, self.fail, lambda: 0, None,
                working_target=MemoryBashTarget(self.ghost, "read"), workflow=MANIFEST,
                tools_factory=lambda environment, batch: {}, runtime_root=Path(tmp), ghost=self.ghost)
            self.assertEqual([item["knoxel_id"] for item in module.user_stimuli(payload)], [user.id])
            self.assertEqual(module.user_stimuli({"stimuli": payload["stimuli"][1:]}), [])
        request = StoryRequest(companion_name="Spamuel", user_name="Jesse", character_card_story="",
                               current_user_message="", selected_action="Reach out about the unfinished story.")
        transcript = build_story_messages(request, load_prompts())[-1].content.text
        self.assertNotIn("Jesse says:", transcript)
        self.assertTrue(transcript.endswith('Spamuel says: "'))

    def test_delivery_bookkeeping_does_not_invalidate_another_draft(self):
        correlation = self.proxy.submit("first message")
        source = exchange_record(self.ghost, correlation)
        context = build_public_context(self.ghost, {"stimuli": [{"knoxel_id": source.id}]})
        commit_public_reply(self.ghost, "first answer", correlation, commit_id="first")
        # Another independent action can retain this source after resolution.
        reply_id = commit_public_reply(self.ghost, "a follow-up", "", commit_id="follow-up", context=context)
        context2 = build_public_context(self.ghost, {"stimuli": [{"knoxel_id": reply_id}]})
        acknowledge_reply(self.ghost, reply_id)
        self.proxy.submit("second message arrives while composing")
        new_id = commit_public_reply(self.ghost, "another follow-up", "", commit_id="last", context=context2)
        self.assertEqual(self.ghost.get_knoxel_by_id(new_id).content, "another follow-up")

    def test_narrator_extraction_cannot_rewrite_speech(self):
        draft = "Jesse asks the worst kind of question: an honest one.\n\nRight now I can actually talk to you."
        speech = "Right now I can actually talk to you."
        self.assertEqual(select_spoken_text(draft, SpeechReview(spoken_text=speech)), speech)
        self.assertEqual(strip_narrator_setup(draft, "Jesse"), speech)
        joke = "Jesse asks the worst kind of question, so I charge extra for honesty."
        self.assertEqual(strip_narrator_setup(joke, "Jesse"), joke)
        with self.assertRaises(ValueError):
            select_spoken_text(draft, SpeechReview(spoken_text="I checked Minecraft and it works."))
        with self.assertRaises(ValueError):
            select_spoken_text(draft, SpeechReview(spoken_text=""))

    def test_two_actions_can_follow_up_without_duplicate_replay(self):
        from pm.records.memory_records import ActionSchemaRecord
        correlation = self.proxy.submit("tell me what happened")
        batch = self.ghost.begin_batch("two-actions")
        first = batch.create(ActionSchemaRecord(content="Explain the first part.", action_type=ActionType.Reply))
        second = batch.create(ActionSchemaRecord(content="Follow up with the second part.", action_type=ActionType.Reply))
        self.ghost.commit(batch)
        one = commit_public_reply(self.ghost, "part one", correlation, commit_id="one", action_id=first.id)
        two = commit_public_reply(self.ghost, "part two", correlation, commit_id="two", action_id=second.id)
        self.assertNotEqual(one, two)
        self.assertIsNone(commit_public_reply(self.ghost, "duplicate", correlation, commit_id="again", action_id=first.id))

    def test_transport_classification_does_not_hide_logic_errors(self):
        from workflows.minecraft_body.capabilities import _transport_failure
        self.assertTrue(_transport_failure(ConnectionRefusedError("offline")))
        self.assertFalse(_transport_failure(TypeError("broken local logic")))
        self.assertFalse(_transport_failure(ExceptionGroup("mixed", [OSError("offline"), KeyError("missing field")])))

    def test_cycle_retry_reuses_the_same_workflow_event(self):
        from workflows.main_cognition.host import MainCogLoop
        from pm.bus.envelopes import new_envelope
        loop = MainCogLoop.__new__(MainCogLoop)
        loop.ghost = self.ghost
        loop._event_stimuli = {}
        event = new_envelope("worker", "telemetry", "workflow.event", "request",
                             {"run_id": "observed-run", "event": "completed"})
        first = loop._stimulus_from(event)
        second = loop._stimulus_from(event)
        self.assertEqual(first, second)
        self.assertEqual(len(self.ghost.all_features), 1)

    def test_broken_public_input_stops_the_application_without_speech(self):
        from pm.host.application import Application
        from pm.memory.bash_view import MemoryBashTarget
        from pm.bus.envelopes import new_envelope
        from workflows.dialogue_story.standalone import MANIFEST
        app = Application(self.bus, [])
        with TemporaryDirectory() as tmp:
            module = StoryDialogue("dialogue_story", self.bus, app.on_error, lambda: 0, None,
                working_target=MemoryBashTarget(self.ghost, "read"), workflow=MANIFEST,
                tools_factory=lambda environment, batch: {}, runtime_root=Path(tmp), ghost=self.ghost)
            app.modules = [module]
            self.bus.subscribe(module.module_id, module.accepts, module.deliver)
            self.bus.put(new_envelope("main_cog", "data", "action.request", "broken",
                {"stimuli": [{"knoxel_id": 999, "text": "not an accepted source"}]}, target="dialogue_story"))
            app.run()
            try:
                self.assertTrue(app.shutdown_event.wait(2))
                with self.assertRaisesRegex(KeyError, "does not exist"):
                    app.raise_if_error()
                self.assertEqual(self.ghost.all_features, [])
            finally:
                app.shutdown()
                app.join()
