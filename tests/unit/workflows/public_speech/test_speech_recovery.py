"""Keep exhausted model output local while programming errors stop the app."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from pm.bus.bus import BusSystem
from pm.bus.envelopes import new_envelope
from pm.host.application import Application
from pm.llm.http_client import EmptyCompletionError, ProviderCompletionError
from pm.llm.structured import ModelFormatError
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from tests.support import Recorder
from workflows.dialogue_story.host import StoryDialogue
from workflows.dialogue_story.standalone import MANIFEST
from workflows.main_cognition.schemas import ActionRequest, Stimulus
from workflows.public_speech.publication import exchange_record
from workflows.public_speech.user_input import UserInputProxy


class SpeechRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        self.bus = BusSystem()
        self.app = Application(self.bus, [])
        self.delivered = []
        self.received = Event()

        def emit(text):
            self.delivered.append(text)
            self.received.set()

        self.proxy = UserInputProxy("user_input", self.bus, self.app.on_error, lambda: 0, self.ghost, emit)
        self.receipts = Recorder(self.bus, "action.receipt")

    def module(self, generate, review=None):
        async def accept(*args, **kwargs):
            return '{"reason":"ok","false_task_result":false}'

        module = StoryDialogue("dialogue", self.bus, self.app.on_error, lambda: 0, None,
            working_target=MemoryBashTarget(self.ghost, "read"), workflow=MANIFEST, runtime_root=self.root,
            ghost=self.ghost, tools_factory=lambda *args: {"story_complete": generate, "public_complete": review or accept})
        self.app.modules = [module, self.proxy]
        for item in self.app.modules:
            self.bus.subscribe(item.module_id, item.accepts, item.deliver)
        return module

    def enqueue(self, text):
        correlation = self.proxy.submit(text)
        source = exchange_record(self.ghost, correlation)
        request = ActionRequest(content="Reply to Jesse.", stimuli=[Stimulus(knoxel_id=source.id, text=text)])
        self.bus.put(new_envelope("test", "data", "action.request", correlation,
                                 request.model_dump(mode="json"), target="dialogue"))
        return correlation

    def test_exhausted_output_does_not_stop_the_thread_or_discard_the_next_action(self):
        calls = []

        async def generate(messages, **generation):
            calls.append(messages)
            return " \n" if len(calls) <= 2 else "I can answer this one."

        self.module(generate)
        first = self.enqueue("First message.")
        second = self.enqueue("Second message.")
        self.app.run()
        try:
            self.assertTrue(self.received.wait(5))
            self.app.raise_if_error()
            self.assertFalse(self.app.shutdown_event.is_set())
            self.assertEqual(self.delivered, ["I can answer this one."])
            self.assertEqual(exchange_record(self.ghost, first).metadata["exchange_status"], "pending")
            self.assertEqual(exchange_record(self.ghost, second).metadata["exchange_status"], "resolved")
            failed = next(event for event in self.receipts.events if event.payload["status"] == "generation_failed")
            evidence = json.loads(Path(failed.payload["artifact_path"]).read_text(encoding="utf-8"))
            self.assertEqual(evidence["error_type"], "SpeechOutputError")
            self.assertIn("scratch/story-attempt-02.json", evidence["files"])
            self.assertEqual(len(calls), 3)
        finally:
            self.app.shutdown()
            self.app.join()

    def test_empty_transport_and_exhausted_review_stay_local(self):
        for error in (EmptyCompletionError("empty"), ModelFormatError("review context exhausted"),
                      ProviderCompletionError("provider generation failed")):
            with self.subTest(error=type(error).__name__):
                async def generate(*args, **kwargs):
                    raise error

                module = self.module(generate)
                correlation = self.enqueue("Keep this exchange pending.")
                self.assertTrue(module.step(block=False))
                self.assertFalse(self.app.shutdown_event.is_set())
                self.assertEqual(exchange_record(self.ghost, correlation).metadata["exchange_status"], "pending")

    def test_programming_and_filesystem_errors_still_stop_the_app(self):
        for error in (TypeError("completion bug"), KeyError("missing internal key"),
                      ValueError("bad internal state"), OSError("disk failed")):
            with self.subTest(error=type(error).__name__):
                async def generate(*args, **kwargs):
                    raise error

                module = self.module(generate)
                self.enqueue("This must not hide a bug.")
                module.run()
                try:
                    self.assertTrue(self.app.shutdown_event.wait(5))
                    with self.assertRaises(type(error)) as caught:
                        self.app.raise_if_error()
                    self.assertIs(caught.exception, error)
                    self.assertEqual(self.delivered, [])
                finally:
                    self.app.shutdown()
                    module.join()
                self.app = Application(self.bus, [])
                self.proxy.stop_event.clear()
