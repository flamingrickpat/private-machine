"""Direct debugger tests for checkpoint-before-publication ordering."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from workflows.companion import build_app
from pm.memory.store import MemoryStore
from pm.memory.snapshot import GhostCancelled
from tests.support import ManualClock, Recorder
from tests.unit.pm.memory.test_store_commits import small_ghost


class DurableApplicationTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "ghost.json"
        self.replies = []
        self.app = build_app(None, self.root / "runtime", ghost=small_ghost(self.path),
                             clock=ManualClock(), emit_reply=self.replies.append)
        self.events = Recorder(self.app.bus)

    def test_direct_turn_is_durable_before_each_bus_event(self):
        observed = []

        def receive(envelope):
            reopened = MemoryStore.from_checkpoint(self.path)
            if "knoxel_id" in envelope.payload:
                self.assertIsNotNone(reopened.get_knoxel_by_id(envelope.payload["knoxel_id"]))
            if "gwt_tick" in envelope.payload:
                self.assertIn(envelope.payload["gwt_tick"], reopened.gwt_turns)
            observed.append(envelope.topic)

        self.app.bus.subscribe("disk-reader", lambda envelope: True, receive)
        correlation = self.app.user_input.submit("hello")
        self.assertTrue(self.app.main_cog.step())
        self.assertTrue(self.app.dialogue.step())
        self.assertTrue(self.app.user_input.step())
        self.assertEqual(self.replies, ["I heard you, Jesse."])
        self.assertEqual(observed, ["user.input", "gwt.updated", "action.dialogue", "dialogue.reply"])
        related = [event for event in self.events.events if event.correlation_id == correlation]
        self.assertEqual(related[1].causation_id, related[0].id)
        self.assertEqual(related[2].causation_id, related[1].id)
        self.assertFalse(self.app.main_cog.step(), "own reply started another cognitive turn")
        self.assertTrue(all(module._thread is None for module in self.app.modules))
        restarted = build_app(None, self.root / "restarted", checkpoint_path=self.path,
                              record_lifecycle=False)
        self.assertEqual(restarted.ghost.state_dict(), self.app.ghost.state_dict())

    def test_failed_save_emits_no_gwt_or_action(self):
        self.app.user_input.submit("hello")
        before = self.path.read_bytes()
        blocked = self.root / "blocked"
        blocked.mkdir()
        (blocked / "keep").write_text("keep", encoding="utf-8")
        self.app.ghost.checkpoint = blocked
        with self.assertRaises(OSError):
            self.app.main_cog.step()
        self.assertEqual([event.topic for event in self.events.events], ["user.input"])
        self.assertEqual(self.app.ghost.gwt_turns, {})
        self.assertEqual(self.path.read_bytes(), before)

    def test_repeated_publish_never_emits_uncommitted_ids(self):
        self.app.user_input.submit("hello")
        inputs = list(self.events.events)
        report = {"summary": "hello", "decisions": [{"route": "dialogue", "content": "reply", "correlation_id": inputs[0].correlation_id}]}
        self.app.main_cog.invocation_id = "same-turn"
        self.app.main_cog.publish_output(report, inputs)
        accepted_ids = set(self.app.ghost.all_knoxels)
        self.app.main_cog.publish_output(report, inputs)
        self.assertEqual(set(self.app.ghost.all_knoxels), accepted_ids)
        for event in self.events.events:
            if "knoxel_id" in event.payload:
                self.assertIn(event.payload["knoxel_id"], accepted_ids)
            if "gwt_tick" in event.payload:
                self.assertIn(event.payload["gwt_tick"], self.app.ghost.gwt_turns)

    def test_invalid_second_reply_cannot_save_the_first_reply(self):
        self.app.user_input.submit("hello")
        inputs = list(self.events.events)
        before = self.path.read_bytes()
        with self.assertRaises(KeyError):
            self.app.dialogue.publish_output({"replies": [
                {"content": "first", "correlation_id": inputs[0].correlation_id},
                {"content": "missing correlation"},
            ]}, inputs)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual([event.topic for event in self.events.events], ["user.input"])

    def test_cancelled_reply_cannot_save_or_publish(self):
        self.app.user_input.submit("hello")
        inputs = list(self.events.events)
        before = self.path.read_bytes()
        self.app.dialogue.stop()
        with self.assertRaises(GhostCancelled):
            self.app.dialogue.publish_output({"replies": [
                {"content": "cancelled", "correlation_id": inputs[0].correlation_id},
            ]}, inputs)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual([event.topic for event in self.events.events], ["user.input"])
