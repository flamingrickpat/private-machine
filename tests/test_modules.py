"""Direct module behavior checks without threads."""
from __future__ import annotations

import unittest

from pm.records.memory_records import ActionSchemaRecord, Feature
from pm.records.base import RecordKind
from pm.records.enums import ActionType, FeatureType

from pm.bus.envelopes import Envelope, _new_id

from tests.support import (
    Recorder,
    fresh_ghost,
    load_seed_state,
    make_app,
    store_action,
    store_feature,
)


def _user_input_id(app, correlation_id: str):
    # Recover the stored Feature id by looking for the newest user feature.
    return correlation_id


class ModuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed_dict = load_seed_state()


class UserInputTests(ModuleTests):
    def test_submit(self):
        app = make_app(self.seed_dict)
        recorder = Recorder(app.bus, topic="user.input")
        correlation_id = app.user_input.submit("hello")
        self.assertEqual(app.replies, [])
        self.assertEqual(len(recorder.events), 1)
        self.assertEqual(recorder.events[0].correlation_id, correlation_id)
        knoxel_id = recorder.events[0].payload["knoxel_id"]
        feature = app.ghost.get_knoxel_by_id(knoxel_id)
        self.assertEqual(feature.content, "hello")
        self.assertEqual(feature.feature_type, FeatureType.Dialogue)

    def test_print_public_reply(self):
        app = make_app(self.seed_dict)
        feature_id = store_feature(app.ghost, "public reply text")
        app.bus.put(Envelope(
            id=_new_id(), source="dialogue", kind="data", topic="dialogue.reply",
            correlation_id="corr", payload={"knoxel_id": feature_id},
        ))
        app.user_input.step(block=False)
        self.assertEqual(app.replies, ["public reply text"])


class MainCogTests(ModuleTests):
    def test_batches_two_inputs(self):
        app = make_app(self.seed_dict)
        inputs = Recorder(app.bus, topic="user.input")
        corr_a = app.user_input.submit("hello")
        corr_b = app.user_input.submit("world")
        actions = Recorder(app.bus, topic="action.dialogue")

        before_tick = app.ghost.current_tick_id
        self.assertTrue(app.main_cog.step(block=False))

        self.assertEqual(app.ghost.current_tick_id, before_tick + 1)
        tick = app.ghost.current_tick_id
        gwt = app.ghost.gwt_turns[tick]
        self.assertEqual(len(actions.events), 2)
        self.assertEqual({e.correlation_id for e in actions.events}, {corr_a, corr_b})
        input_ids = {e.payload["knoxel_id"] for e in inputs.events}
        self.assertTrue(input_ids.issubset(set(gwt.knoxel_ids)))

    def test_routes_each_command(self):
        cases = [
            ("/status task-123", ["status"]),
            ("/memory", ["tool_dialogue"]),
            ("please minecraft now", ["schedule", "dialogue"]),
            ("/demo-task", ["schedule", "dialogue"]),
            ("you dumb robot", ["dialogue"]),
            ("how are you", ["dialogue"]),
        ]
        for text, routes in cases:
            with self.subTest(text=text):
                app = make_app(self.seed_dict)
                app.user_input.submit(text)
                app.main_cog.step(block=False)
                recorder = Recorder(app.bus)
                actions = app.ghost.knoxels_by_class(ActionSchemaRecord)
                recent = sorted(actions, key=lambda a: a.id)[-len(routes):]
                self.assertEqual([a.metadata["route"] for a in recent], routes)

    def test_consumes_telemetry_without_new_action(self):
        app = make_app(self.seed_dict)
        feature_id = store_feature(app.ghost, "telemetry feature")
        before = app.ghost.current_tick_id
        app.bus.put(Envelope(
            id=_new_id(), source="scheduler", kind="telemetry", topic="task.telemetry",
            correlation_id="corr", payload={"knoxel_id": feature_id, "task_id": "task-1", "status": "running"},
        ))
        actions = Recorder(app.bus, topic="action.dialogue")
        self.assertTrue(app.main_cog.step(block=False))
        self.assertEqual(len(actions.events), 0)
        gwt = app.ghost.gwt_turns[app.ghost.current_tick_id]
        self.assertIn(feature_id, gwt.knoxel_ids)

    def test_ignores_own_broadcast(self):
        app = make_app(self.seed_dict)
        before = app.ghost.current_tick_id
        app.bus.put(Envelope(
            id=_new_id(), source="main_cog", kind="data", topic="gwt.updated",
            correlation_id="", payload={"gwt_tick": before},
        ))
        self.assertFalse(app.main_cog.step(block=False))
        self.assertEqual(app.ghost.current_tick_id, before)


class DialogueTests(ModuleTests):
    def test_public_reply(self):
        app = make_app(self.seed_dict)
        action = store_action(app.ghost, route="dialogue", content="I heard you, Jesse.")
        app.bus.put(action)
        replies = Recorder(app.bus, topic="dialogue.reply")
        self.assertTrue(app.dialogue.step(block=False))
        self.assertEqual(len(replies.events), 1)
        feature = app.ghost.get_knoxel_by_id(replies.events[0].payload["knoxel_id"])
        self.assertEqual(feature.content, "I heard you, Jesse.")
        self.assertEqual(replies.events[0].correlation_id, action.correlation_id)
        self.assertIn("outbox/report.yaml", app.dialogue.last_environment.list_files("outbox"))


class InnerThoughtTests(ModuleTests):
    def test_timer(self):
        app = make_app(self.seed_dict)
        self.assertFalse(app.inner_thought.step(block=False))
        app.clock.advance(30.0)
        self.assertTrue(app.inner_thought.step(block=False))
        self.assertFalse(app.inner_thought.step(block=False))
        self.assertEqual(app.replies, [])


class SchedulerTests(ModuleTests):
    def test_dummy_task_transitions(self):
        app = make_app(self.seed_dict)
        action = store_action(app.ghost, route="schedule", task_id="task-x", task_kind="dummy")
        app.bus.put(action)
        telemetry = Recorder(app.bus, topic="task.telemetry")

        self.assertTrue(app.scheduler.step(block=False))
        self.assertEqual(telemetry.events[-1].payload["status"], "accepted")

        app.clock.advance(1.0)
        self.assertTrue(app.scheduler.step(block=False))
        self.assertEqual(telemetry.events[-1].payload["status"], "running")

        app.clock.advance(1.0)
        self.assertTrue(app.scheduler.step(block=False))
        self.assertEqual(telemetry.events[-1].payload["status"], "completed")

        record = app.tasks.get("task-x")
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["correlation_id"], action.correlation_id)

    def test_duplicate_schedule(self):
        app = make_app(self.seed_dict)
        action = store_action(app.ghost, route="schedule", task_id="task-x", task_kind="dummy")
        telemetry = Recorder(app.bus, topic="task.telemetry")
        app.bus.put(action)
        app.bus.put(action)
        self.assertTrue(app.scheduler.step(block=False))
        accepted = [e for e in telemetry.events if e.payload["status"] == "accepted"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(app.tasks._records), 1)

    def test_minecraft_is_unsupported(self):
        app = make_app(self.seed_dict)
        action = store_action(app.ghost, route="schedule", task_id="task-m", task_kind="minecraft")
        telemetry = Recorder(app.bus, topic="task.telemetry")
        app.bus.put(action)
        self.assertTrue(app.scheduler.step(block=False))
        self.assertEqual(telemetry.events[-1].payload["status"], "unsupported")
        app.clock.advance(2.0)
        app.scheduler.step(block=False)
        statuses = [e.payload["status"] for e in telemetry.events]
        self.assertNotIn("running", statuses)
        self.assertNotIn("completed", statuses)


class StatusTests(ModuleTests):
    def test_known_task(self):
        app = make_app(self.seed_dict)
        app.tasks.create("task-x", "dummy", "orig-corr")
        record = app.tasks.get("task-x")
        record["status"] = "running"
        action = store_action(app.ghost, route="status", task_id="task-x")
        telemetry = Recorder(app.bus, topic="task.telemetry")
        app.bus.put(action)
        self.assertTrue(app.status.step(block=False))
        event = telemetry.events[-1]
        self.assertEqual(event.correlation_id, action.correlation_id)
        self.assertEqual(event.payload["status"], "running")

    def test_unknown_task(self):
        app = make_app(self.seed_dict)
        action = store_action(app.ghost, route="status", task_id="nope")
        telemetry = Recorder(app.bus, topic="task.telemetry")
        app.bus.put(action)
        self.assertTrue(app.status.step(block=False))
        self.assertEqual(telemetry.events[-1].payload["status"], "unknown")
        self.assertIsNone(app.tasks.get("nope"))


class ToolDialogueTests(ModuleTests):
    def test_real_memory_search(self):
        app = make_app(self.seed_dict)
        action = store_action(app.ghost, route="tool_dialogue")
        app.bus.put(action)
        replies = Recorder(app.bus, topic="dialogue.reply")
        self.assertTrue(app.tool_dialogue.step(block=False))
        environment = app.tool_dialogue.last_environment
        from pm.memory.store_tools import build_tools
        expected = len(build_tools(environment)["ghost_search"]("Jesse"))
        feature = app.ghost.get_knoxel_by_id(replies.events[0].payload["knoxel_id"])
        self.assertEqual(feature.content, f"I found {expected} memories about Jesse.")


if __name__ == "__main__":
    unittest.main()