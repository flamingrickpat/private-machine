"""Check world-event commit and cancellation with a retained real observation."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.bus.bus import BusSystem
from pm.bus.envelopes import new_envelope
from pm.records.global_workspace import GwtTurnRecord
from pm.records.cycle_state import CycleState
from tests.support import ManualClock, Recorder
from tests.unit.test_design_contracts import small_ghost
from workflows.minecraft_body.world_events import WorldEvents, world_conditions


class WorldEventTests(unittest.TestCase):
    def setUp(self):
        self.observation = json.loads(Path("test_data/captures/alpha/minecraft-world-observation.json").read_text(encoding="utf-8"))["observation"]
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock, self.bus, self.ghost = ManualClock(), BusSystem(), small_ghost()
        self.events = Recorder(self.bus, "minecraft.world")
        self.module = WorldEvents("world_events", self.bus, self.fail, self.clock, 10,
                                  ghost=self.ghost, observe=lambda: {"ok": True, "result": deepcopy(self.observation)},
                                  journal=Path(self.tmp.name) / "events.jsonl", quiet_seconds=20)

    def test_silence_commits_observation_before_event_and_coalesces_backlog(self):
        self.clock.advance(10)
        self.module.step()
        self.assertEqual(self.events.events, [])
        self.clock.advance(10)
        self.module.step()
        event = self.events.events[0]
        record = self.ghost.get_knoxel_by_id(event.payload["knoxel_id"])
        self.assertEqual(record.metadata["evidence"], self.observation)
        self.assertTrue(record.metadata["self_origin"])
        self.assertEqual(record.metadata["correlation_id"], event.correlation_id)
        self.clock.advance(100)
        self.module.step()
        self.assertEqual(len(self.events.events), 1)

    def test_accepted_event_allows_a_changed_inventory_event(self):
        self.clock.advance(20)
        self.module.step()
        original_id = self.module.pending_id
        accepted = self.ghost.begin_batch("accepted-world", new_tick=True)
        accepted.set_gwt(GwtTurnRecord(tick_id=accepted.tick_id, knoxel_ids=[original_id]),
                         CycleState(tick_id=accepted.tick_id, stimuli_ids=[original_id]))
        self.ghost.commit(accepted)
        self.module.handle_batch([new_envelope("main_cog", "data", "gwt.updated", "", {"gwt_tick": 1})])
        self.assertIsNone(self.module.pending_id)
        self.clock.advance(10)
        self.module.step()
        self.assertEqual(len(self.events.events), 1)
        self.observation["inventory"][0]["count"] += 1
        self.clock.advance(10)
        self.module.step()
        self.assertEqual(len(self.events.events), 2)
        latest = self.ghost.get_knoxel_by_id(self.module.pending_id)
        original = json.loads(Path("test_data/captures/alpha/minecraft-world-observation.json").read_text(encoding="utf-8"))["observation"]
        self.assertEqual(latest.metadata["previous_conditions"], world_conditions(original))

    def test_shutdown_discards_late_observation(self):
        def observed_after_stop():
            self.module.stop()
            return {"ok": True, "result": self.observation}
        self.module.observe = observed_after_stop
        self.clock.advance(20)
        self.module.step()
        self.assertEqual(self.events.events, [])
        self.assertIsNone(self.module.pending_id)

    def test_user_arrival_during_a_read_cancels_the_silence_baseline(self):
        def observed_after_user_arrival():
            self.module.deliver(new_envelope("user_input", "data", "user.input", "return", {"knoxel_id": 1}))
            return {"ok": True, "result": self.observation}
        self.module.observe = observed_after_user_arrival
        self.clock.advance(20)
        self.module.step()
        self.assertEqual(self.events.events, [])
        self.assertIsNone(self.module.last_conditions)

    def test_rejected_cycle_reobserves_unchanged_conditions_with_a_new_id(self):
        self.clock.advance(20)
        self.module.step()
        original_id, correlation = self.module.pending_id, self.module.pending_correlation
        self.module.handle_batch([new_envelope("main_cog", "telemetry", "action.receipt", correlation,
            {"status": "stale", "recovery_outcome": "cycle_rejected"})])
        self.assertIsNone(self.module.pending_id)
        self.clock.advance(10)
        self.module.step()
        self.assertNotEqual(self.module.pending_id, original_id)
        self.assertEqual(len(self.events.events), 2)
        self.assertFalse(self.module.retry_observation)

    def test_active_project_progress_is_observed_during_conversation(self):
        self.clock.advance(20)
        self.module.step()
        original_id = self.module.pending_id
        accepted = self.ghost.begin_batch("accepted-baseline", new_tick=True)
        accepted.set_gwt(GwtTurnRecord(tick_id=accepted.tick_id, knoxel_ids=[original_id]),
                         CycleState(tick_id=accepted.tick_id, stimuli_ids=[original_id]))
        self.ghost.commit(accepted)
        self.module.handle_batch([new_envelope("main_cog", "data", "gwt.updated", "", {"gwt_tick": accepted.tick_id})])
        live = json.loads(Path("test_data/captures/alpha/world-active-project.json").read_text(encoding="utf-8"))["run_receipt"]
        self.ghost.managed_runs[live["run_id"]] = live
        self.module.deliver(new_envelope("user_input", "data", "user.input", "conversation", {"knoxel_id": 1}))
        self.module.step()
        self.observation["inventory"][0]["count"] += 1
        self.clock.advance(10)
        self.module.step()
        record = self.ghost.get_knoxel_by_id(self.module.pending_id)
        self.assertFalse(record.metadata["user_silent"])
        self.assertIn("during active game work", record.content)

    def test_world_time_and_camera_rotation_do_not_create_per_tick_events(self):
        changed = deepcopy(self.observation)
        changed["camera"]["yaw"] += 90
        changed["player"]["time_of_day"] += 1
        self.assertEqual(world_conditions(changed), world_conditions(self.observation))
        changed["player"]["time_of_day"] = 18000
        self.assertNotEqual(world_conditions(changed), world_conditions(self.observation))

    def test_world_event_never_becomes_user_speech(self):
        from workflows.main_cognition.stages.codelets.steps import stimulus_label
        self.assertEqual(stimulus_label({"kind": "minecraft.world"}, "Jesse"),
                         ("Minecraft observation", "outside", "observed"))
