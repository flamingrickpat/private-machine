"""Wait boundaries with real stores, checkpoints, bus delivery, and kernels.

Deterministic stage reports test scheduling only. These checks do not establish
persona behavior, model selection of Sleep, or companion quality.
"""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pydantic import ValidationError

from pm.bus.bus import BusSystem
from pm.bus.envelopes import new_envelope
from pm.memory.snapshot import GhostCancelled
from pm.memory.store import MemoryStore
from pm.records.enums import FeatureType
from pm.records.memory_records import new_feature
from tests.support import ManualClock, Recorder
from tests.unit.test_design_contracts import small_ghost
from tests.unit.workflows.inner_thought.test_private_host import _module as private_module, _reports as private_reports
from tests.unit.workflows.main_cognition.cycle_support import ScriptedModel, scripted_tools
from tests.unit.workflows.main_cognition.test_cognition_state import _module as main_module, _reports, _store
from workflows.cognitive_control.host import CognitiveControl
from workflows.cognitive_control.schemas import ControlConfig, WaitRequest
from workflows.main_cognition.schemas import RouteAnswer


class WaitTests(unittest.TestCase):
    def setUp(self):
        self.ghost = small_ghost()
        self.bus = BusSystem()
        self.now = datetime(2026, 10, 17, 12)
        self.events = Recorder(self.bus)
        self.control = self.controller()

    def controller(self, ghost=None, **kwargs):
        return CognitiveControl("cognitive_control", self.bus, self.fail, ManualClock(), 1,
                                ghost=ghost or self.ghost, now=lambda: self.now, **kwargs)

    def arm(self, seconds=30, control=None, commit_id="wait"):
        control = control or self.control
        batch = control.ghost.begin_batch(commit_id)
        control.commit(batch, WaitRequest(seconds=seconds), control.generation)
        return batch

    def external(self):
        key = _store(self.ghost, "a new message")
        return new_envelope("user_input", "data", "user.input", "new-input", {"knoxel_id": key})

    def wakes(self, ghost=None):
        return [record for record in (ghost or self.ghost).all_features
                if record.metadata.get("control_kind") == "inference_wake"]

    def test_wait_is_durable_and_timer_makes_no_event_before_deadline(self):
        with TemporaryDirectory() as tmp:
            self.ghost.checkpoint = Path(tmp) / "ghost.json"
            self.arm()
            self.control.handle_batch([])
            reopened = MemoryStore.from_checkpoint(self.ghost.checkpoint)
            self.assertTrue(self.controller(reopened).waiting())
            self.assertEqual(self.events.events, [])
            self.assertEqual(self.wakes(), [])

    def test_checkpoint_failure_accepts_neither_result_nor_wait(self):
        with TemporaryDirectory() as tmp:
            self.ghost.checkpoint = Path(tmp)  # A real directory cannot be a checkpoint file.
            before = self.ghost.state_dict()
            batch = self.ghost.begin_batch("failed-commit", new_tick=True)
            new_feature(self.ghost, "unaccepted result", FeatureType.Thought, source="test", changes=batch)
            with self.assertRaises(OSError):
                self.control.commit(batch, WaitRequest(seconds=30), 0)
            self.assertEqual(self.ghost.state_dict(), before)
            self.assertFalse(self.control.waiting())

    def test_new_input_invalidates_a_sleep_already_in_flight(self):
        generation = self.control.generation
        self.control.observe(self.external())
        self.control.commit(self.ghost.begin_batch("late-sleep"), WaitRequest(seconds=30), generation)
        self.assertFalse(self.control.waiting())
        rejected = [record for record in self.ghost.all_features
                    if record.metadata.get("control_kind") == "inference_wait_rejected"]
        self.assertEqual(rejected[0].metadata["status"], "superseded")

    def test_input_observed_before_run_but_not_in_batch_prevents_wait(self):
        envelope = self.external()
        self.control.observe(envelope)
        self.control.commit(self.ghost.begin_batch("not-handled-yet"), WaitRequest(seconds=30), self.control.generation)
        self.assertFalse(self.control.waiting())
        self.control.commit(self.ghost.begin_batch("handled"), WaitRequest(seconds=30), self.control.generation,
                            envelope_ids={envelope.id})
        self.assertTrue(self.control.waiting())

    def test_repeated_commit_does_not_rearm_or_advance_the_fence(self):
        self.arm()
        generation = self.control.generation
        self.now += timedelta(seconds=10)
        self.arm(commit_id="wait")
        self.assertEqual(self.control.generation, generation)
        self.assertEqual(len([record for record in self.ghost.all_features
                              if record.metadata.get("control_kind") == "inference_wait"]), 1)

    def test_host_enforces_operator_maximum_on_a_valid_model_duration(self):
        control = self.controller(config=ControlConfig(maximum_wait_seconds=5))
        self.arm(seconds=100, control=control)
        self.now += timedelta(seconds=5)
        control.handle_batch([])
        self.assertEqual(len(self.wakes()), 1)
        self.assertEqual(self.wakes()[0].metadata["elapsed_seconds"], 5)

    def test_timeout_commits_once_and_replays_only_until_main_acknowledges(self):
        with TemporaryDirectory() as tmp:
            self.ghost.checkpoint = Path(tmp) / "ghost.json"
            self.arm()
            self.now += timedelta(seconds=50)
            self.control.handle_batch([])
            self.control.handle_batch([])
            self.assertEqual(len(self.events.events), 1)
            wake = self.wakes()[0]
            self.assertEqual(wake.metadata["elapsed_seconds"], 50)
            reopened = MemoryStore.from_checkpoint(self.ghost.checkpoint)
            controller = self.controller(reopened)
            controller.handle_batch([])
            self.assertEqual(len(self.events.events), 2)
            self.assertEqual(self.events.events[-1].payload["knoxel_id"], wake.id)
            controller.commit(reopened.begin_batch("accept-wake"), None, controller.generation, wake_ids=[wake.id])
            restored = MemoryStore.from_checkpoint(self.ghost.checkpoint)
            self.controller(restored).handle_batch([])
            self.assertEqual(len(self.events.events), 2)
            self.assertEqual(len(self.wakes(restored)), 1)

    def test_crash_after_wake_commit_before_delivery_does_not_lose_wake(self):
        with TemporaryDirectory() as tmp:
            self.ghost.checkpoint = Path(tmp) / "ghost.json"
            self.arm()
            self.now += timedelta(seconds=30)
            def broken_delivery(envelope, recipient):
                raise ConnectionError("publication interrupted")
            self.bus.observer = broken_delivery
            with self.assertRaises(ConnectionError):
                self.control.handle_batch([])
            self.assertEqual(len(self.wakes()), 1)
            self.assertEqual(self.events.events, [])
            self.bus.observer = None
            reopened = MemoryStore.from_checkpoint(self.ghost.checkpoint)
            self.controller(reopened).handle_batch([])
            self.assertEqual(self.events.events[0].topic, "cognition.wake")

    def test_startup_cannot_sleep_over_an_unconsumed_timeout(self):
        self.arm()
        self.now += timedelta(seconds=30)
        self.control.handle_batch([])
        wake = self.wakes()[0]
        self.control.commit(self.ghost.begin_batch("startup-sleep"), WaitRequest(seconds=30), self.control.generation)
        self.assertFalse(self.control.waiting())
        self.control.commit(self.ghost.begin_batch("timeout-choice"), WaitRequest(seconds=30),
                            self.control.generation, wake_ids=[wake.id])
        self.assertTrue(self.control.waiting())
        self.assertEqual(self.ghost.get_knoxel_by_id(wake.id).metadata["wake_status"], "consumed")

    def test_cancelled_main_releases_wake_for_replay(self):
        self.arm()
        self.now += timedelta(seconds=30)
        self.control.handle_batch([])
        self.control.release_wakes(self.events.events[:])
        self.control.handle_batch([])
        self.assertEqual(len(self.events.events), 2)
        self.assertEqual(len(self.wakes()), 1)

    def test_external_wake_does_not_invent_a_timer_event_or_speech(self):
        self.arm()
        envelope = self.external()
        self.control.observe(envelope)
        self.control.handle_batch([])
        self.assertFalse(self.control.waiting())
        self.assertEqual(self.events.events, [])
        self.assertEqual(self.wakes(), [])

    def test_shutdown_at_the_timer_commit_boundary_does_not_become_an_application_error(self):
        self.arm()
        before = self.ghost.state_dict()
        self.now += timedelta(seconds=30)
        self.control.stop()
        self.control.handle_batch([])
        self.assertEqual(self.ghost.state_dict(), before)
        self.assertEqual(self.events.events, [])

    def test_self_feedback_and_nonterminal_worker_telemetry_do_not_wake(self):
        self.arm()
        for topic, payload in [("thought.created", {}), ("gwt.updated", {}),
                               ("system.lifecycle", {}), ("workflow.event", {"event": "running"})]:
            self.control.observe(new_envelope("test", "data", topic, "", payload))
        self.assertTrue(self.control.waiting())

    def test_private_commit_cannot_cross_input_or_wait_boundaries(self):
        batch = self.ghost.begin_batch("private")
        new_feature(self.ghost, "old private output", FeatureType.Thought, source="test", changes=batch)
        generation = self.control.generation
        envelope = self.external()
        self.control.observe(envelope)
        with self.assertRaises(GhostCancelled):
            self.control.commit_private(batch, generation)
        self.control.commit(self.ghost.begin_batch("wait"), WaitRequest(seconds=30),
                            self.control.generation, envelope_ids={envelope.id})
        with self.assertRaises(GhostCancelled):
            self.control.commit_private(batch, self.control.generation)

    def test_queued_old_thought_is_filtered_but_current_external_input_is_retained(self):
        thought = new_feature(self.ghost, "old thought", FeatureType.InnerMonologue, source="test",
                              metadata={"control_generation": self.control.generation})
        old = new_envelope("inner_thought", "data", "thought.created", "", {"knoxel_id": thought.id})
        fresh = self.external()
        self.control.observe(fresh)
        accepted, generation = self.control.admit([old, fresh])
        self.assertEqual(accepted, [fresh])
        self.assertEqual(generation, self.control.generation)

    def test_sleep_rejects_bad_timeouts_inside_the_typed_stage_boundary(self):
        for value in (0, -1, 86401, float("inf"), float("nan"), True, "later", None):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                RouteAnswer.model_validate({"action_type": "Sleep", "arguments": {"timeout_seconds": value}})
        answer = RouteAnswer.model_validate({"action_type": "Sleep", "arguments": {"timeout_seconds": 30}})
        self.assertEqual(answer.arguments.timeout_seconds, 30)


class KernelWaitTests(unittest.TestCase):
    def test_restart_preserves_wait_without_redispatching_its_pending_user_source(self):
        from workflows.companion import build_app

        ghost = small_ghost()
        source = new_feature(ghost, "please wait", FeatureType.Dialogue, source=ghost.get_user_name(),
                             metadata={"exchange_correlation": "paused", "exchange_status": "pending"})
        with TemporaryDirectory() as tmp:
            control = CognitiveControl("cognitive_control", BusSystem(), self.fail, ManualClock(), 1, ghost=ghost)
            control.commit(ghost.begin_batch("accepted-wait"), WaitRequest(seconds=300), 0, source_ids=[source.id])
            app = build_app(None, Path(tmp), ghost=ghost, cognition="real",
                            config_path="config.alpha.openrouter.yaml", record_lifecycle=False, world_poll_seconds=None)
            try:
                self.assertTrue(app.cognitive_control.waiting())
                self.assertEqual(app.main_cog.mailbox.take_batch(app.main_cog.stop_event, 0), [])
                self.assertEqual(ghost.get_knoxel_by_id(source.id).metadata["exchange_status"], "pending")
            finally:
                app.shutdown()
                app.join()

    def test_wake_outbox_cannot_be_replayed_as_public_speech(self):
        from workflows.companion import build_app

        ghost = small_ghost()
        now = [datetime(2026, 10, 17)]
        control = CognitiveControl("cognitive_control", BusSystem(), self.fail, ManualClock(), 1,
                                   ghost=ghost, now=lambda: now[0])
        control.commit(ghost.begin_batch("wait"), WaitRequest(seconds=3), 0)
        now[0] += timedelta(seconds=3)
        control.handle_batch([])
        with TemporaryDirectory() as tmp:
            app = build_app(None, Path(tmp), ghost=ghost, cognition="real",
                            config_path="config.alpha.openrouter.yaml", record_lifecycle=False, world_poll_seconds=None)
            try:
                self.assertEqual(app.user_input.mailbox.take_batch(app.user_input.stop_event, 0), [])
                self.assertEqual(app.main_cog.mailbox.take_batch(app.main_cog.stop_event, 0), [])
                app.cognitive_control.handle_batch([])
                queued = app.main_cog.mailbox.take_batch(app.main_cog.stop_event, 0)
                self.assertEqual([item.topic for item in queued], ["cognition.wake"])
            finally:
                app.shutdown()
                app.join()

    def test_waiting_invokes_neither_private_nor_main_stages(self):
        ghost = small_ghost()
        source_id = _store(ghost, "old message")
        calls = []
        with TemporaryDirectory() as tmp:
            private, bus, _ = private_module(ghost, private_reports(), tmp, calls)
            main, _, _ = main_module(ghost, _reports(source_id), tmp)
            control = CognitiveControl("cognitive_control", bus, self.fail, ManualClock(), 1, ghost=ghost)
            private.cognitive_control = main.cognitive_control = control
            control.commit(ghost.begin_batch("wait"), WaitRequest(seconds=30), 0)
            before = ghost.state_dict()
            private.handle_batch([])
            main.handle_batch([new_envelope("inner_thought", "data", "thought.created", "", {"knoxel_id": source_id})])
            self.assertIsNone(private.last_result)
            self.assertIsNone(main.last_result)
            self.assertEqual(calls, [])
            self.assertEqual(ghost.state_dict(), before)

    def test_kernel_commits_sleep_with_frame_and_wakes_on_new_input(self):
        ghost = small_ghost()
        key = _store(ghost, "leave me quiet for now")
        reports = _reports(key, actions=[{"action_type": "Sleep", "workflow_name": "", "content": "wait",
                                          "arguments": {"timeout_seconds": 30}}])
        with TemporaryDirectory() as tmp:
            module, bus, errors = main_module(ghost, reports, tmp)
            control = CognitiveControl("cognitive_control", bus, self.fail, ManualClock(), 1, ghost=ghost)
            module.cognitive_control = control
            requests = Recorder(bus, "action.request")
            envelope = new_envelope("user_input", "data", "user.input", "quiet", {"knoxel_id": key})
            bus.put(envelope)
            module.step()
            self.assertTrue(control.waiting())
            self.assertEqual(errors, [])
            self.assertEqual(requests.events, [])
            self.assertIn(key, ghost.states[-1].stimuli_ids)
            self.assertEqual(len(ghost.gwt_shard("main")), 1)
            next_key = _store(ghost, "actually, hello again")
            reports.update(_reports(next_key))
            bus.put(new_envelope("user_input", "data", "user.input", "new", {"knoxel_id": next_key}))
            self.assertFalse(control.waiting())
            module.step()
            self.assertIn(next_key, ghost.states[-1].stimuli_ids)
            self.assertEqual(len(requests.events), 1)

    def test_wait_after_private_work_also_parks_the_already_appraised_user_source(self):
        ghost = small_ghost()
        source = new_feature(ghost, "I want silence", FeatureType.Dialogue, source=ghost.get_user_name(),
                             metadata={"exchange_correlation": "quiet", "exchange_status": "pending"})
        reports = _reports(source.id)
        reports["action_selection"]["actions"] = []
        with TemporaryDirectory() as tmp:
            module, bus, errors = main_module(ghost, reports, tmp)
            control = CognitiveControl("cognitive_control", bus, self.fail, ManualClock(), 1, ghost=ghost)
            module.cognitive_control = control
            bus.put(new_envelope("user_input", "data", "user.input", "quiet", {"knoxel_id": source.id}))
            module.step()
            thought = new_feature(ghost, "I will stop active thought", FeatureType.InnerMonologue, source="Spamuel",
                                  metadata={"control_generation": control.generation})
            reports.update(_reports(thought.id, actions=[{"action_type": "Sleep", "workflow_name": ""}]))
            bus.put(new_envelope("inner_thought", "data", "thought.created", "", {"knoxel_id": thought.id}))
            module.step()
            self.assertEqual(errors, [])
            self.assertTrue(control.waiting())
            self.assertIn(source.id, control.paused_source_ids())
            self.assertEqual(ghost.get_knoxel_by_id(source.id).metadata["exchange_status"], "pending")

    def test_timeout_is_system_scheduling_and_duplicate_delivery_needs_no_second_cycle(self):
        ghost = small_ghost()
        now = [datetime(2026, 10, 17)]
        with TemporaryDirectory() as tmp:
            module, bus, errors = main_module(ghost, {}, tmp)
            control = CognitiveControl("cognitive_control", bus, self.fail, ManualClock(), 1,
                                       ghost=ghost, now=lambda: now[0])
            module.cognitive_control = control
            control.commit(ghost.begin_batch("wait"), WaitRequest(seconds=3), 0)
            now[0] += timedelta(seconds=3)
            control.handle_batch([])
            wake = next(record for record in ghost.all_features if record.metadata.get("origin") == "cognition.wake")
            # The real parent and child protocol use fixed reports here. Only
            # routing, atomic consumption, and duplicate admission are checked.
            module.tools_factory = scripted_tools(ScriptedModel(reports=_reports(wake.id)))
            module.step()
            self.assertEqual(errors, [])
            self.assertEqual(ghost.get_knoxel_by_id(wake.id).metadata["wake_status"], "consumed")
            cycle_count = len(ghost.states)
            duplicate = new_envelope("cognitive_control", "data", "cognition.wake", "", {"knoxel_id": wake.id})
            self.assertEqual(module._stimulus_from(duplicate)[2], "cognition.wake")
            bus.put(duplicate)
            module.step()
            self.assertEqual(len(ghost.states), cycle_count)


if __name__ == "__main__":
    unittest.main()
