"""Check scene coverage and evidence projection without simulated inference.

The world fixture is a captured real-server response. Constructed rows and
sentinels exercise projection rules; they do not establish companion behavior.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Timer
import unittest
import time

from dev.work_items.a18b.policy import JessePacket
from dev.work_items.a18b.cli import execution_batches
from dev.work_items.a18b.runner import case_config
from dev.work_items.a18b.runner import (
    actor_send_receipt, await_delivery, delivery_mode, capability_facts, coverage_failures, execution_fact,
    load_suite, make_packets, managed_run_fact, observation_fields,
    public_messages, scene_coverage, wait_case,
    accepted_pause,
)

ROOT = Path(__file__).resolve().parents[3]


class CognitiveRunnerTests(unittest.TestCase):
    def test_pause_requires_a_committed_wait_for_the_pending_exchange(self):
        from pm.bus.bus import BusSystem
        from pm.records.enums import FeatureType
        from pm.records.memory_records import new_feature
        from tests.support import ManualClock
        from tests.unit.test_design_contracts import small_ghost
        from workflows.cognitive_control.host import CognitiveControl
        from workflows.cognitive_control.schemas import WaitRequest
        ghost = small_ghost()
        source = new_feature(ghost, "please leave it quiet", FeatureType.Dialogue,
            source=ghost.get_user_name(), metadata={"exchange_correlation": "pause-current",
                                                   "exchange_status": "pending"})
        control = CognitiveControl("control", BusSystem(), lambda error: None, ManualClock(), None, ghost=ghost)
        self.assertIsNone(accepted_pause(ghost, "pause-current"))
        control.commit(ghost.begin_batch("accepted-pause"), WaitRequest(seconds=60),
                       control.generation, source_ids=[source.id])
        outcome = accepted_pause(ghost, "pause-current")
        self.assertEqual(outcome["source_id"], source.id)
        self.assertEqual(outcome["outcome"], "accepted_pause")
        self.assertIsNone(accepted_pause(ghost, "another-exchange"))
        from datetime import timedelta
        control.now = lambda: datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=120)
        control.handle_batch([])
        self.assertIsNone(accepted_pause(ghost, "pause-current"))

    def test_cases_keep_model_settings_without_an_embedding_cache(self):
        from pm.llm.config import load_config
        source = ROOT / "config.alpha.openrouter.yaml"
        original = load_config(source)
        with TemporaryDirectory() as temporary:
            paths = []
            for name in ("one", "two"):
                root = Path(temporary) / name
                root.mkdir()
                configured = load_config(case_config(source, root))
                self.assertEqual(configured.chat, original.chat)
                self.assertEqual(configured.embedding.model, original.embedding.model)
                self.assertEqual(configured.embedding.dimensions, original.embedding.dimensions)
                self.assertNotIn("cache_dir", configured.embedding.model_fields)
                paths.append(configured.embedding.model)
            self.assertEqual(*paths)

    def test_parallel_groups_keep_minecraft_exclusive_and_normal_cases_first(self):
        suite = load_suite()
        batches = execution_batches(suite.scenarios, 3)
        flattened = [item for batch in batches for _, item in batch]
        self.assertEqual({item.id for item in flattened}, {item.id for item in suite.scenarios})
        self.assertEqual(len(flattened), len(suite.scenarios))
        edge = False
        for item in flattened:
            edge = edge or item.priority == "edge"
            self.assertFalse(edge and item.priority == "normal")
        for batch in batches:
            self.assertLessEqual(sum(item.minecraft for _, item in batch), 1)
            self.assertLessEqual(len(batch), 3)
        serial = execution_batches(suite.scenarios, 1)
        self.assertTrue(all(len(batch) == 1 for batch in serial))

    def setUp(self):
        self.suite = load_suite()
        self.scenario = self.suite.scenarios[0]
        self.at = datetime.now(timezone.utc).isoformat()

    def rows(self, speakers):
        return [{"at": self.at, "speaker": speaker, "text": "Boundary input."} for speaker in speakers]

    def episodes(self, scenario):
        return [{"scene": index, "sent": scene.messages, "completed_at": self.at, "new_companion_messages": 1}
                for index, scene in enumerate(scenario.scenes, start=1)]

    def test_normal_cases_wait_after_every_message(self):
        normal = [item for item in self.suite.scenarios if item.priority == "normal"]
        edge = [item for item in self.suite.scenarios if item.priority == "edge"]
        self.assertGreaterEqual(len(normal) / len(self.suite.scenarios), 0.8)
        self.assertEqual({item.id for item in edge}, {"overlapping_threads", "minecraft_interrupt_and_cancel"})
        for scenario in normal:
            for scene in scenario.scenes:
                self.assertEqual(scene.burst_size, 1)
                for number in range(scene.messages):
                    self.assertEqual(delivery_mode(scene, number), "normal")
                    self.assertEqual(await_delivery(scene, number), scene.await_reply)

    def test_five_message_burst_is_explicit_and_does_not_change_the_default(self):
        scene = next(item for item in self.suite.scenarios if item.id == "overlapping_threads").scenes[0]
        self.assertEqual(scene.burst_size, 5)
        self.assertEqual([await_delivery(scene, index) for index in range(6)], [False, False, False, False, True, True])
        self.assertEqual([delivery_mode(scene, index) for index in range(6)],
                         ["normal", "overlap", "overlap", "overlap", "overlap", "normal"])

    def test_normal_wait_cannot_advance_on_a_response_timer(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            timer = Timer(0.02, lambda: (root / "STOP").touch())
            timer.start()
            cancel = Event()
            try:
                with self.assertRaises(InterruptedError):
                    wait_case(None, root, time.monotonic(), 3, cancel, observed=lambda: False)
            finally:
                timer.join()
            self.assertTrue(cancel.is_set())

    def test_quiet_announcement_can_end_without_forcing_a_reply(self):
        scenario = next(item for item in self.suite.scenarios if item.id == "privacy_and_boundaries")
        scene = scenario.scenes[2]
        self.assertFalse(scene.await_reply)
        self.assertEqual(scene.messages, 1)
        self.assertFalse(await_delivery(scene, 0))
        episodes = self.episodes(scenario)
        episodes[2]["new_companion_messages"] = 0
        self.assertEqual(coverage_failures(scenario, episodes, []), [])

    def test_present_setup_does_not_reassert_a_corrected_fact(self):
        scenario = next(item for item in self.suite.scenarios if item.id == "corrected_memory")
        self.assertNotIn("blue", scenario.human_situation)
        self.assertNotIn("green", scenario.human_situation)
        self.assertIn("blue", scenario.scenes[0].human_beat)
        self.assertIn("green", scenario.scenes[1].human_beat)
        for scenario in self.suite.scenarios:
            for implementation_term in ("screenshot", "implemented emotional state", "inner life"):
                self.assertNotIn(implementation_term, scenario.premise)

    def test_scene_region_excludes_background_without_pairing_messages(self):
        rows = self.rows(["Spamuel", "Jesse", "Jesse", "Spamuel", "Spamuel"])
        coverage = scene_coverage(rows, 1, 5)
        self.assertEqual(coverage["new_companion_messages"], 2)
        self.assertEqual(coverage["start_public_id"], "m0002")
        self.assertEqual(coverage["end_public_id"], "m0005")
        self.assertEqual(scene_coverage(rows, 1, 3)["new_companion_messages"], 0)
        with self.assertRaises(ValueError):
            scene_coverage(rows, 3, 2)

    def test_one_reply_cannot_cover_every_scene(self):
        episodes = self.episodes(self.scenario)
        episodes[2]["new_companion_messages"] = 0
        errors = coverage_failures(self.scenario, episodes, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("Scene 3", errors[0])

    def test_restart_needs_postreopen_speech(self):
        scenario = next(item for item in self.suite.scenarios if item.id == "restart_continuity")
        episodes = self.episodes(scenario)
        self.assertTrue(coverage_failures(scenario, episodes, []))
        self.assertEqual(coverage_failures(scenario, episodes, [{"scene": 3, "new_companion_messages": 1}]), [])

    def test_quiet_only_scene_needs_no_synthetic_reply(self):
        scenario = next(item for item in self.suite.scenarios if item.id == "initiative_during_silence")
        episodes = self.episodes(scenario)
        episodes[2]["new_companion_messages"] = 0
        self.assertEqual(scenario.scenes[2].messages, 0)
        self.assertEqual(coverage_failures(scenario, episodes, []), [])

    def test_offline_wait_obeys_stop_and_deadline(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "STOP").touch()
            cancelled = Event()
            with self.assertRaises(InterruptedError):
                wait_case(60, root, time.monotonic(), 100, cancelled)
            self.assertTrue(cancelled.is_set())
            (root / "STOP").unlink()
            with self.assertRaises(TimeoutError):
                wait_case(60, root, time.monotonic() - 101, 100, Event())

    def test_send_receipt_keeps_the_frozen_context_and_actual_delivery(self):
        packet = JessePacket(situation="Present human circumstances.", current_beat="Current human intent.",
                             conversation=public_messages(self.rows(["Jesse", "Spamuel"])))
        receipt = actor_send_receipt(packet, composing_at=self.at, composed_at=self.at, submitted_at=self.at,
            delivery={"at": self.at, "correlation": "actual-correlation", "text": "A delivered message."}, public_id="m0004")
        self.assertEqual(receipt["packet_sha256"], hashlib.sha256(packet.model_dump_json().encode()).hexdigest())
        self.assertEqual(receipt["visible_messages"], 2)
        self.assertEqual(receipt["public_id"], "m0004")
        self.assertEqual(receipt["correlation"], "actual-correlation")

    def test_actual_world_response_projects_values_without_images(self):
        capture = json.loads((ROOT / "test_data/captures/alpha/cognitive-world-observation.json").read_text(encoding="utf-8"))
        selected = observation_fields(capture["response"])
        self.assertIn("camera", selected)
        self.assertEqual(selected["inventory"], [])
        self.assertEqual(selected["player"]["health"], 20)
        raw = json.loads(json.dumps(capture["response"]))
        raw["result"]["private_plan"] = "PRIVATE_SENTINEL"
        raw["result"]["inventory"] = [{"item": "minecraft:dirt", "count": 1, "lore": "PRIVATE_SENTINEL"}]
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(observation_fields(raw)))
        raw["ok"] = False
        self.assertEqual(observation_fields(raw), {})

    def test_capability_details_remain_grounding_only(self):
        scenario = next(item for item in self.suite.scenarios if item.id == "minecraft_limited_senses")
        from dev.work_items.a18b.policy import Fact
        facts = tuple(Fact(id=f"f{index}", at=at, about=about, text=text)
                      for index, (at, about, text) in enumerate(capability_facts(scenario, self.at)))
        packets = make_packets(scenario, self.suite, (), facts)
        self.assertFalse(packets["editor"].facts)
        self.assertNotIn("screenshot pixel", packets["editor"].model_dump_json())
        self.assertIn("screenshot pixel", packets["grounding"].model_dump_json())

    def test_managed_facts_do_not_claim_world_success_or_disclose_plans(self):
        run = {"workflow_name": "minecraft", "state": "completed", "instruction": "PRIVATE_SENTINEL",
               "latest_event": "PRIVATE_SENTINEL", "events": [{"message": {"event": "completed", "summary": "PRIVATE_SENTINEL"}}]}
        fact = managed_run_fact(run, self.at, "sampling time only", "r0001")
        self.assertIsNotNone(fact)
        self.assertNotIn("PRIVATE_SENTINEL", fact[2])
        self.assertIn("does not prove", fact[2])
        self.assertEqual(fact[0], self.at)
        marker = execution_fact({"event": "uncertain", "time": 1792100000, "tool": "minecraft_mine",
                                 "arguments": {"private": "PRIVATE_SENTINEL"}}, "call time only", "r0001")
        self.assertNotIn("PRIVATE_SENTINEL", marker[2])
        self.assertIn("uncertain", marker[2])
