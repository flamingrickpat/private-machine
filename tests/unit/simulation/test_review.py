"""Check immutable review registration and scenario timing without model calls."""
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.introspection.record_packages import capture_records
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature
from pm import vclock
from simulation.plans import ScenarioPlan, StageJudgment
from simulation.review.catalog import load_capture, register
from simulation.review.record_format import present_packages


class SimulationReviewTests(unittest.TestCase):
    def test_supported_facts_require_delivery_citations(self):
        judgment = {"step_complete": True, "supported_fact_ids": ["greeting"], "reason": "The actor asked for a greeting."}
        with self.assertRaises(ValueError):
            StageJudgment.model_validate(judgment)
        judgment["fact_evidence"] = {"greeting": ["recorded-delivery"]}
        self.assertEqual(StageJudgment.model_validate(judgment).fact_evidence["greeting"], ["recorded-delivery"])

    def test_registered_packages_keep_exact_text_and_reject_evidence_replacement(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
            record = store.store_knoxel(Feature(content="Exact private thought", feature_type=FeatureType.Thought))
            package = capture_records(store.snapshot(), [record.id], title="Engineering capture")
            bundle = {"id": "engineering", "title": "Engineering capture", "items": [{"packages": [package]}]}
            url = register(bundle, root=root)
            self.assertTrue(url.endswith("#capture/engineering"))
            self.assertEqual(load_capture("engineering", root=root), bundle)
            self.assertEqual(register(bundle, root=root), url)
            changed = json.loads(json.dumps(bundle))
            changed["title"] = "Different evidence"
            with self.assertRaises(ValueError):
                register(changed, root=root)
            display = present_packages(bundle["items"][0])
            self.assertEqual(display["cognition"]["records"][str(record.id)]["text"], record.content)

    def test_combining_conflicting_record_copies_fails(self):
        with TemporaryDirectory() as directory:
            store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
            record = store.store_knoxel(Feature(content="Original", feature_type=FeatureType.Thought))
            package = capture_records(store.snapshot(), [record.id], title="Original")
            changed = json.loads(json.dumps(package))
            changed["records"][0]["data"]["content"] = "Conflicting copy"
            with self.assertRaises(ValueError):
                register({"id": "conflict", "items": [{"packages": [package, changed]}]}, root=Path(directory))

    def test_restored_clock_retains_elapsed_time_and_offset(self):
        clock = vclock.VirtualClock(datetime.fromisoformat("2026-10-28T23:00:00+01:00"))
        clock.set(datetime.fromisoformat("2026-10-29T07:00:00+01:00"))
        restored = vclock.VirtualClock.from_state(clock.state())
        self.assertEqual(restored.monotonic(), 8 * 3600)
        self.assertEqual(restored.now().hour, 7)
        self.assertEqual(restored.utc().hour, 6)

    def test_timed_fact_cannot_move_to_a_different_stage_time(self):
        plan = {"title": "Morning", "brief": "Wake at seven", "start_time": "2026-10-28T21:00:00+01:00",
                "facts": [{"id": "wake", "text": "Wake at seven", "at": "2026-10-29T07:00:00+01:00"}],
                "steps": [{"id": "morning", "at": "2026-10-29T07:00:00+01:00", "instruction": "Wake",
                           "fact_ids": ["wake"], "completion": "The actor reports waking"}]}
        ScenarioPlan.model_validate(plan)
        plan["steps"][0]["at"] = "2026-10-29T07:00:00+00:00"
        with self.assertRaises(ValueError):
            ScenarioPlan.model_validate(plan)
