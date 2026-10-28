"""Restore real ghost files and the routing ledger at one coordinated boundary."""
from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.host.durable_files import atomic_json
from pm.memory.checkpoint import write_checkpoint
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Feature
from pm.vclock import VirtualClock
from simulation.engine import RunEngine
from simulation.personas import write_home
from simulation.plans import ScenarioPlan


def admit(store, text):
    batch = store.begin_batch(text, new_tick=True)
    record = batch.create(Feature(content=text, feature_type=FeatureType.Thought))
    batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, knoxel_ids=[record.id]))
    store.commit(batch)
    return record.id


class CoordinatedCheckpointTests(unittest.TestCase):
    def test_restore_keeps_both_ghosts_clock_and_transit_on_the_same_branch(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            seed = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
            first = admit(seed, "Before the barrier")
            home = write_home(seed, root / "source")
            plan = ScenarioPlan.model_validate({"title": "Checkpoint", "brief": "Say hello", "start_time": "2026-10-28T21:00:00+01:00",
                "facts": [{"id": "hello", "text": "Say hello"}],
                "steps": [{"id": "hello", "at": "2026-10-28T21:00:00+01:00", "instruction": "Say hello",
                           "fact_ids": ["hello"], "completion": "Hello was delivered"}]})
            original_clock = VirtualClock(datetime.fromisoformat("2026-10-28T20:00:00+01:00"))
            original_clock.set(plan.start_time)
            atomic_json(home.root / "runtime/clock.json", {"clock": original_clock.state(), "timers": {"inner_thought": 3630}})
            engine = RunEngine.create(root / "run", plan, [{"id": "primary", "home": str(home.root)},
                {"id": "other", "home": str(home.root)}], [], "config.alpha.example.yaml")
            engine.state["attempt"] = 1
            engine.state["transit"] = [{"delivery_id": "waiting-at-barrier", "status": "pending"}]
            engine.save()
            snapshots = {}
            for item in engine.state["participants"]:
                participant = Path(item["home"])
                store = MemoryStore.from_checkpoint(participant / "ghost.pickle")
                snapshot = participant / "runtime/snapshots/barrier.pickle"
                write_checkpoint(snapshot, store.checkpoint_data(), lambda: None)
                atomic_json(snapshot.with_suffix(".json"), {"clock": VirtualClock(plan.start_time).state(), "timers": {"inner_thought": 30}})
                snapshots[item["id"]] = {"path": str(snapshot)}
                admit(store, "After the barrier")
            atomic_json(engine.root / "attempts/1/coordinated-checkpoint.json", {
                "clock": engine.state["clock"], "controller": engine.view(), "snapshots": snapshots})
            engine.state["attempt"] = 2
            engine.state["transit"] = []
            engine.save()
            engine.restore(1)
            self.assertEqual(engine.state["attempt"], 2)
            self.assertEqual(engine.state["transit"][0]["delivery_id"], "waiting-at-barrier")
            for item in engine.state["participants"]:
                participant = Path(item["home"])
                restored = MemoryStore.from_checkpoint(participant / "ghost.pickle")
                self.assertEqual(restored.get_knoxel_by_id(first).content, "Before the barrier")
                self.assertFalse(any(r.content == "After the barrier" for r in restored.all_knoxels.values()))
                archive = next(participant.glob("ghost.before-restore-*.pickle"))
                later = MemoryStore.from_checkpoint(archive)
                self.assertTrue(any(r.content == "After the barrier" for r in later.all_knoxels.values()))
                clock = json.loads((participant / "runtime/clock.json").read_text())
                self.assertEqual(clock["timers"]["inner_thought"], 30)
            intent = json.loads((engine.root / "restore.json").read_text())
            intent["status"] = "prepared"
            atomic_json(engine.root / "restore.json", intent)
            import shutil
            participant = Path(engine.state["participants"][0]["home"])
            shutil.copyfile(next(participant.glob("ghost.before-restore-*.pickle")), participant / "ghost.pickle")
            interrupted = RunEngine(engine.root)
            interrupted._finish_restore()
            restored = MemoryStore.from_checkpoint(participant / "ghost.pickle")
            self.assertFalse(any(r.content == "After the barrier" for r in restored.all_knoxels.values()))
            interrupted.restore(0)
            self.assertEqual(interrupted.state["transit"], [])
            self.assertEqual(interrupted.state["attempt"], 2)
            clock = json.loads((participant / "runtime/clock.json").read_text())
            self.assertEqual(VirtualClock.from_state(clock["clock"]).monotonic(), 3600)
            self.assertEqual(clock["timers"]["inner_thought"], 3630)
