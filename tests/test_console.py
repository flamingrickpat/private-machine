"""Exercise real worker processes, durable memory, traces, and lifecycle."""
import json
from pathlib import Path
import tempfile
import time
import unittest
import yaml

from pm.memory.store import Ghost
from pm.records.cycle_state import GhostConfig
from pm.records.memory_records import Feature
from pm.records.enums import FeatureType

from console.server import Console, ROOT


class ConsoleTests(unittest.TestCase):
    def wait_for(self, instance, predicate, timeout=25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with instance.lock:
                if predicate(list(instance.events)):
                    return
            if instance.process.poll() is not None:
                self.fail((instance.root / "worker.log").read_text(encoding="utf-8"))
            time.sleep(.05)
        self.fail("The worker did not produce the expected event.")

    def test_worker_trace_memory_sessions_restart_and_kill(self):
        with tempfile.TemporaryDirectory() as directory:
            console = Console(Path(directory))
            ghost = Ghost(GhostConfig(user_name="Rick", companion_name="Spamuel"))
            ghost.store_knoxel(Feature(content="fixture memory", feature_type=FeatureType.Dialogue))
            seed = Path(directory) / "seed.yaml"
            seed.write_text(yaml.safe_dump(ghost.state_dict()), encoding="utf-8")
            deployment = console.deploy({"name": "test", "cognition": "dummy", "seed": str(seed)})
            instance = console.instances[deployment["id"]]
            try:
                instance.start()
                self.wait_for(instance, lambda events: any(e["kind"] == "ready" for e in events))
                instance.command({"action": "chat", "session": "alice-session", "user": "Alice", "text": "Hello"})
                self.wait_for(instance, lambda events: any(e["kind"] == "chat.reply" for e in events))
                replies = [e for e in instance.events if e["kind"] == "chat.reply"]
                self.assertEqual(replies[0]["session"], "alice-session")
                self.assertTrue(any(e["kind"] == "bus" and e["recipient"] == "main_cog" for e in instance.events))
                self.assertTrue(any(e["kind"] == "workflow.end" for e in instance.events))
                instance.command({"action": "snapshot"})
                records = instance.snapshot["records"]
                record_id, record = next((key, value) for key, value in records.items() if value["__type__"] == "Feature")
                revision = instance.snapshot["revisions"][record_id]
                changed = json.loads(json.dumps(record))
                changed["data"]["content"] = "Injected operator memory"
                instance.command({"action": "memory", "record": changed, "revision": revision})
                self.assertEqual(instance.snapshot["records"][record_id]["data"]["content"], "Injected operator memory")
                with self.assertRaisesRegex(ValueError, "memory changed"):
                    instance.command({"action": "memory", "record": changed, "revision": revision})
                instance.command({"action": "memory", "record": changed, "create": True})
                self.assertEqual(len(instance.snapshot["records"]), len(records) + 1)
                instance.command({"action": "chat", "session": "bob-session", "user": "Bob", "text": "Hello again"})
                self.wait_for(instance, lambda events: sum(e["kind"] == "chat.reply" for e in events) == 2)
                self.assertEqual([e for e in instance.events if e["kind"] == "chat.reply"][-1]["session"], "bob-session")
                instance.command({"action": "stop"})
                instance.process.wait(timeout=20)
                self.assertEqual(instance.process.returncode, 0)
                self.assertTrue((instance.root / "ghost.json").is_file())
                instance.start()
                self.wait_for(instance, lambda events: sum(e["kind"] == "ready" for e in events) == 2)
                instance.command({"action": "snapshot"})
                self.assertEqual(instance.snapshot["records"][record_id]["data"]["content"], "Injected operator memory")
                instance.kill()
                self.assertEqual(instance.info()["state"], "stopped")
            finally:
                instance.kill()
                if instance.process is not None:
                    instance.process.wait(timeout=10)
                    time.sleep(.1)

    def test_deployment_validation_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            console = Console(Path(directory))
            with self.assertRaises(ValueError):
                console.deploy({"seed": str(ROOT / "missing.yaml")})
            with self.assertRaises(ValueError):
                console.deploy({"cognition": "invalid"})
            deployment = console.deploy({"name": "saved", "cognition": "dummy"})
            restored = Console(Path(directory))
            self.assertEqual(restored.instances[deployment["id"]].info()["name"], "saved")


if __name__ == "__main__":
    unittest.main()
