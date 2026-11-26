"""Drain real module threads while fencing intake and autonomous timers."""
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, new_envelope
from pm.app.control import OperatorApplication
from pm.host.application import BusModule
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature
from workflows.public_speech.user_input import UserInputProxy


class GracefulShutdownTests(unittest.TestCase):
    def test_shutdown_waits_for_running_work_and_closes_intake(self) -> None:
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        processed: list[str] = []
        class Worker(BusModule):
            DATA_TOPICS = frozenset({"start"})
            def handle_batch(self, batch: list[Envelope]) -> None:
                processed.extend(item.id for item in batch)
                started.set()
                if not release.wait(5):
                    raise TimeoutError("The test did not release its worker.")
                finished.set()
        bus = BusSystem()
        app = OperatorApplication(bus, [])
        worker = Worker("worker", app, tick_seconds=3600)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        checkpoint = Path(directory.name) / "ghost.pickle"
        store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=checkpoint)
        proxy = UserInputProxy("user_input", app, store, lambda text: None)
        app.modules = [worker, proxy]
        app.bind_operator(store, proxy, Path(directory.name))
        bus.subscribe("worker", worker.accepts, worker.deliver)
        app.run()
        try:
            in_flight = new_envelope("test", "data", "start", "", {})
            bus.put(in_flight)
            self.assertTrue(started.wait(3))
            received = store.begin_batch("received")
            received.preserve_on_recovery()
            message = received.create(Feature(content="Received before stop.", feature_type=FeatureType.Dialogue))
            store.commit(received)
            queued = new_envelope("test", "data", "start", "", {})
            bus.put(queued)
            app.request_shutdown()
            app.request_shutdown()
            with app.admission_lock:
                worker.next_tick_seconds = app.clock.monotonic()
                worker.mailbox.wake()
            self.assertFalse(app.shutdown_event.is_set())
            self.assertFalse(worker.stop_event.is_set())
            with self.assertRaises(RuntimeError):
                proxy.submit("too late")
            deadline = time.monotonic() + 3
            while not checkpoint.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertFalse(finished.is_set())
            saved = MemoryStore.from_checkpoint(checkpoint)
            self.assertIsNotNone(saved.get_knoxel_by_id(message.id))

            result = store.begin_batch("completed")
            result.preserve_on_recovery()
            completed = result.create(Feature(content="Completed during the drain.", feature_type=FeatureType.StateUpdate))
            store.commit(result)
            release.set()
            self.assertTrue(app.shutdown_event.wait(3))
            app.join()
            self.assertTrue(finished.is_set())
            self.assertEqual(processed, [in_flight.id])
            self.assertTrue(app.drained.is_set())
            app.raise_if_error()
            final = MemoryStore.from_checkpoint(checkpoint)
            self.assertIsNotNone(final.get_knoxel_by_id(completed.id))
        finally:
            release.set()
            app.shutdown()
            app.join()

    def test_failed_final_checkpoint_surfaces_original_error(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ghost.pickle"
            checkpoint.mkdir()
            store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=checkpoint)
            bus = BusSystem()
            app = OperatorApplication(bus, [])
            proxy = UserInputProxy("user_input", app, store, print)
            app.bind_operator(store, proxy, Path(directory))
            app.request_shutdown()
            self.assertTrue(app.shutdown_event.wait(5))
            app.join()
            self.assertFalse(app.drained.is_set())
            with self.assertRaises(OSError):
                app.raise_if_error()
