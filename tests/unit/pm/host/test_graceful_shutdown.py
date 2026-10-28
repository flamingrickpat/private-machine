"""Drain real module threads while fencing intake and autonomous timers."""
import threading
import time
import unittest

from pm.bus.bus import BusSystem
from pm.bus.envelopes import new_envelope
from pm.host.application import Application, BusModule
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from workflows.public_speech.user_input import UserInputProxy


class GracefulShutdownTests(unittest.TestCase):
    def test_shutdown_waits_for_running_work_and_closes_intake(self):
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        class Worker(BusModule):
            DATA_TOPICS = frozenset({"start"})
            def handle_batch(self, batch):
                started.set()
                if not release.wait(5):
                    raise TimeoutError("The test did not release its worker.")
                finished.set()
        bus = BusSystem()
        app = Application(bus, [])
        worker = Worker("worker", bus, app.on_error, time.monotonic)
        store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        proxy = UserInputProxy("user_input", bus, app.on_error, time.monotonic, store, lambda text: None)
        app.modules = [worker, proxy]
        app.user_input = proxy
        app.ghost = store
        bus.subscribe("worker", worker.accepts, worker.deliver)
        app.run()
        try:
            bus.put(new_envelope("test", "data", "start", "", {}))
            self.assertTrue(started.wait(3))
            app.request_shutdown()
            self.assertFalse(app.shutdown_event.is_set())
            self.assertFalse(worker.stop_event.is_set())
            with self.assertRaises(RuntimeError):
                proxy.submit("too late")
            release.set()
            self.assertTrue(app.shutdown_event.wait(3))
            app.join()
            self.assertTrue(finished.is_set())
            self.assertTrue(app.drained.is_set())
            app.raise_if_error()
        finally:
            release.set()
            app.shutdown()
            app.join()
