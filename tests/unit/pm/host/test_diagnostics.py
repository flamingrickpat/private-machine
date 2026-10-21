"""Record a real worker crash and its shutdown in a private log."""
from pathlib import Path
import tempfile
import time
import unittest

from pm.bus.bus import BusSystem
from pm.host.application import Application, BusModule
from pm.host.diagnostics import close_logging, configure_logging


class DiagnosticsTests(unittest.TestCase):
    def test_worker_crash_records_thread_traceback_and_join(self):
        class FailingModule(BusModule):
            def handle_batch(self, batch):
                raise RuntimeError("diagnostic crash marker")

        with tempfile.TemporaryDirectory() as directory:
            path = configure_logging(Path(directory))
            try:
                app = Application(BusSystem(), [])
                module = FailingModule("crashing-worker", app.bus, app.on_error, time.monotonic, .01)
                app.modules.append(module)
                app.run()
                self.assertTrue(app.shutdown_event.wait(5))
                app.join()
                with self.assertRaisesRegex(RuntimeError, "diagnostic crash marker"):
                    app.raise_if_error()
            finally:
                close_logging()
            text = path.read_text(encoding="utf-8")
            for expected in ("pid=", "thread=", "crashing-worker", "Traceback", "diagnostic crash marker",
                             "Thread join end", "Application shutdown complete"):
                self.assertIn(expected, text)
            self.assertEqual(path.parent, Path(directory) / "logs")
