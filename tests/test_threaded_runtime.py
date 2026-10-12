"""Threaded integration checks, separate from the direct module tests."""
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest

from workflows.companion import build_app
from pm.host.application import Application
from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope
from tests.support import Recorder
from tests.test_design_contracts import small_ghost
from tests.test_shell import _MANIFEST
from pm.memory.bash_view import MemoryBashTarget
from pm.runs.workflow_module import WorkflowModule


class ThreadedRuntimeTests(unittest.TestCase):
    def test_dialogue_uses_real_module_threads(self):
        with TemporaryDirectory() as tmp:
            replied = threading.Event()
            replies = []

            def receive(text):
                replies.append(text)
                replied.set()

            app = build_app(None, tmp, thought_period=3600, ghost=small_ghost(), emit_reply=receive)
            app.run()
            try:
                app.user_input.submit("hello")
                self.assertTrue(replied.wait(10), "dialogue did not reach the output module")
            finally:
                app.shutdown()
                app.join()
            app.raise_if_error()
            self.assertEqual(replies, ["I heard you, Jesse."])
            self.assertTrue(all(not module._thread.is_alive() for module in app.modules))

    def test_bus_terminates_an_active_workflow(self):
        class WaitingWorkflow(WorkflowModule):
            DATA_TOPICS = frozenset({"start"})

            def prepare_input(self, batch):
                return {}

            def publish_output(self, report, batch):
                raise AssertionError("terminated workflow published output")

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "role.py").write_text(
                "def run(context):\n"
                "    context.runtime.tools['started']()\n"
                "    context.runtime.stop_event.wait()\n"
                "    context.runtime.check_cancelled()\n",
                encoding="utf-8",
            )
            manifest = root / "waiting.workflow.md"
            manifest.write_text(_MANIFEST.replace("skill: role.py", f"skill: {(root / 'role.py').as_posix()}"), encoding="utf-8")
            bus = BusSystem()
            app = Application(bus, [])
            started = threading.Event()
            module = WaitingWorkflow(
                "waiting", bus, app.on_error, time.monotonic, None,
                working_target=MemoryBashTarget(small_ghost()), workflow=manifest,
                tools_factory=lambda env, batch: {"started": started.set}, runtime_root=root / "runtime",
            )
            app.modules = [module]
            bus.subscribe(module.module_id, module.accepts, module.deliver)
            terminations = Recorder(bus, "module.terminated")
            app.run()
            try:
                bus.put(Envelope("start", "test", "data", "start", "start", {}))
                self.assertTrue(started.wait(10), "role never reached its active boundary")
                bus.put(Envelope("stop", "test", "control", "terminate", "start", {}, "waiting"))
                module._thread.join(10)
                self.assertFalse(module._thread.is_alive(), "control stayed behind the active data invocation")
            finally:
                app.shutdown()
                app.join()
            app.raise_if_error()
            self.assertEqual(module.last_kernel_summary["terminal_status"], "terminated")
            self.assertEqual(len(terminations.events), 1)


if __name__ == "__main__":
    unittest.main()
