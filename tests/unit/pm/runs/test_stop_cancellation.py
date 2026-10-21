"""Cancellation-at-stop contracts for the private loop and the runtime.

The A17 live gate exposed the runtime half: cancelling in-flight model work
during a module stop surfaced as an application error. The kernel half is a
regression pin so a future change cannot reintroduce the crash.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest

from pm.bus.bus import BusSystem
from pm.memory.bash_view import MemoryBashTarget
from pm.runs.workflow_module import PROJECT_ROOT, WorkflowModule

from tests.support import ManualClock

MANIFEST = PROJECT_ROOT / "workflows" / "inner_thought" / "inner_thought.workflow.md"


class StopCancellationTests(unittest.TestCase):
    def test_failure_during_stop_is_discarded_without_an_app_error(self):
        """A cancelled in-flight role must not crash the application.

        The role waits until the run is stopped, then raises the same
        ``CancelledError`` the model boundary produces. The module discards
        the run and records no error, exactly as the silent Cancelled path
        does.
        """
        with TemporaryDirectory() as tmp:
            (Path(tmp) / "role.py").write_text(
                "import asyncio\n"
                "def run(context):\n"
                "    context.runtime.tools['await_stop']()\n"
                "    raise asyncio.CancelledError()\n",
                encoding="utf-8",
            )
            manifest = Path(tmp) / "stopping.workflow.md"
            manifest.write_text(
                "---\n"
                "name: stopping\n"
                "driver: {kind: python}\n"
                "human_resolver: {mode: forbid}\n"
                "state_policy: {mutation_model: append_only}\n"
                "roles:\n"
                "  work:\n"
                f"    skill: {(Path(tmp) / 'role.py').as_posix()}\n"
                "    tools: [await_stop]\n"
                "    result_contract:\n"
                "      type: json\n"
                "      schema:\n"
                "        status: {enum: [done]}\n"
                "        summary: string\n"
                "phases:\n"
                "  - name: work\n"
                "    kind: role\n"
                "    role: work\n"
                "    on_status: {done: stop}\n"
                "    on_invalid: {action: stop}\n"
                "---\n"
                "\n"
                "# Stopping\n",
                encoding="utf-8",
            )
            errors: list[BaseException] = []
            stopped = threading.Event()

            def await_stop():
                stopped.set()
                # Block until the module is stopped, then return so the role
                # raises its cancellation into a stopped run.
                while not module.stop_event.wait(0.05):
                    pass

            class StoppingModule(WorkflowModule):
                def prepare_input(self, batch):
                    return {}

                def publish_output(self, report, batch):
                    raise AssertionError("a stopped run must not publish")

            from tests.unit.test_design_contracts import small_ghost
            module = StoppingModule(
                "stopping_module", BusSystem(), errors.append, ManualClock(), None,
                working_target=MemoryBashTarget(small_ghost(), "read"), workflow=manifest,
                tools_factory=lambda environment, batch: {"await_stop": await_stop},
                runtime_root=Path(tmp),
            )
            runner = threading.Thread(target=module.handle_batch, args=([],))
            runner.start()
            self.assertTrue(stopped.wait(10), "the role never reached its waiting boundary")
            module.stop()
            runner.join(10)
            self.assertFalse(runner.is_alive(), "the stopped run did not finish")
            self.assertEqual(errors, [], "a stop-time cancellation must not raise")


if __name__ == "__main__":
    unittest.main()
