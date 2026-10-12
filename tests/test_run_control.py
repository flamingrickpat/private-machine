"""Per-run cancellation and waiting-input routing at the module boundary."""
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope
from runtime.workflow_runner import WorkflowStatus
from tests.support import Recorder
from working_target import FolderTarget
from workflow_adapter import ShellModuleWorkflow


def _wait_until(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class _ProbeWorkflow(ShellModuleWorkflow):
    DATA_TOPICS = frozenset({"start"})

    def prepare_input(self, batch):
        return {}

    def publish_output(self, report, batch):
        self.published.append(report)


class RunControlTests(unittest.TestCase):
    def _module(self, root, manifest, tools_factory, **kwargs):
        module = _ProbeWorkflow(
            "probe", BusSystem(), self.fail, time.monotonic, None,
            working_target=FolderTarget(root / "target", "bash"),
            workflow=manifest, tools_factory=tools_factory,
            runtime_root=root / "runtime", **kwargs,
        )
        module.published = []
        return module

    def test_run_cancellation_leaves_the_module_usable(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            role = root / "role.py"
            role.write_text(
                "def run(context):\n"
                "    context.runtime.tools['started']()\n"
                "    if context.runtime.tools['block']():\n"
                "        context.runtime.stop_event.wait()\n"
                "        context.runtime.check_cancelled()\n"
                "    context.runtime.environment.write_text('outbox/report.yaml', 'answer: ok\\n')\n"
                "    return {'status': 'done', 'summary': 'ok'}\n",
                encoding="utf-8",
            )
            manifest = root / "probe.workflow.md"
            manifest.write_text(
                "---\nname: probe\ndriver: {kind: python}\nroles:\n"
                f"  work:\n    skill: {role.as_posix()}\n"
                "    result_contract:\n      schema: {status: {enum: [done]}, summary: string}\n"
                "phases:\n  - name: work\n    kind: role\n    role: work\n"
                "    on_status: {done: stop}\n    on_invalid: {action: stop}\n---\n",
                encoding="utf-8",
            )
            started = threading.Event()
            state = {"block": True, "runs": 0}

            def tools(environment, batch):
                def mark_started():
                    state["runs"] += 1
                    started.set()

                def block():
                    value = state["block"]
                    state["block"] = False
                    return value

                return {"started": mark_started, "block": block}

            module = self._module(root, manifest, tools)
            module.run()
            try:
                module.mailbox.put(Envelope("one", "test", "data", "start", "one", {}))
                self.assertTrue(started.wait(10), "first run never started")
                module.cancel_run()
                self.assertTrue(
                    _wait_until(lambda: module.last_result is not None
                                and module.last_result.status is WorkflowStatus.Cancelled),
                    "cancelled run did not return a cancelled result",
                )
                self.assertFalse(module.stop_event.is_set(), "cancelling a run stopped the module")
                self.assertTrue(module._thread.is_alive(), "the module thread exited with the run")

                started.clear()
                module.mailbox.put(Envelope("two", "test", "data", "start", "two", {}))
                self.assertTrue(started.wait(10), "the module did not accept a later run")
                self.assertTrue(
                    _wait_until(lambda: module.last_result is not None
                                and module.last_result.status is WorkflowStatus.Completed),
                    "later run did not complete",
                )
                self.assertEqual(module.published, [{"answer": "ok"}])
            finally:
                module.stop()
                module.join()

    def test_waiting_question_is_published_and_resumed_by_answer(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work.py"
            work.write_text(
                "import yaml\n"
                "def run(context):\n"
                "    context.runtime.environment.write_text(\n"
                "        'outbox/report.yaml', yaml.safe_dump({'answer': context.answer}))\n"
                "    return {'status': 'done', 'summary': 'answered'}\n",
                encoding="utf-8",
            )
            manifest = root / "asker.workflow.md"
            manifest.write_text(
                "---\nname: asker\ndriver: {kind: python}\n"
                "human_resolver: {mode: external}\nroles:\n"
                f"  work:\n    skill: {work.as_posix()}\n"
                "    result_contract:\n      schema: {status: {enum: [done]}, summary: string}\n"
                "phases:\n"
                "  - name: ask\n    kind: human\n    question: \"Which color?\"\n    next: work\n"
                "  - name: work\n    kind: role\n    role: work\n"
                "    on_status: {done: stop}\n    on_invalid: {action: stop}\n---\n",
                encoding="utf-8",
            )
            module = self._module(root, manifest, lambda environment, batch: {})
            bus = module.bus
            bus.subscribe(module.module_id, module.accepts, module.deliver)
            questions = Recorder(bus, "workflow.question")

            module.mailbox.put(Envelope("start", "test", "data", "start", "start", {}))
            self.assertTrue(module.step())
            self.assertIs(module.last_result.status, WorkflowStatus.WaitingForInput)
            self.assertEqual(module.last_result.question, "Which color?")
            self.assertEqual(module.last_result.resume_phase, "ask")
            self.assertIsNotNone(module._pending)
            self.assertEqual(module.published, [])
            self.assertEqual(len(questions.events), 1)
            self.assertEqual(questions.events[0].payload["question"], "Which color?")
            self.assertEqual(questions.events[0].payload["resume_phase"], "ask")

            question_id = questions.events[0].payload["question_id"]
            bus.put(Envelope(
                "answer", "test", "data", "workflow.answer", "start",
                {"question_id": question_id, "answer": "blue"}, target="probe",
            ))
            self.assertTrue(module.step())
            self.assertIs(module.last_result.status, WorkflowStatus.Completed)
            self.assertEqual(module.published, [{"answer": "blue"}])
            self.assertIsNone(module._pending)


if __name__ == "__main__":
    unittest.main()