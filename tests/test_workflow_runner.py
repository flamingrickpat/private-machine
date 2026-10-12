"""Run actual Python workflows synchronously on real and virtual targets."""
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest

import yaml
from pm_workflows.runtime import WorkflowRuntime

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope
from runtime.workflow_runner import WorkflowRequest, WorkflowRunner, WorkflowStatus
from pm.host.folder_target import FolderTarget
from pm.memory.bash_view import GhostBashEnvironment
from pm.host.workflow_module import ShellModuleWorkflow


class WorkflowRunnerTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def runner(self, body, *, phase=None, success_status="done"):
        role = self.root / "role.py"
        role.write_text("def run(context):\n" + body, encoding="utf-8")
        manifest = self.root / "probe.workflow.md"
        document = {
            "name": "probe", "driver": {"kind": "python"},
            "human_resolver": {"mode": "external"},
            "roles": {"work": {
                "skill": str(role),
                "result_contract": {"schema": {
                    "status": {"enum": ["done", "rejected"]}, "summary": "string",
                }},
            }},
            "phases": [phase or {
                "name": "work", "kind": "role", "role": "work",
                "on_status": {"done": "stop", "rejected": "stop"},
                "on_invalid": {"action": "stop"},
            }],
        }
        manifest.write_text("---\n" + yaml.safe_dump(document) + "---\n", encoding="utf-8")
        return WorkflowRunner(manifest, self.root / "runtime", base_dir=self.root, success_status=success_status)

    def runtime(self, target="virtual"):
        environment = (
            GhostBashEnvironment({}, {}, "read") if target == "virtual"
            else FolderTarget(self.root / "target", "bash").open("probe")
        )
        return WorkflowRuntime(environment, threading.Event(), {}, propagate_python_errors=True)

    def test_stop_route_completes_on_both_targets_in_the_calling_thread(self):
        runner = self.runner(
            "    import threading, yaml\n"
            "    env = context.runtime.environment\n"
            "    request = yaml.safe_load(env.read_text('inbox/request.yaml'))\n"
            "    request['thread_id'] = threading.get_ident()\n"
            "    env.write_text('outbox/report.yaml', yaml.safe_dump(request))\n"
            "    return {'status': 'done', 'summary': 'copied'}\n"
        )
        for target in ("virtual", "folder"):
            with self.subTest(target=target):
                runtime = self.runtime(target)
                result = runner.invoke(WorkflowRequest(target, {"text": "input"}), runtime)
                self.assertEqual(result.status, WorkflowStatus.Completed)
                self.assertEqual(result.report, {"text": "input", "thread_id": threading.get_ident()})
                self.assertFalse(result.kernel_summary["ok"])
                self.assertEqual(result.kernel_summary["exit_reason"], "work: stop")
                self.assertEqual(result.run_id, target)
                self.assertTrue((result.artifact_dir / "journal.jsonl").is_file())

    def test_failed_role_does_not_load_its_report(self):
        runner = self.runner(
            "    context.runtime.environment.write_text('outbox/report.yaml', 'partial: output')\n"
            "    return {'status': 'rejected', 'summary': 'not accepted'}\n"
        )
        result = runner.invoke(WorkflowRequest("failed", {}), self.runtime())
        self.assertEqual(result.status, WorkflowStatus.Failed)
        self.assertIsNone(result.report)
        self.assertEqual(result.kernel_summary["terminal_status"], "rejected")

    def test_invalid_contract_cannot_be_success(self):
        runner = self.runner("    return {'status': 'done'}\n")
        result = runner.invoke(WorkflowRequest("invalid", {}), self.runtime())
        self.assertEqual(result.status, WorkflowStatus.Failed)
        self.assertIsNone(result.report)

    def test_pre_cancelled_run_does_not_write_input_or_execute_role(self):
        runner = self.runner("    raise AssertionError('cancelled role ran')\n")
        runtime = self.runtime()
        runtime.stop_event.set()
        result = runner.invoke(WorkflowRequest("cancel-before", {}), runtime)
        self.assertEqual(result.status, WorkflowStatus.Cancelled)
        self.assertEqual(runtime.environment.list_files("inbox"), [])
        self.assertIsNone(result.report)

    def test_cancellation_inside_role_discards_report(self):
        runner = self.runner(
            "    context.runtime.environment.write_text('outbox/report.yaml', 'partial: output')\n"
            "    context.runtime.stop_event.set()\n"
            "    context.runtime.check_cancelled()\n"
        )
        result = runner.invoke(WorkflowRequest("cancel-inside", {}), self.runtime())
        self.assertEqual(result.status, WorkflowStatus.Cancelled)
        self.assertIsNone(result.report)

    def test_external_question_returns_waiting_without_reading_report(self):
        runner = self.runner("    raise AssertionError('unused role')\n", phase={
            "name": "question", "kind": "human", "question": "Which color?", "next": "stop",
        })
        result = runner.invoke(WorkflowRequest("question", {}), self.runtime())
        self.assertEqual(result.status, WorkflowStatus.WaitingForInput)
        self.assertEqual(result.kernel_summary["exit_reason"], "suspended:user")
        self.assertIsNone(result.report)

    def test_original_python_exception_reaches_debugger(self):
        runner = self.runner("    raise ValueError('original role error')\n")
        with self.assertRaisesRegex(ValueError, "original role error"):
            runner.invoke(WorkflowRequest("broken", {}), self.runtime())

    def test_success_requires_a_mapping_report(self):
        runner = self.runner(
            "    context.runtime.environment.write_text('outbox/report.yaml', '- not a mapping')\n"
            "    return {'status': 'done', 'summary': 'invalid report'}\n"
        )
        with self.assertRaisesRegex(ValueError, "report.*mapping"):
            runner.invoke(WorkflowRequest("wrong-report", {}), self.runtime())

    def module(self, runner):
        class ProbeWorkflow(ShellModuleWorkflow):
            def prepare_input(self, batch):
                return batch[0].payload

            def publish_output(self, report, batch):
                self.published.append(report)

        module = ProbeWorkflow(
            "probe", BusSystem(), self.fail, lambda: 0, None,
            working_target=FolderTarget(self.root / "module-target", "bash"),
            workflow=runner.workflow, tools_factory=lambda environment, batch: {},
            runtime_root=self.root / "runtime",
        )
        module.published = []
        module.mailbox.put(Envelope("request", "test", "data", "probe", "request", {"text": "hello"}))
        return module

    def test_module_step_uses_the_normalized_result_in_the_calling_thread(self):
        runner = self.runner(
            "    import threading, yaml\n"
            "    context.runtime.environment.write_text('outbox/report.yaml', yaml.safe_dump({'thread_id': threading.get_ident()}))\n"
            "    return {'status': 'done', 'summary': 'complete'}\n"
        )
        module = self.module(runner)
        self.assertTrue(module.step())
        self.assertEqual(module.last_result.status, WorkflowStatus.Completed)
        self.assertEqual(module.published, [{"thread_id": threading.get_ident()}])
        self.assertIs(module.last_result.kernel_summary, module.last_kernel_summary)
        self.assertIsNone(module._thread)

    def test_module_exposes_waiting_result_without_publication(self):
        runner = self.runner("    raise AssertionError('unused')\n", phase={
            "name": "question", "kind": "human", "question": "Which color?", "next": "stop",
        })
        module = self.module(runner)
        self.assertTrue(module.step())
        self.assertEqual(module.last_result.status, WorkflowStatus.WaitingForInput)
        self.assertEqual(module.published, [])

    def test_module_exposes_failed_result_and_raises_without_publication(self):
        module = self.module(self.runner("    return {'status': 'rejected', 'summary': 'not accepted'}\n"))
        with self.assertRaisesRegex(RuntimeError, "ended unsuccessfully"):
            module.step()
        self.assertEqual(module.last_result.status, WorkflowStatus.Failed)
        self.assertEqual(module.published, [])


if __name__ == "__main__":
    unittest.main()
