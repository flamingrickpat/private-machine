"""Application adapters preserve scoped input versions and exact text."""
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

import yaml
from pm_workflows.inputs import InputError, InputResolver, InputScope
from pm_workflows.runtime import WorkflowRuntime
from runtime.workflow_runner import WorkflowRequest, WorkflowRunner, WorkflowStatus
from working_target import FolderEnvironment, GhostBashEnvironment


class WorkflowInputTests(unittest.TestCase):
    def test_missing_ghost_file_keeps_the_missing_cause(self):
        environment = GhostBashEnvironment({}, {}, "read")
        resolver = InputResolver(environment, InputScope("test", "run"))
        with self.assertRaises(InputError) as caught:
            resolver.capture_file("logs/missing.log")
        self.assertEqual(caught.exception.code, "missing_file")

    def test_application_metadata_keeps_host_time_and_unknown_virtual_time(self):
        with TemporaryDirectory() as directory:
            for environment, host in [(FolderEnvironment(Path(directory), "bash"), True), (GhostBashEnvironment({}, {}, "read"), False)]:
                environment.write_text("logs/report.txt", "12 passed\r\n\u00df\n")
                resolver = InputResolver(environment, InputScope("test", "run"))
                listing = resolver.list_folder("logs")
                candidate = listing.candidates[0]
                self.assertEqual(candidate.modified_time is not None, host)
                self.assertIsNone(candidate.producer)
                self.assertEqual(resolver.capture_selected(listing, candidate.candidate_id).value, "12 passed\r\n\u00df\n")
                environment.write_text("logs/report.txt", "1 failed\n")
                with self.assertRaisesRegex(InputError, "stale_input"):
                    resolver.capture_selected(listing, candidate.candidate_id)

    def test_application_paths_reject_absolute_and_traversal_reads(self):
        with TemporaryDirectory() as directory:
            for environment in [FolderEnvironment(Path(directory), "bash"), GhostBashEnvironment({}, {}, "read")]:
                for path in ["../outside.txt", "/outside.txt", "C:/outside.txt"]:
                    with self.assertRaises(InputError):
                        environment.read_text(path)

    def test_runner_supplies_request_fields_and_records_capture(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "role.py").write_text(
                "def run(context):\n"
                "    import yaml\n"
                "    context.runtime.environment.write_text('outbox/report.yaml', yaml.safe_dump({'job': context.inputs['job']}))\n"
                "    return {'status': 'done', 'job': context.inputs['job']}\n", encoding="utf-8")
            manifest = root / "probe.workflow.md"
            manifest.write_text("---\n" + yaml.safe_dump({
                "name": "request-bind", "driver": {"kind": "python"},
                "roles": {"work": {"skill": "role.py", "result_contract": {"schema": {"status": {"enum": ["done"]}}}}},
                "phases": [{"name": "work", "kind": "role", "role": "work", "on_status": {"done": "stop"}, "on_invalid": {"action": "stop"},
                            "inputs": {"job": {"request": "job.id", "expected_type": "string"}},
                            "outputs": {"job": {"pointer": "/job", "expected_type": "string"}}}],
            }) + "---\n", encoding="utf-8")
            runtime = WorkflowRuntime(GhostBashEnvironment({}, {}, "read"), Event(), {}, True)
            result = WorkflowRunner(manifest, root / "runtime", base_dir=root).invoke(WorkflowRequest("request", {"job": {"id": "current"}}), runtime)
            self.assertEqual(result.status, WorkflowStatus.Completed)
            self.assertEqual(result.report, {"job": "current"})
            self.assertFalse((root / "runtime/standalone/request/workspace/outbox/report.yaml").exists())

    def test_binary_host_input_fails_as_malformed_text(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / "binary.bin").write_bytes(b"\xff\xfe")
            resolver = InputResolver(FolderEnvironment(Path(directory), "bash"), InputScope("test", "run"))
            with self.assertRaisesRegex(InputError, "malformed_value"):
                resolver.capture_file("binary.bin")
