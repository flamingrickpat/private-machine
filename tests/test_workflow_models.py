"""Exercise lazy PM wiring with real files, cancellation, and connection errors."""
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from pydantic import ValidationError
from pm_workflows.inputs import InputError
from pm_workflows.runtime import WorkflowRuntime, WorkflowTerminated

from runtime.model_activation import ModelActivation
from runtime.models import ConnectionConfig, load_config
from runtime.workflow_models import WorkflowModels
from runtime.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from pm.host.folder_target import FolderEnvironment


class WorkflowModelTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.config = load_config(PROJECT_ROOT / "config.alpha.example.yaml")
        self.owner = ModelActivation()
        self.receipt = self.root / "model-sessions.jsonl"

    def test_legacy_python_workflow_never_activates_closed_chat(self):
        for name, connection in self.config.chat.items():
            self.config.chat[name] = connection.model_copy(update={"base_url": "http://127.0.0.1:1/v1", "startup_script": self.root / "missing.py"})
        models = WorkflowModels(self.config, self.owner, live_test=True)
        runtime = WorkflowRuntime(FolderEnvironment(self.root / "target", "bash"), Event(), {})
        runner = WorkflowRunner(PROJECT_ROOT / "workflows/local_task/local_task.workflow.md", self.root / "runtime", models=models)
        result = runner.invoke(WorkflowRequest("legacy", {"instruction": "Observe the real local worker."}), runtime)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        self.assertEqual(result.report["instruction"], "Observe the real local worker.")
        self.assertEqual(self.owner._ready, {})
        self.assertFalse((result.artifact_dir / "model-sessions.jsonl").exists())
        self.assertIsNone(runtime.classification_session)

    def test_unknown_connection_keeps_explicit_invalid_cause(self):
        models = WorkflowModels(self.config, self.owner, live_test=True)
        with self.assertRaises(InputError) as error:
            with models.session("Missing", False, Event(), receipt_path=self.receipt):
                self.fail("Unknown connections cannot admit inference.")
        self.assertEqual(error.exception.code, "missing_connection")
        self.assertEqual(self.owner._active_calls, 0)

    def test_actual_connection_refusal_releases_activation(self):
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            connection = self.config.chat["Default"].model_copy(update={"base_url": f"http://127.0.0.1:{reserved.getsockname()[1]}/v1"})
            self.config.chat["Default"] = connection
            models = WorkflowModels(self.config, self.owner, live_test=True)
            with self.assertRaises(InputError) as error:
                with models.session("Default", False, Event(), receipt_path=self.receipt):
                    self.fail("The closed port cannot admit inference.")
        self.assertEqual(error.exception.code, "endpoint_unavailable")
        self.assertFalse(self.owner._switching)
        self.assertEqual(self.owner._active_calls, 0)

    def test_precancelled_session_never_contacts_chat(self):
        cancel = Event()
        cancel.set()
        models = WorkflowModels(self.config, self.owner, live_test=True)
        with self.assertRaises(WorkflowTerminated):
            with models.session("Default", False, cancel, receipt_path=self.receipt):
                self.fail("Cancelled inference cannot start.")
        self.assertFalse(self.receipt.exists())
        self.assertEqual(self.owner._ready, {})

    def test_child_keeps_its_target_and_cancellation_with_inherited_session(self):
        models = WorkflowModels(self.config, self.owner)
        child = WorkflowRuntime(FolderEnvironment(self.root / "child", "bash"), Event(), {})
        parent = models.bind(WorkflowRuntime(FolderEnvironment(self.root / "parent", "bash"), Event(), {}, child_factory=lambda scope: child), self.receipt)
        resolved = parent.child("child")
        self.assertIs(resolved.classification_session, parent.classification_session)
        self.assertIs(resolved.environment, child.environment)
        self.assertIs(resolved.stop_event, child.stop_event)
        self.assertIsNone(child.classification_session)
        self.assertFalse(self.receipt.exists())

    def test_context_bound_is_strict_and_survives_resolution(self):
        connection = ConnectionConfig(base_url="http://localhost/v1", model="qwen", context_window=4096)
        self.assertEqual(connection.resolve().context_window, 4096)
        for value in (True, 1.5, "4096", 0, -1):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ConnectionConfig(base_url="http://localhost/v1", model="qwen", context_window=value)
