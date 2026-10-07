"""Steward domain tests: dormancy, evidence, update boundary, termination."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm_workflows.runtime import WorkflowRuntime

from pm_core.ghost.service import Ghost
from pm_core.model.ghost_state import GhostConfig
from runtime.managed_runs import ManagedRunSupervisor, WorkflowSpec
from runtime.steward import Steward
from runtime.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.steward.schemas import StewardReport
from working_target import GhostBashEnvironment

LOCAL_TASK = PROJECT_ROOT / "workflows" / "local_task"
STEWARD_MANIFEST = PROJECT_ROOT / "workflows" / "steward" / "steward.workflow.md"


def _wait_for(predicate, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class StewardTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ghost = Ghost(
            GhostConfig(user_name="Jesse", companion_name="Spamuel"),
            checkpoint=self.root / "ghost.json",
        )
        self.supervisor = ManagedRunSupervisor(
            self.ghost,
            runtime_root=self.root / "runtime",
            workflows={
                "local_task": WorkflowSpec(
                    LOCAL_TASK / "local_task.workflow.md", namespace="local_task",
                ),
                "local_task_question": WorkflowSpec(
                    LOCAL_TASK / "local_task_question.workflow.md",
                    namespace="local_task_question",
                ),
            },
            clock=time.monotonic,
            task_root=self.root / "tasks",
        )
        self.steward = Steward(self.supervisor, clock=time.monotonic)

    def tearDown(self):
        self.supervisor.join_all(timeout=10)
        self._tmp.cleanup()

    def _start(self, action_id: str, workflow: str = "local_task", **arguments):
        return self.supervisor.start(
            action_id=action_id, workflow_name=workflow,
            instructions="note", arguments=arguments or None,
        )

    def test_dormant_steward_changes_nothing(self):
        receipt = self._start("action-0", wait_for_cancel=True)
        self.assertTrue(self.steward.dormant)
        self.assertFalse(self.steward.status(receipt["run_id"])["accepted"])
        self.assertFalse(self.steward.update(receipt["run_id"], "new")["accepted"])
        self.assertFalse(self.steward.terminate(receipt["run_id"])["accepted"])
        # It never sets the cancellation signal.
        self.assertNotEqual(self.supervisor.status(receipt["run_id"])["state"], "cancelled")
        self.supervisor.terminate(receipt["run_id"])

    def test_active_status_is_evidence_backed(self):
        receipt = self._start("action-1")
        run_id = receipt["run_id"]
        self.assertTrue(_wait_for(lambda: self.supervisor.status(run_id)["state"] == "completed"))
        self.steward.activate()
        report = self.steward.status(run_id)
        self.assertTrue(report["accepted"])
        self.assertEqual(report["state"], "completed")
        self.assertGreater(report["observed_at"], 0)
        self.assertTrue(report["latest_event"])
        self.assertIn("outbox/report.yaml", report["evidence"])

    def test_outstanding_question_id_is_reported_and_answerable(self):
        receipt = self._start("action-2", workflow="local_task_question")
        run_id = receipt["run_id"]
        self.assertTrue(
            _wait_for(lambda: self.supervisor.status(run_id)["state"] == "waiting_for_input")
        )
        self.steward.activate()
        report = self.steward.status(run_id)
        self.assertTrue(report["question_id"])
        answered = self.steward.answer(run_id, report["question_id"], "the red artifact")
        self.assertTrue(answered["accepted"])
        self.assertTrue(_wait_for(lambda: self.supervisor.status(run_id)["state"] == "completed"))

    def test_update_boundary_reaches_the_same_run(self):
        receipt = self._start("action-3", workflow="local_task_question")
        run_id = receipt["run_id"]
        self.assertTrue(
            _wait_for(lambda: self.supervisor.status(run_id)["state"] == "waiting_for_input")
        )
        self.steward.activate()
        updated = self.steward.update(run_id, "revised instruction", request_id="req-3")
        self.assertTrue(updated["accepted"])
        self.assertEqual(updated["revision"], 1)
        self.assertEqual(updated["request_id"], "req-3")
        # The duplicate request does not bump the applied revision.
        again = self.steward.update(run_id, "revised instruction", request_id="req-3")
        self.assertEqual(again["revision"], 1)
        # Answering resumes the run with the revised instruction.
        report = self.steward.status(run_id)
        self.steward.answer(run_id, report["question_id"], "answer")
        self.assertTrue(_wait_for(lambda: self.supervisor.status(run_id)["state"] == "completed"))
        observations = (Path(receipt["artifact_path"]) / "scratch" / "observations.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("revised instruction", observations)

    def test_termination_is_code_only_and_preserves_artifacts(self):
        receipt = self._start("action-4", wait_for_cancel=True)
        run_id = receipt["run_id"]
        artifact = Path(receipt["artifact_path"])
        self.assertTrue(_wait_for(lambda: (artifact / "scratch" / "started.md").is_file()))
        self.steward.activate()
        result = self.steward.terminate(run_id)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["state"], "cancelled")
        self.assertTrue((artifact / "scratch" / "started.md").is_file())


class StewardWorkflowTests(unittest.TestCase):
    def test_workflow_reports_dormant_refusal(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ghost = Ghost(GhostConfig(user_name="Jesse", companion_name="Spamuel"))
            supervisor = ManagedRunSupervisor(
                ghost, runtime_root=root / "runtime",
                workflows={
                    "local_task": WorkflowSpec(
                        LOCAL_TASK / "local_task.workflow.md", namespace="local_task",
                    ),
                },
                clock=time.monotonic,
                task_root=root / "tasks",
            )
            steward = Steward(supervisor, clock=time.monotonic)
            receipt = supervisor.start(
                action_id="action-0", workflow_name="local_task", instructions="note",
            )
            environment = GhostBashEnvironment({}, {}, "read")
            runtime = WorkflowRuntime(
                environment=environment, stop_event=threading.Event(),
                tools={
                    "steward_status": steward.status,
                    "steward_answer": steward.answer,
                    "steward_update": steward.update,
                    "steward_terminate": steward.terminate,
                    "steward_resume": steward.resume,
                },
                propagate_python_errors=True,
            )
            runner = WorkflowRunner(STEWARD_MANIFEST, root / "runtime", namespace="steward")
            result = runner.invoke(
                WorkflowRequest("steward-1", {"operation": "status", "run_id": receipt["run_id"]}),
                runtime,
            )
            self.assertEqual(result.status, WorkflowStatus.Completed)
            report = StewardReport.model_validate(result.report)
            self.assertFalse(report.accepted)
            self.assertIn("dormant", report.summary)
            supervisor.join_all(timeout=10)


if __name__ == "__main__":
    unittest.main()
