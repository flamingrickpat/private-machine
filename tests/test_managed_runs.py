"""Managed-run table, worker, and lifecycle operation tests.

Real Ghost checkpoints, real kernel invocations, and real local task files.
No module thread, bus, or model endpoint is required.
"""
from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.records.cycle_state import GhostConfig
from pm_core.ghost.service import Ghost
from runtime.managed_runs import (
    ManagedRunError,
    ManagedRunSupervisor,
    WorkflowSpec,
)
from pm.host.workflow_module import PROJECT_ROOT

LOCAL_TASK = PROJECT_ROOT / "workflows" / "local_task"


def _wait_for(predicate, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class ManagedRunTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ghost = Ghost(
            GhostConfig(user_name="Jesse", companion_name="Spamuel"),
            checkpoint=self.root / "ghost.json",
        )
        self.clock_value = 1000.0
        self.events: list[dict] = []
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
            clock=self._clock,
            event_sink=self.events.append,
        )

    def tearDown(self):
        for run_id in self.supervisor.receipts():
            self.supervisor.terminate(run_id)
        self.supervisor.join_all(timeout=10)
        self._tmp.cleanup()

    def _clock(self) -> float:
        return self.clock_value

    def test_start_status_and_completed_artifacts(self):
        receipt = self.supervisor.start(
            action_id="action-0", workflow_name="local_task",
            parent_correlation="corr", instructions="write a note",
        )
        self.assertEqual(receipt["state"], "accepted")
        self.assertEqual(self.ghost.managed_runs[receipt["run_id"]]["state"], "accepted")
        self.assertTrue(
            _wait_for(lambda: self.supervisor.status(receipt["run_id"])["state"] == "completed"),
            "local task did not complete",
        )
        status = self.supervisor.status(receipt["run_id"])
        self.assertIn("outbox/report.yaml", status["evidence"])
        report = (Path(status["artifact_path"]) / "outbox" / "report.yaml").read_text(encoding="utf-8")
        self.assertIn("write a note", report)
        self.assertEqual(
            [event["event"] for event in self.events],
            ["started", "completed"],
        )

    def test_duplicate_start_returns_the_original_receipt(self):
        first = self.supervisor.start(
            action_id="action-7", workflow_name="local_task", instructions="one",
        )
        second = self.supervisor.start(
            action_id="action-7", workflow_name="local_task", instructions="two",
        )
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(second["instructions"], "one")
        self.assertEqual(len(self.ghost.managed_runs), 1)

    def test_unavailable_workflow_is_an_explicit_failure(self):
        receipt = self.supervisor.start(
            action_id="action-9", workflow_name="minecraft",
        )
        self.assertEqual(receipt["state"], "failed")
        self.assertIn("not managed", receipt["latest_event"])

    def test_update_reaches_the_same_run_and_is_idempotent(self):
        receipt = self.supervisor.start(
            action_id="action-1", workflow_name="local_task",
            instructions="draft", arguments={"wait_for_cancel": True},
        )
        run_id = receipt["run_id"]
        self.assertTrue(_wait_for(lambda: self.supervisor.status(run_id)["state"] == "running"))
        updated = self.supervisor.update(run_id, "final instructions", request_id="req-1")
        self.assertEqual(updated["revision"], 1)
        self.assertEqual(updated["instructions"], "final instructions")
        # A duplicate request id does not bump the revision.
        again = self.supervisor.update(run_id, "final instructions", request_id="req-1")
        self.assertEqual(again["revision"], 1)
        self.assertEqual(self.supervisor.status(run_id)["instructions"], "final instructions")

    def test_terminate_cancels_and_preserves_artifacts(self):
        receipt = self.supervisor.start(
            action_id="action-2", workflow_name="local_task",
            instructions="long job", arguments={"wait_for_cancel": True},
        )
        run_id = receipt["run_id"]
        artifact = Path(receipt["artifact_path"])
        self.assertTrue(
            _wait_for(lambda: (artifact / "scratch" / "started.md").is_file()),
            "local task never wrote its first artifact",
        )
        cancelled = self.supervisor.terminate(run_id)
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertTrue((artifact / "scratch" / "started.md").is_file())
        self.assertFalse((artifact / "outbox" / "report.yaml").exists())

    def test_unchanged_boundaries_do_not_write_checkpoint(self):
        receipt = self.supervisor.start(action_id="boundary-read", workflow_name="local_task",
            instructions="hold", arguments={"wait_for_cancel": True})
        run_id = receipt["run_id"]
        artifact = Path(receipt["artifact_path"]) / "scratch/started.md"
        self.assertTrue(_wait_for(artifact.is_file))
        self.supervisor.instructions_at_boundary(run_id)
        checkpoint = self.root / "ghost.json"
        before = checkpoint.stat().st_mtime_ns
        for _ in range(5):
            self.supervisor.instructions_at_boundary(run_id)
        self.assertEqual(checkpoint.stat().st_mtime_ns, before)
        self.supervisor.instructions_at_boundary(run_id, close=True)
        self.assertFalse(self.supervisor.status(run_id)["updates_open"])
        self.supervisor.terminate(run_id)

    def test_nested_children_keep_configuration_and_separate_folders(self):
        from threading import Event
        from pm_workflows.runtime import WorkflowRuntime
        from runtime.managed_runs import scoped_children
        from pm.host.folder_target import FolderEnvironment

        runtime = scoped_children(WorkflowRuntime(FolderEnvironment(self.root / "scope", "bash"),
            Event(), {}, mcp_servers={"minecraft": {"url": "http://127.0.0.1:6767/mcp"}}), self.root / "scope")
        child = runtime.child_factory("gather")
        nested = child.child_factory("return")
        nested.environment.write_text("scratch/position.txt", "retained")
        self.assertTrue((self.root / "scope/children/gather/children/return/scratch/position.txt").is_file())
        self.assertIs(child.stop_event, runtime.stop_event)
        self.assertIs(nested.tools, runtime.tools)
        self.assertEqual(nested.mcp_servers, runtime.mcp_servers)
        with self.assertRaises(ValueError):
            runtime.child_factory("../escape")

    def test_blocking_question_is_answered_and_resumed(self):
        receipt = self.supervisor.start(
            action_id="action-3", workflow_name="local_task_question",
            instructions="answer me",
        )
        run_id = receipt["run_id"]
        self.assertTrue(
            _wait_for(lambda: self.supervisor.status(run_id)["state"] == "waiting_for_input"),
            "run did not suspend on its question",
        )
        waiting = self.supervisor.status(run_id)
        self.assertTrue(waiting["question_id"])
        self.supervisor.answer(run_id, waiting["question_id"], "the blue artifact")
        self.assertTrue(
            _wait_for(lambda: self.supervisor.status(run_id)["state"] == "completed"),
            "answered run did not complete",
        )
        observations = (Path(waiting["artifact_path"]) / "scratch" / "observations.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("the blue artifact", observations)
        self.assertEqual(self.supervisor.status(run_id)["answer"], "the blue artifact")

    def test_terminate_is_refused_after_a_terminal_state(self):
        receipt = self.supervisor.start(
            action_id="action-4", workflow_name="local_task", instructions="quick",
        )
        run_id = receipt["run_id"]
        self.assertTrue(_wait_for(lambda: self.supervisor.status(run_id)["state"] == "completed"))
        # Terminating a completed run returns its unchanged receipt.
        same = self.supervisor.terminate(run_id)
        self.assertEqual(same["state"], "completed")

    def test_update_is_refused_for_a_terminal_run(self):
        receipt = self.supervisor.start(
            action_id="action-5", workflow_name="local_task", instructions="quick",
        )
        run_id = receipt["run_id"]
        self.assertTrue(_wait_for(lambda: self.supervisor.status(run_id)["state"] == "completed"))
        with self.assertRaises(ManagedRunError):
            self.supervisor.update(run_id, "too late")

    def test_table_survives_a_checkpoint_reopen(self):
        receipt = self.supervisor.start(
            action_id="action-6", workflow_name="local_task", instructions="persist",
        )
        run_id = receipt["run_id"]
        self.assertTrue(_wait_for(lambda: self.supervisor.status(run_id)["state"] == "completed"))
        reopened = Ghost.from_checkpoint(self.root / "ghost.json")
        self.assertEqual(reopened.managed_runs[run_id]["state"], "completed")
        self.assertEqual(reopened.managed_runs[run_id]["action_id"], "action-6")


if __name__ == "__main__":
    unittest.main()
