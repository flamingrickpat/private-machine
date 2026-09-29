"""Evidence-backed consolidation tests.

Real Ghost, real task artifacts, real kernel invocation. No model endpoint.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm_workflows.runtime import WorkflowRuntime

from pm_core.ghost.service import Ghost
from pm_core.model.ghost_state import GhostConfig
from pm_core.model.knoxel_enums import FeatureType
from runtime.consolidation import Consolidator
from runtime.managed_runs import ManagedRunSupervisor, WorkflowSpec
from runtime.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.memory_consolidation.schemas import ConsolidationReport
from working_target import GhostBashEnvironment

LOCAL_TASK = PROJECT_ROOT / "workflows" / "local_task"
CONSOLIDATION_MANIFEST = (
    PROJECT_ROOT / "workflows" / "memory_consolidation" / "memory_consolidation.workflow.md"
)


def _wait_for(predicate, timeout: float = 15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class ConsolidatorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ghost = Ghost(
            GhostConfig(user_name="Jesse", companion_name="Spamuel"),
            checkpoint=self.root / "ghost.json",
        )
        self.clock_value = 5.0
        self.supervisor = ManagedRunSupervisor(
            self.ghost, runtime_root=self.root / "runtime",
            workflows={
                "local_task": WorkflowSpec(
                    LOCAL_TASK / "local_task.workflow.md", namespace="local_task",
                ),
            },
            clock=self._clock, task_root=self.root / "tasks",
        )
        self.consolidator = Consolidator(self.ghost, clock=self._clock)

    def tearDown(self):
        self.supervisor.join_all(timeout=10)
        self._tmp.cleanup()

    def _clock(self) -> float:
        return self.clock_value

    def _completed_run(self, action_id: str = "action-0") -> dict:
        receipt = self.supervisor.start(
            action_id=action_id, workflow_name="local_task",
            instructions="record the observation",
        )
        run_id = receipt["run_id"]
        self.assertTrue(_wait_for(lambda: self.supervisor.status(run_id)["state"] == "completed"))
        return self.supervisor.status(run_id)

    def test_consolidation_commits_evidence_backed_memory(self):
        run = self._completed_run()
        receipt = self.consolidator.consolidate(
            run["run_id"], run["artifact_path"], source_event_ids=[101],
        )
        self.assertEqual(receipt["state"], "consolidated")
        self.assertTrue(receipt["memory_ids"])
        self.assertEqual(receipt["source_event_ids"], [101])
        paths = {item["path"] for item in receipt["evidence"]}
        self.assertIn("outbox/report.yaml", paths)
        features = [self.ghost.get_knoxel_by_id(mid) for mid in receipt["memory_ids"]]
        outcome = next(f for f in features if f.metadata.get("kind") == "outcome")
        self.assertIn("local task finished", outcome.content)
        self.assertEqual(outcome.metadata["run_id"], run["run_id"])
        self.assertIn("outbox/report.yaml", outcome.metadata["evidence"])
        self.assertIn(FeatureType.InnerMonologue, [f.feature_type for f in features])

    def test_repeat_consolidation_does_not_duplicate_memory(self):
        run = self._completed_run("action-1")
        first = self.consolidator.consolidate(run["run_id"], run["artifact_path"])
        second = self.consolidator.consolidate(run["run_id"], run["artifact_path"])
        self.assertEqual(first["key"], second["key"])
        self.assertEqual(first["memory_ids"], second["memory_ids"])
        self.assertEqual(len(self.ghost.consolidations), 1)
        # The checkpoint persisted the ledger.
        reopened = Ghost.from_checkpoint(self.root / "ghost.json")
        self.assertIn(first["key"], reopened.consolidations)

    def test_changed_artifact_produces_a_new_receipt(self):
        run = self._completed_run("action-2")
        artifact = Path(run["artifact_path"])
        first = self.consolidator.consolidate(run["run_id"], run["artifact_path"])
        (artifact / "scratch" / "observations.md").write_text(
            "# Observations\n\nchanged\n", encoding="utf-8",
        )
        second = self.consolidator.consolidate(run["run_id"], run["artifact_path"])
        self.assertNotEqual(first["key"], second["key"])
        self.assertEqual(len(self.ghost.consolidations), 2)
        # The first memories are untouched.
        self.assertNotEqual(first["memory_ids"], second["memory_ids"])

    def test_missing_artifact_path_still_returns_a_receipt(self):
        receipt = self.consolidator.consolidate("ghost-run", self.root / "nowhere")
        self.assertEqual(receipt["memory_ids"], [])
        self.assertEqual(receipt["evidence"], [])


class ConsolidationWorkflowTests(unittest.TestCase):
    def test_workflow_converts_artifacts_and_reuses_the_receipt(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ghost = Ghost(GhostConfig(user_name="Jesse", companion_name="Spamuel"))
            artifact = root / "artifacts"
            (artifact / "outbox").mkdir(parents=True)
            (artifact / "scratch").mkdir(parents=True)
            (artifact / "outbox" / "report.yaml").write_text(
                "summary: rebuilt the fence\n", encoding="utf-8",
            )
            (artifact / "scratch" / "result.md").write_text(
                "# Result\n\nfence is upright\n", encoding="utf-8",
            )
            consolidator = Consolidator(ghost, clock=time.monotonic)

            def invoke(run_id: str) -> ConsolidationReport:
                environment = GhostBashEnvironment({}, {}, "read")
                runtime = WorkflowRuntime(
                    environment=environment, stop_event=threading.Event(),
                    tools={
                        "consolidated_artifacts": consolidator.consolidate,
                        "list_consolidations": lambda: list(ghost.consolidations.values()),
                    },
                    propagate_python_errors=True,
                )
                runner = WorkflowRunner(
                    CONSOLIDATION_MANIFEST, root / "runtime", namespace="consolidation",
                )
                result = runner.invoke(WorkflowRequest(run_id, {
                    "operation": "consolidate",
                    "run_id": "run-x",
                    "artifact_path": str(artifact),
                }), runtime)
                self.assertEqual(result.status, WorkflowStatus.Completed)
                return ConsolidationReport.model_validate(result.report)

            first = invoke("c1")
            second = invoke("c2")
            self.assertTrue(first.accepted)
            self.assertEqual(first.receipt["memory_ids"], second.receipt["memory_ids"])
            self.assertEqual(len(ghost.consolidations), 1)


if __name__ == "__main__":
    unittest.main()
