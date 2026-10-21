"""Evidence-backed consolidation tests.

Real MemoryStore, real task artifacts, real kernel invocation. No model endpoint.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm_workflows.runtime import WorkflowRuntime

from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType
from workflows.memory_consolidation.artifact_consolidation import Consolidator
from pm.runs.supervisor import ManagedRunSupervisor
from pm.runs.workflow_spec import WorkflowSpec
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.memory_consolidation.schemas import ConsolidationReport
from pm.memory.bash_view import MemoryBashEnvironment
from pm.records.base import persisted_type_name
from pm.records.media import record_media

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
        self.ghost = MemoryStore(
            PersonaConfig(user_name="Jesse", companion_name="Spamuel"),
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
        reopened = MemoryStore.from_checkpoint(self.root / "ghost.json")
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

    def test_spatial_records_and_screenshot_evidence(self):
        artifact = self.root / "world-artifacts"
        (artifact / "outbox").mkdir(parents=True)
        (artifact / "screenshots").mkdir(parents=True)
        (artifact / "outbox" / "report.yaml").write_text(
            "summary: chopped one oak log\n", encoding="utf-8",
        )
        (artifact / "outbox" / "world.yaml").write_text(
            "places:\n"
            "  - place_key: spawn\n"
            "    position: {x: 0, y: 1, z: 0}\n"
            "    dimension: overworld\n"
            "    landmarks: [flat area]\n"
            "    confidence: 0.9\n"
            "objects:\n"
            "  - object_key: oak_log_1\n"
            "    object_kind: oak_log\n"
            "    position: {x: 5, y: 1, z: 3}\n"
            "    confidence: 0.8\n"
            "episodes:\n"
            "  - body_id: big_floppa\n"
            "    goal: chop one oak log\n"
            "    outcome: success\n"
            "    action_count: 3\n"
            "    images: [screenshots/step-01.png]\n",
            encoding="utf-8",
        )
        png = b"\x89PNG\r\n\x1a\n" + b"fake-pixels"
        (artifact / "screenshots" / "step-01.png").write_bytes(png)

        receipt = self.consolidator.consolidate("world-run", artifact)
        self.assertTrue(receipt["memory_ids"])
        self.assertEqual(len(receipt["image_evidence"]), 1)
        self.assertEqual(receipt["image_evidence"][0]["media_type"], "image/png")
        self.assertEqual(
            receipt["image_evidence"][0]["sha256"],
            __import__("hashlib").sha256(png).hexdigest(),
        )
        knoxels = [self.ghost.get_knoxel_by_id(mid) for mid in receipt["memory_ids"]]
        kinds = {persisted_type_name(type(k)) for k in knoxels}
        self.assertIn("WorldPlaceKnoxel", kinds)
        self.assertIn("WorldObjectKnoxel", kinds)
        self.assertIn("BodyEpisodeKnoxel", kinds)
        episode = next(k for k in knoxels if persisted_type_name(type(k)) == "BodyEpisodeKnoxel")
        link = episode.evidence["images"][0]
        self.assertEqual(link["media_type"], "image/png")
        self.assertEqual(link["source_artifact"], "screenshots/step-01.png")
        self.assertNotIn("base64", link)
        # The pixels live in the shared record media, with tool provenance.
        image = record_media(episode).images[0]
        self.assertEqual((image.id, image.data(), image.provenance.source, image.provenance.artifact_path),
                         (link["image_id"], png, "tool", "screenshots/step-01.png"))
        self.assertTrue(episode.content.endswith(image.tag))
        # Base64 never enters textual content or a text embedding.
        for knoxel in knoxels:
            self.assertNotIn(image.data_b64, str(getattr(knoxel, "content", "")))
            self.assertEqual(getattr(knoxel, "embedding", []) or [], [])
        # A repeated request reuses the same receipt and memory ids.
        again = self.consolidator.consolidate("world-run", artifact)
        self.assertEqual(again["memory_ids"], receipt["memory_ids"])

    def test_changed_screenshot_creates_a_new_receipt(self):
        artifact = self.root / "image-artifacts"
        (artifact / "outbox").mkdir(parents=True)
        (artifact / "screenshots").mkdir(parents=True)
        (artifact / "outbox" / "report.yaml").write_text("summary: observed\n", encoding="utf-8")
        shot = artifact / "screenshots" / "frame.png"
        shot.write_bytes(b"\x89PNG\r\n\x1a\nfirst")
        first = self.consolidator.consolidate("image-run", artifact)
        shot.write_bytes(b"\x89PNG\r\n\x1a\nsecond")
        second = self.consolidator.consolidate("image-run", artifact)
        self.assertNotEqual(first["key"], second["key"])
        self.assertEqual(len(self.ghost.consolidations), 2)

    def test_failed_consolidation_preserves_the_task_folder(self):
        artifact = self.root / "preserved"
        (artifact / "outbox").mkdir(parents=True)
        (artifact / "outbox" / "world.yaml").write_text(
            "places: [this is not a mapping\n", encoding="utf-8",
        )
        receipt = self.consolidator.consolidate("bad-run", artifact)
        self.assertEqual(receipt["state"], "consolidated")
        self.assertTrue(artifact.is_dir())
        self.assertTrue((artifact / "outbox" / "world.yaml").is_file())


class ConsolidationWorkflowTests(unittest.TestCase):
    def test_workflow_converts_artifacts_and_reuses_the_receipt(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
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
                environment = MemoryBashEnvironment({}, {}, "read")
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
