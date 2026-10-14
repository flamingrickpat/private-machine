"""Replay captured model responses through the real context workflow boundary."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

import yaml
from pm.host.folder_target import FolderEnvironment
from pm.runs.workflow_runner import WorkflowRequest, WorkflowRunner, WorkflowStatus
from pm_workflows.runtime import WorkflowRuntime
from workflows.main_cognition.schemas import ContextResult
from workflows.main_cognition.stages.context.steps import validate_selection

ROOT = Path(__file__).resolve().parents[4]


class ContextRecoveryTests(unittest.TestCase):
    def test_failed_id_repair_keeps_only_real_candidates_and_completes_the_stage(self):
        capture = json.loads((ROOT / "test_data/captures/alpha/context-invalid-memory-selection.json").read_text(encoding="utf-8"))
        responses = iter(row["choices"][0]["message"]["content"] for row in capture["model_results"])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            environment = FolderEnvironment(root / "workspace", "bash")
            runtime = WorkflowRuntime(environment, Event(),
                {"llm_complete": lambda *args, **kwargs: next(responses)}, propagate_python_errors=True)
            runner = WorkflowRunner(ROOT / "workflows/main_cognition/stages/context/context.workflow.md", root / "kernel")
            result = runner.invoke(WorkflowRequest("captured-context-recovery", capture["request"]), runtime)
            self.assertEqual(result.status, WorkflowStatus.Completed)
            report = ContextResult.model_validate(result.report)
            candidates = capture["request"]["memory_candidates"]
            validate_selection(report, candidates)
            receipt = json.loads(environment.read_text("outbox/selection-recovery.json"))
            self.assertTrue(receipt["dropped_ids"])
            self.assertFalse(set(receipt["dropped_ids"]) & set(report.candidate_ids))
            sources = {row["knoxel_id"]: row["text"] for row in candidates}
            self.assertEqual(report.memories, [sources[key] for key in report.candidate_ids])
            self.assertEqual(yaml.safe_load(environment.read_text("inbox/request.yaml")), capture["request"])
