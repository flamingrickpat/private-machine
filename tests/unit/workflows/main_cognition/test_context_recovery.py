"""Replay captured model responses through the retrieve phase boundary."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from pm.host.folder_target import FolderEnvironment
from pm_workflows.runtime import WorkflowRuntime
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.phases import retrieve
from workflows.main_cognition.phases.model import phase_prompts

ROOT = Path(__file__).resolve().parents[4]


class _Context:
    def __init__(self, runtime):
        self.runtime = runtime


class ContextRecoveryTests(unittest.TestCase):
    def test_failed_id_repair_keeps_only_real_candidates_and_completes_the_phase(self):
        capture = json.loads((ROOT / "test_data/captures/alpha/context-invalid-memory-selection.json").read_text(encoding="utf-8"))
        responses = iter(row["choices"][0]["message"]["content"] for row in capture["model_results"])
        with TemporaryDirectory() as directory:
            environment = FolderEnvironment(Path(directory) / "workspace", "bash")
            runtime = WorkflowRuntime(environment, Event(),
                {"llm_complete": lambda *args, **kwargs: next(responses)}, propagate_python_errors=True)
            work = CycleWork(request=CycleRequest.model_validate(capture["request"]))
            work = retrieve.run(_Context(runtime), work)
            candidates = work.request.memory_candidates
            retrieve.validate_selection(work.context, candidates)
            receipt = json.loads(environment.read_text("outbox/selection-recovery.json"))
            self.assertTrue(receipt["dropped_ids"])
            self.assertFalse(set(receipt["dropped_ids"]) & set(work.context.candidate_ids))
            sources = {row.knoxel_id: row.text for row in candidates}
            self.assertEqual(work.context.memories, [sources[key] for key in work.context.candidate_ids])
            self.assertEqual([item.knoxel_id for item in work.selected_memories], work.context.candidate_ids)
            self.assertTrue(phase_prompts("retrieve").user_template)
