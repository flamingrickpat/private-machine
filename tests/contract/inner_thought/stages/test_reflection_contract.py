"""Run the reflection child through the real kernel with explicit contract data.

These boundary checks use fixed completions. They establish no model behavior.
The separate A19 live gate performs actual inference with the full canon.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

from pm_workflows.runtime import WorkflowRuntime
from pm.memory.bash_view import MemoryBashEnvironment
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus


class ReflectionContractTests(unittest.TestCase):
    def test_monitor_assess_and_guide_receive_distinct_tasks(self):
        responses = iter([
            {"finding": "", "relevant": False, "reason": "No relevant episode is retained."},
            {"reason": "No later observation supports a prediction."},
            {"action": "none", "reason": "Wait for relevant evidence."},
        ])
        calls = []
        def complete(messages, **generation):
            calls.append(messages)
            return json.dumps(next(responses))
        environment = MemoryBashEnvironment({}, {}, "read")
        runtime = WorkflowRuntime(environment=environment, stop_event=Event(),
            tools={"llm_complete": complete}, propagate_python_errors=True)
        manifest = PROJECT_ROOT / "workflows/inner_thought/stages/reflection/reflection.workflow.md"
        with TemporaryDirectory() as directory:
            runner = WorkflowRunner(manifest, Path(directory), namespace="reflection", base_dir=PROJECT_ROOT)
            result = runner.invoke(WorkflowRequest("reflection-contract", {
                "reflection": {"records": [], "episodes": [], "experiments": []}}), runtime)
            self.assertIs(result.status, WorkflowStatus.Completed)
            self.assertEqual(result.report["action"], "none")
            self.assertEqual(len(calls), 3)
            self.assertNotEqual(str(calls[0][0]), str(calls[1][0]))
            self.assertIn("monitor", str(calls[1][1]))
            self.assertIn("assessment", str(calls[2][1]))
            self.assertTrue(list(Path(directory).rglob("assessment.json")))

    def test_captured_outcome_error_receives_model_repair_before_acceptance(self):
        capture = json.loads((PROJECT_ROOT / "test_data/captures/alpha/reflection-outcome-rejection.json").read_text(encoding="utf-8"))
        responses = iter([
            {"relevant": True, "reason": "Two actual later decisions exist."},
            {"reason": "The fixture does not discriminate the prediction.", "result": "unresolved"},
            capture["proposal"],
            {**capture["proposal"], "observation_ids": [11226, 11247]},
        ])
        calls = []
        def complete(messages, **generation):
            calls.append(messages)
            return json.dumps(next(responses))
        runtime = WorkflowRuntime(environment=MemoryBashEnvironment({}, {}, "read"), stop_event=Event(),
            tools={"llm_complete": complete}, propagate_python_errors=True)
        manifest = PROJECT_ROOT / "workflows/inner_thought/stages/reflection/reflection.workflow.md"
        with TemporaryDirectory() as directory:
            runner = WorkflowRunner(manifest, Path(directory), namespace="reflection", base_dir=PROJECT_ROOT)
            result = runner.invoke(WorkflowRequest("captured-repair", {"reflection": capture["payload"]}), runtime)
            self.assertIs(result.status, WorkflowStatus.Completed)
            self.assertEqual(result.report["result"], "unresolved")
            self.assertEqual(result.report["observation_ids"], [11226, 11247])
            self.assertEqual(len(calls), 4)
            self.assertIn("Allowed observation IDs", str(calls[-1][1]))
            self.assertTrue(list(Path(directory).rglob("reflection-rejection-0.json")))
