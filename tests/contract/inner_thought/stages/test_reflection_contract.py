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
from tests.contract.inner_thought.stages.test_inner_stages_contract import ContractModel
from pm.context_render import Role, Turn
from pm.llm.structured import complete_json
from workflows.reflection.experiments import _validate_outcome
from workflows.reflection.schemas import ObservationGrant, ReflectionDecision


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
            tools={"llm_complete": ContractModel(complete)}, propagate_python_errors=True)
        manifest = PROJECT_ROOT / "workflows/inner_thought/stages/reflection/reflection.workflow.md"
        with TemporaryDirectory() as directory:
            runner = WorkflowRunner(manifest, Path(directory), namespace="reflection", base_dir=PROJECT_ROOT)
            result = runner.invoke(WorkflowRequest("reflection-contract", {
                "reflection": {"records": [], "episodes": [], "experiments": []}}), runtime)
            self.assertIs(result.status, WorkflowStatus.Completed)
            self.assertEqual(result.report["action"], "none")
            self.assertEqual(len(calls), 3)
            self.assertNotEqual(str(calls[0][0]), str(calls[1][0]))
            self.assertIn("Monitor:", calls[1][1].content.text)
            self.assertIn("Assessment:", calls[2][1].content.text)
            self.assertTrue(list(Path(directory).rglob("assessment.json")))

    def test_captured_outcome_error_receives_structured_repair_before_acceptance(self):
        """The historical capture contains an outcome grant, not a complete experiment plan."""
        capture = json.loads((PROJECT_ROOT / "test_data/captures/alpha/reflection-outcome-rejection.json").read_text(encoding="utf-8"))
        responses = iter([
            capture["proposal"],
            {**capture["proposal"], "observation_ids": [11226, 11247]},
        ])
        calls = []
        def complete(messages, **generation):
            calls.append(messages)
            return json.dumps(next(responses))
        grant = ObservationGrant.model_validate(capture["payload"]["experiments"][0]["observation_grant"])

        def validate(payload: dict[str, object], raw: str) -> ReflectionDecision:
            decision = ReflectionDecision.model_validate(payload)
            _validate_outcome(decision, grant)
            return decision

        model = ContractModel(complete)
        result = complete_json(model, [Turn(Role.User, json.dumps(capture["payload"]))],
                               ReflectionDecision, measure=model.measure_context, validate=validate)
        self.assertEqual(result.result, "unresolved")
        self.assertEqual(result.observation_ids, [11226, 11247])
        self.assertEqual(len(calls), 2)
        self.assertIn("Allowed observation IDs", str(calls[-1][-1]))
