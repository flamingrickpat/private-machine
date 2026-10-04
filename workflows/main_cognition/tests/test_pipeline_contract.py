"""Direct main-cognition pipeline tests.

No model endpoint, module thread, or bus. The kernel test uses the real kernel
and the real parent Python role with a deterministic ``run_stage`` tool.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path

from pm_workflows.runtime import WorkflowRuntime
from runtime.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from workflows.main_cognition.dispatch import resolve_dispatches, resolve_same_cycle
from workflows.main_cognition.pipeline import STAGE_ORDER, run_cycle, stable_action_id
from workflows.main_cognition.schemas import ActionDispatch, CognitiveReport
from working_target import GhostBashEnvironment

MANIFEST = Path(__file__).resolve().parent.parent / "main_cognition.workflow.md"


def example_request() -> dict:
    return {
        "run_id": "cycle-1",
        "companion_name": "Spamuel",
        "user_name": "Jesse",
        "state_summary": "smug",
        "active_goals": ["keep the bit going"],
        "stimuli": [
            {"knoxel_id": 11, "source": "Jesse", "kind": "user.input",
             "text": "the kitchen is on fire"},
        ],
        "memory_candidates": [{"knoxel_id": 42, "text": "Jesse owns a fire extinguisher."}],
        "capabilities": ["I can communicate via text."],
        "available_workflows": [{"name": "dialogue_story", "description": "reply"}],
    }


def fake_reports() -> dict:
    return {
        "triage": {"items": [{"knoxel_id": 11, "category": "Critical"}]},
        "appraisal": {"feeling": "alarmed", "emotion_delta": {"anxiety": 0.5}},
        "context": {"memories": ["Jesse owns a fire extinguisher."], "candidate_ids": [42]},
        "attention": {"selected_ids": [11], "nominations": []},
        "goals": {"changes": [{"change": "add", "goal": "put out the fire"}]},
        "workspace": {"frame_reason": "fire", "focus_ids": [11], "subjective_experience": "Everything is loud."},
        "action_selection": {
            "actions": [
                {"action_id": "action-0", "action_type": "ToolCall", "workflow_name": "self_management",
                 "content": "start fire response", "arguments": {}, "depends_on": []},
                {"action_type": "Reply", "workflow_name": "dialogue_story",
                 "content": "tell Jesse to move", "arguments": {}, "depends_on": ["action-0"]},
            ],
            "reasoning": "act then speak",
        },
    }


class PipelineTests(unittest.TestCase):
    def test_stages_run_in_the_contract_order_with_cumulative_payloads(self):
        seen: list[str] = []
        payloads: dict[str, dict] = {}

        def run_stage(name: str, payload: dict) -> dict:
            seen.append(name)
            payloads[name] = payload
            report = dict(fake_reports()[name])
            report["summary"] = f"{name} done"
            return report

        report = run_cycle(example_request(), run_stage)
        self.assertEqual(tuple(seen), STAGE_ORDER)
        # Later stages see earlier stage reports.
        self.assertIn("triage", payloads["appraisal"])
        self.assertIn("appraisal", payloads["attention"])
        self.assertIn("attention", payloads["workspace"])
        self.assertEqual(len(report.receipts), 7)
        expected_ids = [
            intent.get("action_id") or stable_action_id(intent, index)
            for index, intent in enumerate(fake_reports()["action_selection"]["actions"])
        ]
        self.assertEqual([d.action_id for d in report.dispatches], expected_ids)
        self.assertEqual([d.status for d in report.dispatches], ["dispatched", "dispatched"])

    def test_dependency_on_an_unknown_action_is_rejected(self):
        def run_stage(name: str, payload: dict) -> dict:
            report = dict(fake_reports()[name])
            if name == "action_selection":
                report = {
                    "actions": [
                        {"action_type": "Reply", "workflow_name": "dialogue_story",
                         "content": "claim it started", "arguments": {},
                         "depends_on": ["action-99"]},
                    ],
                    "reasoning": "",
                }
            return report

        report = run_cycle(example_request(), run_stage)
        self.assertEqual(report.dispatches[0].status, "rejected")
        self.assertIn("action-99", report.dispatches[0].rejection_reason)


class DispatchTests(unittest.TestCase):
    def test_external_receipts_release_and_reject(self):
        dispatches = [
            ActionDispatch(action_id="start", workflow_name="self_management",
                           action_type="ToolCall", content="start"),
            ActionDispatch(action_id="claim", workflow_name="dialogue_story",
                           action_type="Reply", content="it started", depends_on=["start"]),
        ]
        pending = resolve_dispatches(dispatches, {})
        self.assertEqual([d.status for d in pending], ["dispatched", "rejected"])
        completed = resolve_dispatches(dispatches, {"start": "completed"})
        self.assertEqual([d.status for d in completed], ["dispatched", "dispatched"])
        failed = resolve_dispatches(dispatches, {"start": "failed"})
        self.assertEqual(failed[1].status, "rejected")

    def test_same_cycle_dependency_ordering(self):
        later = ActionDispatch(action_id="action-1", workflow_name="dialogue_story",
                               action_type="Reply", content="it started", depends_on=["action-0"])
        earlier = ActionDispatch(action_id="action-0", workflow_name="self_management",
                                 action_type="ToolCall", content="start")
        resolved = resolve_same_cycle([later, earlier])
        # action-1 appears before its dependency in the list, so it is rejected.
        self.assertEqual(resolved[0].status, "rejected")
        self.assertEqual(resolved[1].status, "dispatched")


class KernelTests(unittest.TestCase):
    def test_parent_role_runs_through_the_real_kernel(self):
        def run_stage(name: str, payload: dict) -> dict:
            report = dict(fake_reports()[name])
            report["summary"] = f"{name} done"
            return report

        environment = GhostBashEnvironment({}, {}, "read")
        runtime = WorkflowRuntime(
            environment=environment, stop_event=threading.Event(),
            tools={"run_stage": run_stage}, propagate_python_errors=True,
        )
        runner = WorkflowRunner(
            MANIFEST, Path(".runtime/w5-direct-tests"), namespace="main_cognition",
            base_dir=PROJECT_ROOT,
        )
        result = runner.invoke(WorkflowRequest("cycle-kernel-1", example_request()), runtime)
        self.assertEqual(result.status, WorkflowStatus.Completed)
        report = CognitiveReport.model_validate(result.report)
        self.assertEqual(len(report.receipts), 7)
        first = fake_reports()["action_selection"]["actions"][0]
        self.assertEqual(report.dispatches[0].action_id, first.get("action_id") or stable_action_id(first, 0))


if __name__ == "__main__":
    unittest.main()
