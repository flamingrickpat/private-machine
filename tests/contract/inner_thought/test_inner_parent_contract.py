"""The full private cycle through the real parent and child kernels.

A deterministic ``llm_complete`` stands in for the model. The real parent
manifest, the real stage manifests, the real kernel, and the real MemoryStore
environment all run, so the manifest wiring, stage payloads, and report
contracts are exercised exactly as the application runs them offline.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path

from pm_workflows.runtime import WorkflowRuntime
from pm.memory.bash_view import MemoryBashEnvironment
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus

from workflows.inner_thought.pipeline import run_cycle
from workflows.inner_thought.schemas import InnerReport, InnerRequest

ROOT = Path(".runtime/a17-parent-contract")
MANIFEST = PROJECT_ROOT / "workflows" / "inner_thought" / "inner_thought.workflow.md"


def request() -> InnerRequest:
    return InnerRequest(
        run_id="contract-1", companion_name="Spamuel", user_name="Jesse",
        state_summary="quiet", tick_id=100, seconds_since_user_input=4000.0,
        main_frame={"tick": 99, "frame_id": 3, "summary": "the story talk", "surprise": 0.0,
                    "gap_seconds": 0.0},
        active_goal_records=[{"knoxel_id": 5, "text": "finish the story", "tick_id": 80,
                              "fulfilment": 0.1, "affective_valence": -0.2, "revision": 1}],
        pending_expectations=[],
        recent_thoughts=[],
        candidates=[{"knoxel_id": 5, "kind": "goal", "text": "finish the story", "revision": 1,
                     "tick_id": 80, "urgency": 0.6, "reason": "unresolved"}],
        trigger_candidates=[{"name": "idle_opportunity",
                             "cause": "the last user input was 4000 seconds ago",
                             "urgency": 0.25, "operations": ["curiosity_spark"],
                             "reason": "idle time"}],
        operations=[],
    )


def deterministic_llm(completions: list[str]):
    """Return an ``llm_complete`` that answers each stage in order."""
    def llm_complete(messages, **generation):
        return completions.pop(0)
    return llm_complete


class FullCycleTests(unittest.TestCase):
    def test_parent_kernel_runs_the_three_stages_and_returns_a_report(self):
        from workflows.inner_thought.stage_runner import build_inner_stage_runner
        ghost = None  # The stage runner only needs an environment for reads.
        from tests.unit.test_design_contracts import small_ghost
        ghost = small_ghost()
        completions = [
            '{"trigger": "idle_opportunity", "operation": "curiosity_spark",'
            ' "temporal_focus": "future", "candidate_ids": [5], "urgency": 0.25,'
            ' "reason": "the quiet invites a private plan"}',
            '{"thought": "I could sketch the next chapter alone.",'
            ' "evidence_ids": [5], "evidence_summary": "the story goal is unfulfilled",'
            ' "interpretation": "the story matters to me", "uncertainty": 0.4,'
            ' "reality": "internal", "alternatives": [],'
            ' "continuation": "summarize", "continuation_reason": "a plan exists now"}',
            '{"goal_changes": [{"change": "add", "goal": "sketch the next chapter",'
            ' "goal_key": "sketch-chapter", "reason": "the quiet invites it", "priority": 0.5}],'
            ' "expectation": {"action": "open", "expectation_id": null,'
            ' "anticipation": "the sketch helps the next talk",'
            ' "expected_outcome": "a chapter sketch exists", "expected_valence": 0.4,'
            ' "uncertainty": 0.5, "goal_ids": [5], "status": "pending",'
            ' "observation_ids": [], "reason": "the plan anticipates progress"},'
            ' "noticing": "", "disclosure": {"mode": "withhold",'
            ' "shareable_summary": "", "reason": "it is still an idea"},'
            ' "summary": "a private sketch plan formed"}',
        ]
        run_stage = build_inner_stage_runner(ghost, deterministic_llm(completions), ROOT)
        environment = MemoryBashEnvironment({}, {}, "read")
        runtime = WorkflowRuntime(environment=environment, stop_event=threading.Event(),
                                  tools={"run_stage": run_stage}, propagate_python_errors=True)
        runner = WorkflowRunner(MANIFEST, ROOT / "parent", namespace="inner_thought",
                                base_dir=PROJECT_ROOT)
        result = runner.invoke(WorkflowRequest("parent-contract", request().model_dump(mode="json")),
                                runtime)
        self.assertEqual(result.status, WorkflowStatus.Completed, result.kernel_summary)
        report = InnerReport.model_validate(result.report)
        self.assertEqual(report.operation, "curiosity_spark")
        self.assertEqual(report.disclosure.mode, "withhold")
        self.assertEqual(report.goal_changes[0].goal, "sketch the next chapter")
        self.assertEqual(completions, [], "every stage must consume exactly one completion")


if __name__ == "__main__":
    unittest.main()
