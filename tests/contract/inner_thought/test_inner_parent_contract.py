"""The full private cycle through the real parent and child kernels.

A deterministic ``llm_complete`` stands in for the model. The real parent
manifest, the real stage manifests, the real kernel, and the real MemoryStore
environment all run, so the manifest wiring, stage payloads, and report
contracts are exercised exactly as the application runs them offline.
"""
from __future__ import annotations

import json
import threading
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path

from pm_workflows.runtime import WorkflowRuntime
from pm.bus.bus import BusSystem
from pm.memory.bash_view import MemoryBashEnvironment, MemoryBashTarget
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus

from tests.support import ManualClock, Recorder
from tests.unit.test_design_contracts import small_ghost
from workflows.inner_thought.host import PrivateThoughtLoop, build_inner_thought_tools
from workflows.inner_thought.pipeline import citable_ids
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
        environment = MemoryBashTarget(ghost, "read").open("parent-contract")
        tools = build_inner_thought_tools(ghost, ROOT, None)(environment, None)
        runtime = WorkflowRuntime(environment=environment, stop_event=threading.Event(),
                                  tools={**tools, "run_stage": run_stage}, propagate_python_errors=True)
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


class ModelProposalBoundaryTests(unittest.TestCase):
    def _module(self, root, payload, llm_complete):
        class FrozenInputLoop(PrivateThoughtLoop):
            def prepare_input(self, batch):
                # Keep the host's commit context. Replace only the model input
                # with the captured boundary projection, not a live receipt.
                super().prepare_input(batch)
                return {**payload, "run_id": self.invocation_id}

        ghost = small_ghost()
        bus = BusSystem()
        errors = []
        runtime_root = Path(root) / "runtime"
        module = FrozenInputLoop(
            "inner_thought", bus, errors.append, ManualClock(), None,
            working_target=MemoryBashTarget(ghost, "read"), workflow=MANIFEST,
            tools_factory=build_inner_thought_tools(ghost, runtime_root, llm_complete),
            runtime_root=runtime_root, ghost=ghost,
        )
        return module, ghost, Recorder(bus), errors

    def test_captured_inspection_citation_rejection_crosses_parent_and_child_kernels(self):
        capture = json.loads((PROJECT_ROOT / "test_data/captures/alpha/private-thought-citation-rejection.json").read_text(encoding="utf-8"))
        payload = InnerRequest(
            candidates=capture["candidates"], inspection=capture["inspection"],
            trigger_candidates=[{"name": "new_main_frame", "cause": "constructed routing for the captured proposal",
                                 "operations": ["narrative_stitching"]}],
        ).model_dump(mode="json")
        known = citable_ids(InnerRequest.model_validate(payload))
        self.assertEqual([item for item in capture["thought"]["evidence_ids"] if item not in known],
                         [812, 11295])
        completions = [json.dumps({"trigger": "new_main_frame", "operation": "narrative_stitching"}),
                       json.dumps(capture["thought"])]
        with TemporaryDirectory() as tmp:
            module, ghost, events, errors = self._module(tmp, payload, deterministic_llm(completions))
            before = ghost.state_dict()
            module.handle_batch([])
            self.assertEqual(completions, [])
            self.assertEqual(ghost.state_dict(), before)
            self.assertEqual(errors, [])
            self.assertIsNone(module.last_result)
            self.assertEqual([(item.kind, item.topic) for item in events.events],
                             [("telemetry", "thought.rejected")])
            rejection = events.events[0].payload
            self.assertEqual(rejection["error_type"], "ModelProposalError")
            self.assertEqual(rejection["invocation_id"], module.invocation_id)
            self.assertEqual(rejection["reason"],
                             "the thought stage cited ids outside the frozen view: [812, 11295]")
            diagnostic = json.loads(Path(rejection["diagnostic_path"]).read_text(encoding="utf-8"))
            self.assertIn("python_driver", diagnostic["traceback"])
            self.assertIn("run_cycle", diagnostic["traceback"])
            self.assertEqual(len(list((module.runtime_root / "thought" / "kernel").glob("*/results/*.json"))), 1)
            self.assertFalse((module.runtime_root / "integrate").exists())

    def test_untyped_errors_from_a_child_kernel_still_propagate(self):
        reason = "the thought stage cited ids outside the frozen view: [812, 11295]"
        for error in (ValueError(reason), TypeError("broken completion tool")):
            with self.subTest(error=type(error).__name__), TemporaryDirectory() as tmp:
                calls = []

                def llm_complete(messages, **generation):
                    calls.append(messages)
                    if len(calls) == 1:
                        return '{"operation": "curiosity_spark"}'
                    raise error

                module, ghost, events, errors = self._module(tmp, request().model_dump(mode="json"), llm_complete)
                before = ghost.state_dict()
                with self.assertRaises(type(error)) as raised:
                    module.handle_batch([])
                self.assertIs(raised.exception, error)
                self.assertEqual(len(calls), 2)
                self.assertEqual(ghost.state_dict(), before)
                self.assertEqual(events.events, [])
                self.assertEqual(list(module.runtime_root.rglob("model-proposal-rejection.json")), [])

    def test_integration_keeps_one_repair_then_rejects_the_cycle(self):
        completions = [
            '{"operation": "curiosity_spark"}',
            '{"thought": "I want to reread the draft.", "reality": "internal", "evidence_ids": [5]}',
            '{"expectation": {"action": "resolve", "expectation_id": 999, "observation_ids": [5]}}',
            '{"expectation": {"action": "resolve", "expectation_id": 999, "observation_ids": [5]}}',
        ]
        with TemporaryDirectory() as tmp:
            module, ghost, events, errors = self._module(
                tmp, request().model_dump(mode="json"), deterministic_llm(completions))
            before = ghost.state_dict()
            module.handle_batch([])
            self.assertEqual(completions, [])
            self.assertEqual(ghost.state_dict(), before)
            self.assertEqual([(item.kind, item.topic) for item in events.events],
                             [("telemetry", "thought.rejected")])
            self.assertEqual(events.events[0].payload["error_type"], "IntegrationError")
            self.assertEqual(len(list(module.runtime_root.rglob("integration-rejection-*.json"))), 2)
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
