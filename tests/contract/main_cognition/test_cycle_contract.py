"""Contract tests of the main cognition manifest through the real kernel.

No model endpoint, module thread, or bus. The real kernel runs every phase of
``main_cognition.workflow.md`` with a scripted ``llm_complete`` tool.
"""
from __future__ import annotations

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from pm_workflows.runtime import WorkflowRuntime
from pm.memory.bash_view import MemoryBashEnvironment
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus
from tests.unit.workflows.main_cognition.cycle_support import ScriptedModel
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.dispatch import resolve_dispatches, resolve_same_cycle
from workflows.main_cognition.phases.route import stable_action_id
from workflows.main_cognition.schemas import ActionDispatch, ActionIntent
from workflows.main_cognition.steps import PHASES

MANIFEST = PROJECT_ROOT / "workflows/main_cognition/main_cognition.workflow.md"


def example_request() -> dict:
    return CycleRequest(
        run_id="cycle-1", companion_name="Companion", user_name="User", state_summary="smug",
        active_goals=["keep the bit going"],
        stimuli=[{"knoxel_id": 11, "source": "user_input", "kind": "user.input", "text": "the kitchen is on fire",
                  "correlation_id": "c-1"}],
        memory_candidates=[{"knoxel_id": 42, "text": "User owns a fire extinguisher."}],
        # The host supplies a view of every record that the CSM can hold.
        records={11: {"record_id": 11, "kind": "Feature", "source": "User", "content": "the kitchen is on fire"},
                 42: {"record_id": 42, "kind": "DeclarativeFactKnoxel", "content": "User owns a fire extinguisher."}},
        capabilities=["I can communicate via text."],
        available_workflows=[{"name": "dialogue_story", "description": "reply"},
                             {"name": "self_management", "description": "runs"}],
        correlation_id="c-1",
    ).model_dump(mode="json")


ACTIONS = [
    {"action_id": "action-0", "action_type": "ToolCall", "workflow_name": "self_management",
     "content": "start fire response", "depends_on": []},
    {"action_type": "Reply", "workflow_name": "dialogue_story",
     "content": "tell the user to get out", "depends_on": ["action-0"]},
]


def answers(actions=ACTIONS) -> dict:
    return {
        "triage": {"items": [{"knoxel_id": 11, "category": "Critical"}]},
        "appraise": {"feeling": "alarmed", "emotion_delta": {"fear": 0.5}, "evidence_ids": [11]},
        "retrieve": {"candidate_ids": [42]},
        "goals": {"changes": [{"change": "add", "goal": "put out the fire"}]},
        "experience": {"frame_reason": "fire", "focus_ids": [11], "subjective_experience": "Everything is loud."},
        "decide": {"actions": actions, "reasoning": "act then speak"},
    }


def run_cycle(model: ScriptedModel, run_id: str) -> CycleWork:
    environment = MemoryBashEnvironment({}, {}, "read")
    runtime = WorkflowRuntime(environment=environment, stop_event=threading.Event(),
                              tools=model.tools(), propagate_python_errors=True)
    with TemporaryDirectory() as tmp:
        runner = WorkflowRunner(MANIFEST, Path(tmp), namespace="main_cognition", base_dir=PROJECT_ROOT)
        result = runner.invoke(WorkflowRequest(run_id, example_request()), runtime)
    assert result.status is WorkflowStatus.Completed, result.kernel_summary
    return CycleWork.model_validate(result.report)


class CycleTests(unittest.TestCase):
    def test_phases_run_in_manifest_order_and_share_one_typed_state(self):
        model = ScriptedModel(answers())
        work = run_cycle(model, "cycle-contract-1")
        self.assertEqual([item.name for item in work.phases], ["intake", *PHASES])
        # Later phases read earlier results: the retrieve prompt shows the appraisal.
        retrieve = model.messages[model.calls.index("retrieve")]
        self.assertIn("alarmed", retrieve[-1].content.text)
        self.assertEqual(work.selected_memories[0].knoxel_id, 42)
        self.assertEqual(work.context.memories, ["User owns a fire extinguisher."])
        # The appraise phase set the EMA mental state and the appraisal affect.
        self.assertEqual(len(work.affect.delta_vec), len(work.state.mental_state.to_list()))
        self.assertTrue(work.affect.felt.startswith("Companion feels"))
        self.assertEqual(work.state_summary, work.affect.felt)
        self.assertEqual(work.state.stimulus_groups[0].category, "Critical")
        self.assertEqual(len(work.state.decisions), 2)
        expected_ids = [item.action_id or stable_action_id(item, index)
                        for index, item in enumerate(ActionIntent.model_validate(row) for row in ACTIONS)]
        self.assertEqual([item.action_id for item in work.dispatches], expected_ids)
        self.assertEqual([item.status for item in work.dispatches], ["dispatched", "dispatched"])

    def test_without_public_context_no_grounding_turn_is_inserted(self):
        model = ScriptedModel(answers())
        run_cycle(model, "cycle-contract-2")
        for turns in model.messages:
            self.assertFalse(any("Private context." in turn.content.text for turn in turns))

    def test_dependency_on_an_unknown_action_is_rejected(self):
        model = ScriptedModel(answers([{"action_type": "Reply", "workflow_name": "dialogue_story",
                                        "content": "claim it started", "depends_on": ["action-99"]}]))
        work = run_cycle(model, "cycle-contract-3")
        self.assertEqual(work.dispatches[0].status, "rejected")
        self.assertIn("action-99", work.dispatches[0].rejection_reason)


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


if __name__ == "__main__":
    unittest.main()
