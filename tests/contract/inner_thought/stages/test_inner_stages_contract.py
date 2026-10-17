"""Direct private stage tests.

No model endpoint, module thread, or bus. The kernel tests use the real kernel
and the real Python roles with a deterministic ``llm_complete`` tool.
"""
from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path

from pm_workflows.runtime import WorkflowRuntime
from pm.memory.bash_view import MemoryBashEnvironment
from pm.runs.workflow_runner import PROJECT_ROOT, WorkflowRequest, WorkflowRunner, WorkflowStatus

from workflows.inner_thought.schemas import IntegrationResult, ThoughtResult, TriggerChoice
from workflows.inner_thought.stages.trigger import steps as trigger_steps
from workflows.inner_thought.stages.thought import steps as thought_steps
from workflows.inner_thought.stages.integrate import steps as integrate_steps

ROOT = Path(".runtime/a17-contract-tests")


def example_request() -> dict:
    return {
        "companion_name": "Spamuel",
        "user_name": "Jesse",
        "state_summary": "quiet and a little restless",
        "seconds_since_user_input": 4000.0,
        "tick_id": 100,
        "main_frame": {"tick": 99, "frame_id": 3, "summary": "the story talk", "surprise": 0.1,
                        "gap_seconds": 0.0},
        "main_frame_age_ticks": 1,
        "active_goal_records": [
            {"knoxel_id": 5, "text": "finish the story", "tick_id": 80,
             "fulfilment": 0.1, "affective_valence": -0.2, "revision": 1}],
        "pending_expectations": [
            {"knoxel_id": 7, "content": "Jesse replies tonight", "anticipation": "a reply",
             "expected_outcome": "a friendly reply", "tick_id": 90, "revision": 1}],
        "recent_thoughts": [
            {"knoxel_id": 9, "text": "the draft again", "tick_id": 98, "operation": "memory_search",
             "trigger": "unresolved_commitment", "continuation": "summarize",
             "source_ids": [5], "revision": 1}],
        "candidates": [
            {"knoxel_id": 5, "kind": "goal", "text": "finish the story", "revision": 1,
             "tick_id": 80, "urgency": 0.6, "reason": "unresolved"},
            {"knoxel_id": 9, "kind": "thought", "text": "the draft again", "revision": 1,
             "tick_id": 98, "urgency": 0.3, "reason": "continuity"}],
        "trigger_candidates": [
            {"name": "unresolved_commitment", "cause": "goal 5 is 20 ticks old and 0.10 fulfilled",
             "urgency": 0.6, "operations": ["commitment_review", "plan_next_step"],
             "reason": "An accepted goal with no progress needs private attention."},
            {"name": "idle_opportunity", "cause": "the last user input was 4000 seconds ago",
             "urgency": 0.25, "operations": ["curiosity_spark"],
             "reason": "Idle time is the opportunity for private work."}],
        "channels": {},
        "operations": [{"name": "- commitment_review (evaluate, present): review"}],
    }


def run_kernel(manifest: Path, payload: dict, completion: str) -> dict:
    captured: dict = {}

    def llm_complete(messages, **generation):
        captured["messages"] = messages
        captured["generation"] = generation
        return completion

    environment = MemoryBashEnvironment({}, {}, "read")
    runtime = WorkflowRuntime(environment=environment, stop_event=threading.Event(),
                              tools={"llm_complete": llm_complete}, propagate_python_errors=True)
    runner = WorkflowRunner(manifest, ROOT / manifest.parent.name, namespace=manifest.parent.name,
                            base_dir=PROJECT_ROOT)
    result = runner.invoke(WorkflowRequest(f"{manifest.parent.name}-contract", payload), runtime)
    assert result.status is WorkflowStatus.Completed, result.kernel_summary
    return result.report


class TriggerContractTests(unittest.TestCase):
    def test_messages_name_the_companion_and_every_candidate(self):
        messages = trigger_steps.build_messages(example_request(), trigger_steps.load_prompts())
        self.assertEqual([message.role.value for message in messages], ["system", "user"])
        user_text = messages[1].content.text
        self.assertIn("Spamuel", messages[0].content.text)
        self.assertIn("unresolved_commitment", user_text)
        self.assertIn("commitment_review", user_text)
        self.assertIn("id=5", user_text)
        self.assertIn("finish the story", user_text)

    def test_kernel_returns_a_valid_trigger_choice(self):
        completion = json.dumps({"trigger": "unresolved_commitment", "operation": "commitment_review",
                                 "temporal_focus": "present", "candidate_ids": [5],
                                 "urgency": 0.6, "reason": "the goal stalled"})
        report = run_kernel(Path(trigger_steps.__file__).resolve().parent / "trigger.workflow.md",
                            example_request(), completion)
        parsed = TriggerChoice.model_validate(report)
        self.assertEqual(parsed.operation, "commitment_review")
        self.assertEqual(parsed.candidate_ids, [5])


class ThoughtContractTests(unittest.TestCase):
    def test_inspection_evidence_reaches_the_prompt(self):
        payload = {**example_request(),
                   "inspection": {
                       "status": "ok",
                       "capability_overlay": {"available": True, "stale": [], "capabilities": [
                           {"name": "private_thought", "state": "observed",
                            "summary": "Run the inner loop.", "revision": "r1"}]},
                       "episodes": [{"episode_id": 7, "loop": "inner", "operation": "memory_search",
                                     "outcome": "committed", "summary": "a private recollection"}],
                       "newest_episode_answers": {}},
                   "trigger": {"trigger": "capability_doubt", "operation": "capability_inspection",
                               "temporal_focus": "present", "candidate_ids": [], "urgency": 0.4,
                               "reason": "the evidence is partial"},
                   "operation": {"name": "capability_inspection", "group": "inspect",
                                 "mode": "discriminatory", "temporal_focus": "present",
                                 "description": "Read the active capability evidence."}}
        messages = thought_steps.build_messages(payload, thought_steps.load_prompts())
        user_text = messages[1].content.text
        self.assertIn("Selected implementation evidence", user_text)
        self.assertIn("capability=private_thought", user_text)
        self.assertIn("episode=7", user_text)

    def test_kernel_returns_a_valid_thought_result(self):
        payload = {**example_request(),
                   "trigger": {"trigger": "unresolved_commitment", "operation": "commitment_review",
                               "temporal_focus": "present", "candidate_ids": [5], "urgency": 0.6,
                               "reason": "the goal stalled"},
                   "operation": {"name": "commitment_review", "group": "evaluate", "mode": "discriminatory",
                                 "temporal_focus": "present",
                                 "description": "Review an accepted commitment."}}
        completion = json.dumps({"thought": "I keep postponing the story.",
                                 "evidence_ids": [5], "evidence_summary": "goal 5 is unfulfilled",
                                 "interpretation": "I avoid it", "uncertainty": 0.4,
                                 "reality": "internal", "alternatives": [],
                                 "continuation": "summarize", "continuation_reason": "done"})
        report = run_kernel(Path(thought_steps.__file__).resolve().parent / "thought.workflow.md",
                            payload, completion)
        parsed = ThoughtResult.model_validate(report)
        self.assertEqual(parsed.reality, "internal")
        self.assertEqual(parsed.evidence_ids, [5])

    def test_thought_receives_the_next_operation_catalogue(self):
        payload = {**example_request(),
                   "trigger": {"trigger": "idle_opportunity", "operation": "curiosity_spark"},
                   "operation": {"name": "curiosity_spark"},
                   "operations": [{"name": "- plan_next_step (plan, future): Develop a concrete next step."}]}
        messages = thought_steps.build_messages(payload, thought_steps.load_prompts())
        self.assertIn("Available next operations:", messages[1].content.text)
        self.assertIn("plan_next_step", messages[1].content.text)


class IntegrateContractTests(unittest.TestCase):
    def test_kernel_returns_a_valid_integration(self):
        payload = {**example_request(),
                   "trigger": {"trigger": "unresolved_commitment", "operation": "commitment_review",
                               "temporal_focus": "present", "candidate_ids": [5], "urgency": 0.6,
                               "reason": "the goal stalled"},
                   "thought": {"thought": "I keep postponing the story.", "evidence_ids": [5],
                               "evidence_summary": "goal 5 is unfulfilled",
                               "interpretation": "I avoid it", "uncertainty": 0.4,
                               "reality": "internal", "alternatives": [],
                               "continuation": "summarize", "continuation_reason": "done"},
                   "operation": {"name": "commitment_review", "group": "evaluate"}}
        completion = json.dumps({
            "goal_changes": [{"change": "reprioritize", "goal": "finish the story",
                              "goal_key": "finish-story", "reason": "it keeps returning",
                              "priority": 0.7}],
            "expectation": {"action": "none", "expectation_id": None, "anticipation": "",
                            "expected_outcome": "", "expected_valence": 0.0, "uncertainty": 0.5,
                            "goal_ids": [], "status": "pending", "observation_ids": [],
                            "reason": ""},
            "noticing": "the story topic keeps returning",
            "disclosure": {"mode": "withhold", "shareable_summary": "", "reason": "private"},
            "summary": "reprioritized the story goal"})
        report = run_kernel(Path(integrate_steps.__file__).resolve().parent / "integrate.workflow.md",
                            payload, completion)
        parsed = IntegrationResult.model_validate(report)
        self.assertEqual(parsed.goal_changes[0].change, "reprioritize")
        self.assertEqual(parsed.disclosure.mode, "withhold")


if __name__ == "__main__":
    unittest.main()
