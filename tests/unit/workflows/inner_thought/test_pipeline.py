"""Private pipeline tests. A deterministic ``run_stage`` stands in for the kernel."""
from __future__ import annotations

import unittest

from workflows.inner_thought.pipeline import citable_ids, run_cycle
from workflows.inner_thought.schemas import InnerRequest, TriggerCandidate


def request(**overrides) -> InnerRequest:
    values = dict(
        run_id="run-1", companion_name="Spamuel", user_name="Jesse",
        tick_id=100,
        main_frame={"tick": 99, "frame_id": 3, "summary": "the story talk", "surprise": 0.0,
                    "gap_seconds": 0.0},
        active_goal_records=[{"knoxel_id": 5, "text": "finish the story", "tick_id": 80,
                              "fulfilment": 0.1, "affective_valence": -0.2, "revision": 1}],
        pending_expectations=[{"knoxel_id": 7, "content": "Jesse replies tonight",
                               "anticipation": "a reply", "expected_outcome": "a reply",
                               "tick_id": 90, "revision": 1}],
        recent_thoughts=[],
        candidates=[{"knoxel_id": 5, "kind": "goal", "text": "finish the story",
                     "revision": 1, "tick_id": 80, "urgency": 0.6, "reason": "unresolved"}],
        trigger_candidates=[TriggerCandidate(
            name="unresolved_commitment", cause="goal 5 is old", urgency=0.5,
            operations=["commitment_review"], reason="test")],
        channels={}, operations=[],
        seconds_since_user_input=10.0,
    )
    values.update(overrides)
    return InnerRequest(**values)


def stage_reports(operation="commitment_review"):
    return {
        "trigger": {"trigger": "unresolved_commitment", "operation": operation,
                    "temporal_focus": "present", "candidate_ids": [5], "urgency": 0.5,
                    "reason": "the goal stalled"},
        "thought": {"thought": "The story keeps stalling.",
                    "evidence_ids": [5], "evidence_summary": "goal 5 is unfulfilled",
                    "interpretation": "I keep avoiding it.", "uncertainty": 0.4,
                    "reality": "internal",
                    "alternatives": [{"text": "ask Jesse to co-write", "disposition": "chosen",
                                      "reason": "it shares the effort"}],
                    "continuation": "continue", "continuation_reason": "the plan needs a next step"},
        "integrate": {"goal_changes": [{"change": "add", "goal": "invite Jesse to co-write",
                                        "goal_key": "co-write", "reason": "sharing helps",
                                        "priority": 0.6}],
                      "expectation": {"action": "open", "expectation_id": None,
                                      "anticipation": "Jesse says yes",
                                      "expected_outcome": "a shared session",
                                      "expected_valence": 0.5, "uncertainty": 0.4,
                                      "goal_ids": [5], "status": "pending",
                                      "observation_ids": [], "reason": "the invitation runs tonight"},
                      "noticing": "watch for avoidance of the story topic",
                      "disclosure": {"mode": "candid",
                                     "shareable_summary": "I want to invite Jesse to co-write.",
                                     "reason": "it leads to a concrete offer"},
                      "summary": "accepted the co-write plan"},
    }


class PipelineTests(unittest.TestCase):
    def test_no_trigger_commits_nothing_and_calls_no_stage(self):
        calls = []

        def run_stage(name, payload):
            calls.append(name)
            raise AssertionError("no stage may run without a trigger")

        report = run_cycle(request(trigger_candidates=[]), run_stage)
        self.assertEqual(report.operation, "none")
        self.assertEqual(calls, [])

    def test_full_cycle_assembles_the_report(self):
        reports = stage_reports()
        report = run_cycle(request(), lambda name, payload: reports[name])
        self.assertEqual(report.operation, "commitment_review")
        self.assertEqual(report.thought, "The story keeps stalling.")
        self.assertEqual(report.reality, "internal")
        self.assertEqual(report.goal_changes[0].goal, "invite Jesse to co-write")
        self.assertEqual(report.noticing, "watch for avoidance of the story topic")
        self.assertEqual(report.disclosure.mode, "candid")
        self.assertEqual(report.candidate_ids, [5])

    def test_unknown_operation_is_rejected(self):
        reports = stage_reports(operation="TH_COMPOSE_REPLY")
        with self.assertRaises(KeyError):
            run_cycle(request(), lambda name, payload: reports[name])

    def test_evidence_outside_the_frozen_view_is_rejected(self):
        reports = stage_reports()
        reports["thought"] = {**reports["thought"], "evidence_ids": [999]}
        with self.assertRaisesRegex(ValueError, "outside the frozen view"):
            run_cycle(request(), lambda name, payload: reports[name])

    def test_trigger_none_short_circuits_before_the_thought_stage(self):
        calls = []
        reports = stage_reports()
        reports["trigger"] = {"trigger": "unresolved_commitment", "operation": "none",
                              "temporal_focus": "present", "candidate_ids": [], "urgency": 0.0,
                              "reason": "not now"}

        def run_stage(name, payload):
            calls.append(name)
            return reports[name]

        report = run_cycle(request(), run_stage)
        self.assertEqual(report.operation, "none")
        self.assertEqual(calls, ["trigger"])

    def test_resolve_must_name_a_pending_expectation(self):
        reports = stage_reports()
        reports["integrate"] = {**reports["integrate"],
                                "expectation": {"action": "resolve", "expectation_id": 999,
                                                "anticipation": "", "expected_outcome": "",
                                                "expected_valence": 0.0, "uncertainty": 0.5,
                                                "goal_ids": [], "status": "supported",
                                                "observation_ids": [5], "reason": "a reply came"}}
        with self.assertRaisesRegex(ValueError, "pending expectation"):
            run_cycle(request(), lambda name, payload: reports[name])

    def test_resolve_without_observations_is_rejected(self):
        reports = stage_reports()
        reports["integrate"] = {**reports["integrate"],
                                "expectation": {"action": "resolve", "expectation_id": 7,
                                                "anticipation": "", "expected_outcome": "",
                                                "expected_valence": 0.0, "uncertainty": 0.5,
                                                "goal_ids": [], "status": "supported",
                                                "observation_ids": [], "reason": ""}}
        with self.assertRaisesRegex(ValueError, "observations"):
            run_cycle(request(), lambda name, payload: reports[name])

    def test_citable_ids_cover_goals_expectations_thoughts_and_frame(self):
        parsed = request(recent_thoughts=[{"knoxel_id": 9, "text": "x", "tick_id": 99,
                                           "operation": "", "trigger": "", "continuation": "",
                                           "source_ids": [], "revision": 1}])
        self.assertEqual(citable_ids(parsed), {5, 7, 9, 3})


if __name__ == "__main__":
    unittest.main()
