"""Trigger policy tests. No model, kernel, or MemoryStore."""
from __future__ import annotations

import unittest

from workflows.inner_thought.schemas import InnerRequest
from workflows.inner_thought.triggers import evaluate_triggers


def request(**overrides) -> InnerRequest:
    values = dict(
        companion_name="Spamuel", user_name="Jesse",
        tick_id=100,
        active_goal_records=[{"knoxel_id": 5, "text": "finish the story", "tick_id": 100,
                              "fulfilment": 0.1, "affective_valence": -0.2, "revision": 1}],
        pending_expectations=[{"knoxel_id": 7, "content": "Jesse replies tonight",
                               "anticipation": "a reply", "expected_outcome": "a reply",
                               "tick_id": 100, "revision": 1}],
        recent_thoughts=[{"knoxel_id": 9, "text": "again the story", "tick_id": 99,
                          "operation": "memory_search", "trigger": "unresolved_commitment",
                          "continuation": "summarize", "source_ids": [5], "revision": 1}],
        seconds_since_user_input=10.0,
        main_frame={"tick": 99, "frame_id": 3, "summary": "the story talk", "surprise": 0.0,
                   "gap_seconds": 0.0},
        main_frame_age_ticks=1,
        candidates=[],
        trigger_candidates=[],
        channels={},
        operations=[],
    )
    values.update(overrides)
    return InnerRequest(**values)


class TriggerTests(unittest.TestCase):
    def test_unresolved_goal_fires_with_plan_operations(self):
        goal = {"knoxel_id": 5, "text": "finish the story", "tick_id": 80,
                "fulfilment": 0.1, "affective_valence": -0.2, "revision": 1}
        triggers = evaluate_triggers(request(active_goal_records=[goal]))
        names = [item.name for item in triggers]
        self.assertIn("unresolved_commitment", names)
        trigger = next(item for item in triggers if item.name == "unresolved_commitment")
        self.assertIn("commitment_review", trigger.operations)
        self.assertGreater(trigger.urgency, 0.4)

    def test_fresh_goal_fires_nothing(self):
        goal = {"knoxel_id": 5, "text": "finish the story", "tick_id": 100,
                "fulfilment": 0.9, "affective_valence": 0.2, "revision": 1}
        triggers = evaluate_triggers(request(
            active_goal_records=[goal],
            pending_expectations=[{"knoxel_id": 7, "content": "x", "anticipation": "x",
                                   "expected_outcome": "x", "tick_id": 100, "revision": 1}],
            seconds_since_user_input=10.0,
            main_frame={"tick": 99, "frame_id": 3, "summary": "", "surprise": 0.0, "gap_seconds": 0.0},
            main_frame_age_ticks=5,
        ))
        self.assertEqual(triggers, [])

    def test_idle_fires_when_no_user_input_is_recorded(self):
        triggers = evaluate_triggers(request(seconds_since_user_input=None))
        self.assertIn("idle_opportunity", [item.name for item in triggers])

    def test_surprise_fires_a_debrief(self):
        triggers = evaluate_triggers(request(
            main_frame={"tick": 100, "frame_id": 3, "summary": "unexpected", "surprise": 0.6,
                        "gap_seconds": 0.0},
            main_frame_age_ticks=0,
        ))
        trigger = next(item for item in triggers if item.name == "surprising_outcome")
        self.assertIn("emotional_debrief", trigger.operations)

    def test_repeated_source_fires_rumination_notice(self):
        thoughts = [{"knoxel_id": index, "text": f"thought {index}", "tick_id": 98,
                     "operation": "memory_search", "trigger": "", "continuation": "",
                     "source_ids": [5], "revision": 1} for index in range(3)]
        triggers = evaluate_triggers(request(recent_thoughts=thoughts))
        self.assertIn("repeated_concern", [item.name for item in triggers])

    def test_startup_gap_fires_once(self):
        triggers = evaluate_triggers(request(
            main_frame={"tick": 100, "frame_id": 3, "summary": "back", "surprise": 0.0,
                        "gap_seconds": 7200.0},
        ))
        self.assertIn("startup_gap", [item.name for item in triggers])


if __name__ == "__main__":
    unittest.main()
