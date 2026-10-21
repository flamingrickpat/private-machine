"""Felt-state channel math tests. No model, kernel, or MemoryStore."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from pm.records.cycle_state import CycleState
from pm.records.mental_state import FullMentalState, create_empty_state
from pm.records.qualia_records import QualeChannelState
from workflows.inner_thought.felt_state import (
    channel_position, is_reportable, notable_channels, previous_cycle_sample,
    project_channels,
)


def state(**emotions) -> FullMentalState:
    base = create_empty_state()
    vector = base.to_list()
    for name, value in emotions.items():
        base.state_emotions.__setattr__(name, value)
    return base


class ChannelPositionTests(unittest.TestCase):
    def test_valence_projects_positive_and_negative_emotions(self):
        happy = channel_position("affect.valence", state(joy=0.8, tenderness=0.2))
        sad = channel_position("affect.valence", state(sadness=0.8, anger=0.4))
        self.assertGreater(happy[0], 0.0)
        self.assertLess(sad[0], 0.0)
        self.assertGreaterEqual(happy[0], -1.0)
        self.assertLessEqual(sad[0], 1.0)

    def test_relationship_channel_reports_absent_source_without_partner(self):
        position, note = channel_position("relationship.warmth", state())
        self.assertEqual((position, note), (0.0, "absent source"))

    def test_goal_conflict_counts_unhappy_unfulfilled_goals(self):
        class Goal:
            def __init__(self, valence, fulfilment):
                self.affective_valence = valence
                self.fulfilment = fulfilment

        conflicted = channel_position("goal.conflict", state(), active_goals=[
            Goal(-0.5, 0.2), Goal(0.5, 0.2), Goal(-0.4, 0.1)])
        self.assertAlmostEqual(conflicted[0], 2 / 3)
        empty = channel_position("goal.conflict", state(), active_goals=[])
        self.assertEqual(empty, (0.0, "absent source"))


class DerivativeTests(unittest.TestCase):
    def test_first_sample_is_initialized_not_measured_stillness(self):
        now = datetime.now(timezone.utc)
        sample = project_channels(state(joy=0.6), previous=None, previous_time=None,
                                   now=now, driver_ids=[11])
        channel = sample["affect.valence"]
        self.assertEqual(channel.velocity, 0.0)
        self.assertEqual(channel.acceleration, 0.0)

    def test_velocity_uses_elapsed_seconds(self):
        now = datetime.now(timezone.utc)
        earlier = now - timedelta(seconds=10)
        previous = {"affect.valence": QualeChannelState(
            channel_id="affect.valence", position=0.2, velocity=0.0, acceleration=0.0)}
        sample = project_channels(state(joy=1.0), previous=previous, previous_time=earlier,
                                  now=now, driver_ids=[])
        channel = sample["affect.valence"]
        # Position moved from 0.2 to 0.2 (1.0/5.0); joy 1.0 gives 0.2.
        self.assertAlmostEqual(channel.position, 0.2)
        self.assertAlmostEqual(channel.velocity, 0.0)

    def test_nonpositive_interval_reports_no_derivative(self):
        now = datetime.now(timezone.utc)
        previous = {"affect.arousal": QualeChannelState(
            channel_id="affect.arousal", position=0.9, velocity=0.5, acceleration=0.0)}
        sample = project_channels(state(fear=1.0), previous=previous, previous_time=now,
                                  now=now, driver_ids=[])
        channel = sample["affect.arousal"]
        self.assertEqual(channel.velocity, 0.0)
        self.assertEqual(channel.acceleration, 0.0)

    def test_mixed_naive_and_aware_timestamps_still_measure_elapsed_time(self):
        # A live A17 run exposed this: CycleState timestamps are naive, and an
        # aware ``now`` must not crash the projection.
        now = datetime.now(timezone.utc)
        earlier_naive = now.replace(tzinfo=None) - timedelta(seconds=10)
        previous = {"affect.valence": QualeChannelState(
            channel_id="affect.valence", position=0.1, velocity=0.0, acceleration=0.0)}
        sample = project_channels(state(joy=1.0), previous=previous,
                                  previous_time=earlier_naive, now=now, driver_ids=[])
        channel = sample["affect.valence"]
        # joy 1.0 projects to 0.2; 0.1 over 10 seconds gives 0.01/s.
        self.assertAlmostEqual(channel.position, 0.2)
        self.assertAlmostEqual(channel.velocity, 0.01)

    def test_notable_channels_flag_movement(self):
        sample = {key: QualeChannelState(channel_id=key, position=0.0, velocity=0.2,
                                         acceleration=0.0) for key in ("affect.valence", "goal.conflict")}
        self.assertEqual(notable_channels(sample), ["affect.valence", "goal.conflict"])


class GateTests(unittest.TestCase):
    def test_reportable_requires_attention_or_causal_driver(self):
        self.assertTrue(is_reportable([1, 2], attention_ids={2}, causal_ids=set()))
        self.assertTrue(is_reportable([1], attention_ids=set(), causal_ids={1}))
        self.assertFalse(is_reportable([1], attention_ids={2}, causal_ids={2}))


class ContinuityTests(unittest.TestCase):
    def test_previous_cycle_sample_reads_newest_cycle_with_channels(self):
        empty = CycleState(tick_id=1)
        filled = CycleState(tick_id=2, qualia_channels={"affect.valence": QualeChannelState(
            channel_id="affect.valence", position=0.5)})
        channels, timestamp = previous_cycle_sample([empty, filled])
        self.assertIn("affect.valence", channels)
        self.assertEqual(timestamp, filled.timestamp)

    def test_no_cycle_yields_no_sample(self):
        self.assertEqual(previous_cycle_sample([]), (None, None))


if __name__ == "__main__":
    unittest.main()
