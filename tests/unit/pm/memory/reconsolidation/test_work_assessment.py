"""Exercise the main decision's memory cue on real store views and world times."""
from datetime import datetime, timedelta
import unittest

from pm.memory.reconsolidation.assessment import MemoryWorkReader, StoryHistoryObservation
from pm.memory.store import MemoryStore
from pm.records.base import RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ClusterType, FeatureType
from pm.records.memory_records import Feature, MemoryClusterRecord


class MemoryWorkAssessmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 11, 23, 13)
        self.ghost = MemoryStore(PersonaConfig(companion_name="Spamuel", user_name="Jesse"))
        self.reader = MemoryWorkReader()

    def add(self, when: datetime, kind: FeatureType, scope: RecordScope = RecordScope()) -> Feature:
        batch = self.ghost.begin_batch(f"source-{scope.reality_id}-{kind.value}-{when.isoformat()}", embed=False)
        row = batch.create(Feature(content=f"Event at {when:%H:%M}", feature_type=kind, scope=scope, causal=True,
                                   timestamp_world_begin=when, timestamp_world_end=when))
        self.ghost.commit(batch)
        return row

    def test_membership_and_story_omissions_follow_the_actual_read_scope(self) -> None:
        first = self.add(self.now - timedelta(days=1), FeatureType.Dialogue)
        middle = self.add(self.now - timedelta(days=1, minutes=-1), FeatureType.WorldEvent)
        last = self.add(self.now - timedelta(days=1, minutes=-2), FeatureType.Dialogue)
        self.add(self.now - timedelta(days=1), FeatureType.Thought)
        elsewhere = self.add(self.now - timedelta(days=1), FeatureType.Dialogue, RecordScope(reality_id=3))
        self.add(self.now, FeatureType.Dialogue)
        self.add(self.now + timedelta(days=1), FeatureType.Dialogue)
        frozen = self.ghost.snapshot(capture=False)
        observation = StoryHistoryObservation(observed_at=self.now, current_message_id=last.id,
            covered_features=1, omitted_features=2, minimum_cover_prompt_tokens=507, selected_prompt_tokens=452,
            available_history_tokens=3)
        self.reader.observe_story(observation)
        earlier = StoryHistoryObservation(observed_at=self.now - timedelta(hours=1), current_message_id=first.id,
            covered_features=2, omitted_features=0, minimum_cover_prompt_tokens=130, selected_prompt_tokens=144)
        self.reader.observe_story(earlier)
        self.assertEqual(self.reader.last_story, observation)
        old = self.reader.assess(frozen, set(), self.now)
        self.assertEqual(old.external_features, 3)
        self.assertGreater(old.estimated_raw_text_tokens, 0)
        self.assertEqual(old.oldest_source_id, first.id)
        self.assertIn("did not fit", old.decision_evidence())
        self.assertIn("word for word", old.decision_evidence())
        other = self.reader.assess(frozen, {elsewhere.scope}, self.now)
        self.assertEqual(other.external_features, 1)
        self.assertIsNone(other.last_story)
        batch = self.ghost.begin_batch("sparse episode", embed=False)
        batch.create(MemoryClusterRecord(cluster_type=ClusterType.Topical, level=100, content="A shared exchange.",
            included_event_ids=f"{first.id},{last.id}", timestamp_world_begin=first.timestamp_world_begin,
            timestamp_world_end=last.timestamp_world_end))
        self.ghost.commit(batch)
        current = self.reader.assess(self.ghost.snapshot(capture=False), set(), self.now)
        self.assertEqual(current.external_features, 1)
        self.assertEqual(current.oldest_source_id, middle.id)
        self.assertEqual(self.reader.assess(frozen, set(), self.now), old)

    def test_continuous_chat_accrues_work_and_a_short_recent_topic_stays_open(self) -> None:
        largest_open_topic = 2 * self.reader.config.min_block - 1
        for minute in range(largest_open_topic):
            self.add(self.now - timedelta(minutes=largest_open_topic - minute), FeatureType.Dialogue)
        self.assertEqual(self.reader.assess(self.ghost.snapshot(capture=False), set(), self.now).external_features, 0)
        self.add(self.now, FeatureType.Dialogue)
        frozen = self.ghost.snapshot(capture=False)
        active = self.reader.assess(frozen, set(), self.now)
        self.assertEqual(active.external_features, 1)
        self.assertEqual(active.sessions_with_work, 1)
        idle = self.reader.assess(frozen, set(), self.now + timedelta(minutes=self.reader.config.open_topic_minutes))
        self.assertEqual(idle.external_features, largest_open_topic + 1)
