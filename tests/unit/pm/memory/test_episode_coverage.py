"""Exercise episode membership, attribution, and Story fitting on a real store."""
import asyncio
import unittest
from datetime import datetime, timedelta

from pm.memory.cluster_membership import ClusterMembership
from pm.memory.context_plan import Focus, LaneSpec, PlanSpec, plan_context
from pm.memory.retrieval import candidate, retrieve, scoped_records
from pm.memory.store import MemoryStore
from pm.memory.temporal_context import TemporalCoverageProblem
from pm.records.base import RecordScope
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ClusterType, EntityClass, FeatureType, FrameEnum
from pm.records.lineage import RecordLineage
from pm.records.memory_records import Entity, Feature, MemoryClusterRecord
from workflows.public_speech.context_planning import EmbeddedFocus, FocusInput, freeze_temporal_history, plan_reply_context
from workflows.public_speech.context_trace import SemanticContextTrace


class EpisodeCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        self.start = datetime(2026, 1, 1, 12)
        batch = self.store.begin_batch("two-speakers")
        self.one = batch.create(Entity(content="Morgan", aliases=["Pine"], entity_class=EntityClass.Agent))
        self.two = batch.create(Entity(content="Robin", aliases=["Finch"], entity_class=EntityClass.Human))
        self.events: list[Feature] = []
        for index, speaker in enumerate((self.one, self.two, self.one)):
            # All three arrive in one tick. The speaker IDs distinguish the
            # exchanges and knoxel IDs order simultaneous messages.
            event = batch.create(Feature(content=f"Statement {index}. " * 40, entity_id=speaker.id,
                override_source=speaker.content, feature_type=FeatureType.Dialogue, causal=True,
                timestamp_world_begin=self.start, timestamp_world_end=self.start))
            self.events.append(event)
        self.store.commit(batch)

    def add_episode(self, members: list[Feature], text: str = "A shared decision.") -> MemoryClusterRecord:
        batch = self.store.begin_batch(f"episode-{len(self.store.all_knoxels)}")
        cluster = MemoryClusterRecord(content=text, cluster_type=ClusterType.Topical, level=100,
            included_event_ids=",".join(str(item.id) for item in members),
            timestamp_world_begin=self.start, timestamp_world_end=self.start,
            lineage=RecordLineage(episode_id=batch.commit_id, operation="memory_reconsolidation.topical",
                                  source_event_ids=[members[0].id]))
        cluster.metadata["participant_ids"] = [self.one.id, self.two.id]
        batch.create(cluster)
        self.store.commit(batch)
        return cluster

    def lifetime(self) -> TemporalCoverageProblem:
        view = self.store.snapshot()
        records, _ = scoped_records(view)
        focus = EmbeddedFocus([], SemanticContextTrace(status="disabled"))
        return TemporalCoverageProblem(freeze_temporal_history(view, records, FocusInput(), focus))

    def test_interleaved_episode_is_retrievable_by_each_participant_without_hiding_the_middle_event(self) -> None:
        first, middle, last = self.events
        cluster = self.add_episode([first, last])
        problem = self.lifetime()
        self.assertNotIn(cluster.id, problem.graph.candidates_by_id)
        self.assertEqual(problem.minimum_cover().covered_feature_ids, (first.id, middle.id, last.id))
        omissions = problem.selection((), (first.id, middle.id, last.id))
        pinned = problem.include_visible_records(omissions, {middle.id, cluster.id})
        self.assertEqual(pinned.covered_feature_ids, (middle.id,))
        for alias in ("Pine", "Finch"):
            found = asyncio.run(retrieve(self.store.snapshot(), alias, method="lexical", reserve_context=False))
            self.assertEqual([row.knoxel_id for row in found.matches], [cluster.id])
            self.assertEqual(found.matches[0].source_ids, [first.id])
        planned = asyncio.run(plan_reply_context(self.store.snapshot(), "story", FocusInput(required_ids=[cluster.id])))
        self.assertEqual([(row.knoxel_id, row.lane) for row in planned.evidence], [(cluster.id, "focus")])

    def test_episode_supplies_an_immediate_contiguous_cover_in_both_planners(self) -> None:
        first, middle, last = self.events
        cluster = self.add_episode([first, middle])
        minimum = self.lifetime().minimum_cover()
        self.assertEqual([item.record_id for item in minimum.items], [cluster.id, last.id])
        self.assertEqual(minimum.covered_feature_ids, (first.id, middle.id, last.id))
        self.assertEqual(minimum.items[0].source_ids, (first.id,))
        view = self.store.snapshot()
        records, _ = scoped_records(view)
        costs = {key: (1 if key == cluster.id else 100) for key in records}
        vectors = {key: None for key in records}
        lane = LaneSpec(name="history", budget_share=1, temporal_mode="preserve_history")
        spec = PlanSpec(name="story", total_tokens=101, lanes=[lane])
        selected = plan_context(records, spec, Focus(), vectors=vectors, token_costs=costs,  # pm-lint: allow PM005 -- per-record vector and cost indexes
                                now=self.start + timedelta(days=1))
        self.assertEqual(set(selected.ids()), {last.id, cluster.id})

    def test_legacy_child_membership_resolves_without_guessing_or_promoting_accounted_sources(self) -> None:
        first, middle, last = self.events
        child = self.add_episode([first, last])
        batch = self.store.begin_batch("calendar")
        parent = batch.create(MemoryClusterRecord(content="A calendar recollection.", cluster_type=ClusterType.Temporal,
            level=5, included_cluster_ids=str(child.id), timestamp_world_begin=self.start, timestamp_world_end=self.start,
            lineage=RecordLineage(episode_id="calendar", operation="memory_reconsolidation.temporal",
                                  parent_record_ids=[child.id])))
        ambiguous = batch.create(MemoryClusterRecord(content="A legacy memory with only dates.",
            cluster_type=ClusterType.Temporal, level=1,
            timestamp_world_begin=self.start, timestamp_world_end=self.start))
        self.store.commit(batch)
        view = self.store.snapshot()
        membership = ClusterMembership(view.all_knoxels)
        self.assertEqual(membership.source_ids(parent), {first.id, last.id})
        self.assertEqual(membership.source_ids(ambiguous), frozenset())
        self.assertEqual(candidate(view, parent.id, parent).source_ids, [first.id])
        self.assertNotIn(parent.id, self.lifetime().graph.candidates_by_id)
        self.assertNotIn(ambiguous.id, self.lifetime().graph.candidates_by_id)
        self.assertIn(middle.id, self.lifetime().minimum_cover().covered_feature_ids)

    def test_mixed_scope_cluster_is_unusable_even_when_both_frames_are_visible(self) -> None:
        first = self.events[0]
        batch = self.store.begin_batch("other-frame")
        private = batch.create(Feature(content="A meta-level observation.", scope=RecordScope(frame=FrameEnum.Meta),
            feature_type=FeatureType.WorldEvent, causal=True))
        self.store.commit(batch)
        mixed = self.add_episode([first, private], text="Merged across frames.")
        view = self.store.snapshot()
        visible, rejected = scoped_records(view, scopes={(-1, "in_world"), (-1, "meta")})
        self.assertIn(first.id, visible)
        self.assertIn(private.id, visible)
        self.assertEqual(rejected[mixed.id], "different_source_scope")
        self.assertFalse(ClusterMembership.from_snapshot(view).source_ids(mixed))
        focus = FocusInput(current_ids=(first.id,), exclude_ids=[first.id, private.id])
        planned = asyncio.run(plan_reply_context(view, "story", focus))
        assert planned.temporal_history is not None
        story_ids = {row.record_id for row in planned.temporal_history.items}
        self.assertTrue({self.events[1].id, self.events[2].id}.issubset(story_ids))
        self.assertNotIn(private.id, story_ids)
        self.assertNotIn(mixed.id, story_ids)

    def test_granted_historical_child_proves_membership_after_a_summary_is_revised(self) -> None:
        first, second, _ = self.events
        child = self.add_episode([first, second])
        batch = self.store.begin_batch("older-day")
        parent = batch.create(MemoryClusterRecord(content="The prior calendar recollection.", cluster_type=ClusterType.Temporal,
            level=5, included_cluster_ids=str(child.id), timestamp_world_begin=self.start, timestamp_world_end=self.start))
        self.store.commit(batch)
        update = self.store.begin_batch("corrected-episode")
        current = update.replace(child.copy_for_edit())
        current.content = "The updated episode."
        self.store.commit(update)
        view = self.store.snapshot()
        self.assertIn(child.id, view.historical_records)
        visible, _ = scoped_records(view)
        self.assertIn(current.id, visible)
        self.assertIn(parent.id, visible)
        self.assertNotIn(child.id, visible)
        self.assertEqual(ClusterMembership.from_snapshot(view).source_ids(parent), {first.id, second.id})
        self.assertEqual(candidate(view, parent.id, parent).source_ids, [first.id])
        restricted = self.store.snapshot([parent.id, first.id, second.id])
        _, rejected = scoped_records(restricted)
        self.assertEqual(rejected[parent.id], "unavailable_evidence")
        self.assertFalse(ClusterMembership.from_snapshot(restricted).source_ids(parent))
