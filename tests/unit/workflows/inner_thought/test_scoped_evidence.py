"""Check the scope of reflection and inspection evidence for a private step.

The evidence comes from real stores and real introspection indexes. No model
runs, and no synthetic projection substitutes for a record the store holds.
These checks cover deterministic selection, the stated limit, and rendering;
they do not show that a model uses the material well.
"""
from __future__ import annotations

import unittest

from pm.memory.store import MemoryStore
from pm.records.base import Provenance, RecordScope
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.mental_state import FullMentalState
from pm.records.enums import EntityClass, FeatureType, FrameEnum, SourceKind
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Entity, Feature
from workflows.inner_thought.rendering import inspection
from workflows.inner_thought.request_views import PrivateInspection
from workflows.reflection.experiments import context as reflection_context
from workflows.reflection.rendering import private_material


class ScopedEvidenceTests(unittest.TestCase):
    """Only records in the observer's reality and frame reach a private step."""

    def setUp(self) -> None:
        self.store: MemoryStore = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        batch = self.store.begin_batch("anchors", embed=False)
        self.observer: Entity = batch.create(
            Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        self.inside: Feature = batch.create(Feature(content="The sketch is still unfinished.",
            feature_type=FeatureType.Dialogue, provenance=Provenance(source_kind=SourceKind.User)))
        self.foreign: Feature = batch.create(Feature(content="A harbour I only imagined.",
            feature_type=FeatureType.Dialogue, provenance=Provenance(source_kind=SourceKind.Self)))
        self.foreign.scope = RecordScope(frame=FrameEnum.Meta)
        batch.revise(self.foreign)
        batch.preserve_on_recovery()
        self.store.commit(batch)
        for record_id, summary in ((self.inside.id, "the sketch came up"),
                                   (self.foreign.id, "the imagined harbour")):
            turn = self.store.begin_batch(f"main-turn-{record_id}", new_tick=True, embed=False)
            turn.depend_on([record_id])
            turn.set_gwt(GwtTurnRecord(tick_id=turn.tick_id, slot_id="main", knoxel_ids=[record_id],
                                       stimulus_ids=[record_id], summary=summary),
                         CycleState(tick_id=turn.tick_id, mental_state=FullMentalState()))
            self.store.commit(turn)
        self.view = self.store.snapshot(capture=False)
        self.permitted = frozenset({self.observer.id, self.inside.id})

    def test_unscoped_reflection_context_keeps_every_record(self) -> None:
        unscoped = reflection_context(self.view)
        record_ids = {record.knoxel_id for record in unscoped.records}
        self.assertIn(self.foreign.id, record_ids)
        self.assertEqual([episode.tick_id for episode in unscoped.episodes], [2, 1])

    def test_scoped_reflection_context_omits_foreign_records_and_episodes(self) -> None:
        scoped = reflection_context(self.view, permitted_ids=self.permitted)
        record_ids = {record.knoxel_id for record in scoped.records}
        self.assertEqual(record_ids, {self.inside.id})
        self.assertEqual([episode.tick_id for episode in scoped.episodes], [1])
        self.assertIn("1 episodes are outside this observer's reality and frame.", scoped.episode_limits)

    def test_rendered_material_carries_the_selected_episode_and_its_source(self) -> None:
        scoped = reflection_context(self.view, permitted_ids=self.permitted)
        material = str(private_material(scoped))
        self.assertIn("the sketch came up", material)
        self.assertIn("The sketch is still unfinished.", material)
        self.assertNotIn("A harbour I only imagined.", material)

    def test_empty_material_is_stated_as_a_limit(self) -> None:
        empty = reflection_context(self.view, permitted_ids=frozenset({self.observer.id}))
        self.assertIn("no reflection evidence", str(private_material(empty)))

    def test_scoped_inspection_omits_an_episode_that_names_a_foreign_record(self) -> None:
        from pm.introspection.episodes import EpisodeIndex
        recent = EpisodeIndex(self.view).recent(limit=5)
        kept = [episode for episode in recent if episode.record_ids() <= self.permitted]
        self.assertEqual([episode.tick_id for episode in kept], [1])
        self.assertEqual([episode.tick_id for episode in recent], [2, 1])

    def test_capability_evidence_stays_implementation_evidence(self) -> None:
        evidence = PrivateInspection.model_validate({"status": "ok", "capability_overlay": {
            "available": True, "capabilities": [
                {"name": "source_inspection", "state": "implemented", "summary": "Read a source.", "revision": "r1"}]}})
        rendered = inspection(evidence)
        self.assertIn("capability=source_inspection", rendered)
        self.assertIn("no implementation evidence", inspection(None))
