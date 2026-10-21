"""Durable private ontology state. No model endpoint.

The observer writes real ``OntologyTermKnoxel`` and ``OntologyBindingKnoxel``
records into a real MemoryStore. These checks fix the A09R rules: terms come
only from implemented mechanisms, a notable feeling binds to the accepted
state, a changed state supersedes the old binding, and no term can invent an
unimplemented sense or emotion.
"""
from __future__ import annotations

import unittest

from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig, CycleState
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import Feature
from pm.records.enums import FeatureType
from pm.records.mental_state import FullMentalState
from workflows.public_speech.ontology import MENTAL_MOVES, ontology_lines, observe_ontology, seed_terms, term_lines
from workflows.public_speech.public_context import PublicContext


def ghost_with_state(*, anger: float = 0.0, tenderness: float = 0.0) -> MemoryStore:
    ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
    state = FullMentalState()
    state.state_emotions.anger = anger
    state.state_emotions.tenderness = tenderness
    batch = ghost.begin_batch("ontology-state", new_tick=True)
    feature = batch.create(Feature(
        content="accepted appraisal", feature_type=FeatureType.StateUpdate, causal=True,
        override_source="Spamuel", mental_state_appraisal_vec=state.to_list(),
        metadata={"memory_kind": "internal_state", "evidence_ids": []},
    ))
    batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, knoxel_ids=[feature.id]),
                  CycleState(tick_id=batch.tick_id, mental_state=state))
    ghost.commit(batch)
    return ghost


class SeedTermTests(unittest.TestCase):
    def test_seeds_cover_implemented_emotions_and_mental_moves(self):
        terms = {item["term"]: item for item in seed_terms()}
        for name in ("anger", "tenderness", "guilt", "pride"):
            self.assertEqual(terms[name]["term_kind"], "mood_word")
        for name, _ in MENTAL_MOVES:
            self.assertEqual(terms[name]["term_kind"], "mental_move")

    def test_a_sense_that_the_runtime_does_not_have_is_not_a_term(self):
        names = {item["term"] for item in seed_terms()}
        self.assertNotIn("smell", names)
        self.assertNotIn("sight", names)


class OntologyRenderTests(unittest.TestCase):
    def context(self) -> PublicContext:
        return PublicContext(companion_name="Spamuel", user_name="Jesse", ontology=[
            {"id": 1, "term": "anger", "term_kind": "mood_word",
             "definition": "Frustration at a blocked goal.", "usage_rule": "Use with state."},
            {"id": 2, "term_id": 1, "active": True, "fit_score": 0.7, "explanation": "accepted anger"},
        ])

    def test_term_lines_render_words_without_the_binding(self):
        lines = term_lines(self.context())
        self.assertIn("- anger (mood_word): Frustration at a blocked goal.; Use with state.", lines)
        self.assertFalse(any("fits right now" in line for line in lines))

    def test_ontology_lines_include_the_active_binding(self):
        lines = ontology_lines(self.context())
        self.assertIn("- anger fits right now (fit 0.7): accepted anger", lines)


class ObserveTests(unittest.TestCase):
    def test_notable_feelings_bind_and_terms_are_committed_once(self):
        ghost = ghost_with_state(anger=0.8, tenderness=0.5)
        receipt = observe_ontology(ghost)
        self.assertIn("anger", receipt["bound_terms"])
        self.assertIn("tenderness", receipt["bound_terms"])
        records = list(ghost.all_knoxels.values())
        terms = [item for item in records if type(item).__name__ == "OntologyTermRecord"]
        bindings = [item for item in records if type(item).__name__ == "OntologyBindingRecord"]
        self.assertGreaterEqual(len(terms), len(seed_terms()))
        self.assertEqual(len([b for b in bindings if b.active]), 2)
        count = len(records)
        again = observe_ontology(ghost)
        self.assertEqual(again["created_terms"], [])
        self.assertEqual(again["created_bindings"], [])
        self.assertEqual(len(ghost.all_knoxels), count)

    def test_a_weak_feeling_does_not_bind(self):
        ghost = ghost_with_state(anger=0.1)
        receipt = observe_ontology(ghost)
        self.assertNotIn("anger", receipt["bound_terms"])

    def test_a_changed_state_supersedes_the_old_binding(self):
        ghost = ghost_with_state(anger=0.9)
        observe_ontology(ghost)
        state = FullMentalState()
        state.state_emotions.anger = 0.4
        batch = ghost.begin_batch("ontology-state-2", new_tick=True)
        feature = batch.create(Feature(
            content="accepted appraisal", feature_type=FeatureType.StateUpdate, causal=True,
            override_source="Spamuel", mental_state_appraisal_vec=state.to_list(),
            metadata={"memory_kind": "internal_state", "evidence_ids": []},
        ))
        batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, knoxel_ids=[feature.id]),
                      CycleState(tick_id=batch.tick_id, mental_state=state))
        ghost.commit(batch)
        observe_ontology(ghost)
        active = [b for b in ghost.all_knoxels.values()
                  if type(b).__name__ == "OntologyBindingRecord" and b.active]
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].bound_knoxel_ids[0], feature.id)


if __name__ == "__main__":
    unittest.main()
