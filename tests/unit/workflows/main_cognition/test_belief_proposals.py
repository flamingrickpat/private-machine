"""The belief owner: accepted belief proposals become revisable beliefs.

These tests use the real MemoryStore batch path. A proposal is an
interpretation: it cites observed records and never rewrites them. Repeats
retain; a changed text under the same topic supersedes and keeps history.
"""
from __future__ import annotations

import unittest

from pm.records.cognitive_records import BeliefRecord, BeliefShiftRecord
from pm.records.enums import FeatureType
from pm.records.memory_records import Feature
from workflows.main_cognition.schemas import AttentionNomination
from workflows.main_cognition.stage_commits import apply_belief_proposals


def _ghost():
    from tests.unit.test_design_contracts import small_ghost
    return small_ghost()


def _store(ghost, content: str) -> int:
    return ghost.store_knoxel(Feature(
        content=content, feature_type=FeatureType.Dialogue, causal=True,
        override_source=ghost.get_user_name(),
    )).id


def nomination(text: str, *, topic: str = "my_reliability", targets=(), codelet="consistency_check",
               reality="inferred", nomination_id="n1", status="selected") -> AttentionNomination:
    return AttentionNomination(
        nomination_id=nomination_id, codelet=codelet, family="identity",
        attention_kind="belief", status=status, text=text, topic=topic,
        target_ids=[int(item) for item in targets], reality=reality,
    )


def _beliefs(ghost) -> list[BeliefRecord]:
    return [record for record in ghost.all_knoxels.values() if isinstance(record, BeliefRecord)]


class BeliefOwnerTests(unittest.TestCase):
    def test_an_accepted_belief_cites_its_evidence_without_rewriting_it(self):
        ghost = _ghost()
        source = _store(ghost, "Jesse: you said you would help with the story.")
        original = ghost.get_knoxel_by_id(source).content
        batch = ghost.begin_batch("belief-1")
        receipt = apply_belief_proposals(
            batch, [nomination("I keep letting Jesse down when it matters.",
                               targets=[source])],
            companion_name=ghost.get_companion_name(), catalogue_version="a16.1",
        )
        ghost.commit(batch)
        beliefs = _beliefs(ghost)
        self.assertEqual(len(beliefs), 1)
        self.assertEqual(beliefs[0].supporting_knoxel_ids, [source])
        self.assertEqual(beliefs[0].belief_kind, "interpretation")
        self.assertEqual(beliefs[0].metadata["memory_kind"], "subjective_belief")
        self.assertEqual(beliefs[0].metadata["catalogue_version"], "a16.1")
        # The observed record stays untouched: an interpretation never
        # becomes the event it reads.
        self.assertEqual(ghost.get_knoxel_by_id(source).content, original)
        self.assertEqual(receipt["accepted_ids"], [beliefs[0].id])

    def test_a_repeat_of_the_same_interpretation_retains_without_stacking(self):
        ghost = _ghost()
        source = _store(ghost, "Jesse: we finished the house at last.")
        first = ghost.begin_batch("belief-1")
        apply_belief_proposals(
            first, [nomination("Jesse and I finish what we start.",
                               topic="our_follow_through", targets=[source], codelet="self_story")],
            companion_name=ghost.get_companion_name(),
        )
        ghost.commit(first)
        second_source = _store(ghost, "Jesse: the farm works now too.")
        second = ghost.begin_batch("belief-2")
        receipt = apply_belief_proposals(
            second, [nomination("Jesse and I finish what we start.",
                               topic="our_follow_through", targets=[second_source], codelet="self_story")],
            companion_name=ghost.get_companion_name(),
        )
        ghost.commit(second)
        beliefs = _beliefs(ghost)
        self.assertEqual(len(beliefs), 1)
        self.assertEqual(sorted(beliefs[0].supporting_knoxel_ids), sorted({source, second_source}))
        self.assertEqual(receipt["retained_ids"], [beliefs[0].id])
        self.assertEqual(receipt["accepted_ids"], [])

    def test_a_changed_interpretation_supersedes_and_keeps_history(self):
        ghost = _ghost()
        source = _store(ghost, "Jesse: I talked with the other assistant today.")
        first = ghost.begin_batch("belief-1")
        apply_belief_proposals(
            first, [nomination("Jesse is probably pulling away from me.",
                               topic="my_place_with_jesse", targets=[source], codelet="self_story")],
            companion_name=ghost.get_companion_name(),
        )
        ghost.commit(first)
        second = ghost.begin_batch("belief-2")
        receipt = apply_belief_proposals(
            second, [nomination("Jesse still tells me things first, even busy weeks.",
                               topic="my_place_with_jesse", targets=[source], codelet="self_story")],
            companion_name=ghost.get_companion_name(),
        )
        ghost.commit(second)
        beliefs = sorted(_beliefs(ghost), key=lambda record: record.id)
        self.assertEqual(len(beliefs), 2)
        old, new = beliefs
        self.assertFalse(old.active)
        self.assertEqual(old.superseded_by_id, new.id)
        self.assertTrue(new.active)
        self.assertEqual(new.metadata["revised_from"], old.id)
        self.assertEqual(receipt["revised"], [{"old_id": old.id, "new_id": new.id}])
        shifts = [record for record in ghost.all_knoxels.values()
                  if isinstance(record, BeliefShiftRecord)]
        self.assertEqual(len(shifts), 1)
        self.assertEqual(shifts[0].shift_type, "supersede")

    def test_an_observed_reality_belief_cannot_become_a_fact(self):
        ghost = _ghost()
        batch = ghost.begin_batch("belief-1")
        receipt = apply_belief_proposals(
            batch, [nomination("Jesse left me.", reality="observed")],
            companion_name=ghost.get_companion_name(),
        )
        ghost.commit(batch)
        self.assertEqual(_beliefs(ghost), [])
        self.assertEqual(len(receipt["rejected"]), 1)
        self.assertIn("not a permitted interpretation", receipt["rejected"][0]["reason"])

    def test_losing_stale_and_quiet_nominations_are_not_consumed(self):
        ghost = _ghost()
        batch = ghost.begin_batch("belief-1")
        receipt = apply_belief_proposals(
            batch,
            [
                nomination("I am unreliable.", status="deferred"),
                nomination("I am unreliable.", nomination_id="n2", status="stale"),
                AttentionNomination(nomination_id="n3", codelet="goal_pressure",
                                   attention_kind="action_tendency", status="selected",
                                   text="I want to work on the story."),
                nomination("", nomination_id="n4"),
            ],
            companion_name=ghost.get_companion_name(), stale_ids={"n2"},
        )
        ghost.commit(batch)
        self.assertEqual(_beliefs(ghost), [])
        self.assertEqual(receipt["accepted_ids"], [])

    def test_a_target_outside_the_view_is_dropped_visibly(self):
        ghost = _ghost()
        batch = ghost.begin_batch("belief-1")
        receipt = apply_belief_proposals(
            batch, [nomination("Something happened once.", targets=[9999])],
            companion_name=ghost.get_companion_name(),
        )
        ghost.commit(batch)
        beliefs = _beliefs(ghost)
        self.assertEqual(len(beliefs), 1)
        # Missing evidence stays missing: no supporting id is invented.
        self.assertEqual(beliefs[0].supporting_knoxel_ids, [])
        self.assertEqual(beliefs[0].metadata["dropped_target_ids"], [9999])
        # An interpretation without evidence starts weaker, not equal.
        self.assertLess(beliefs[0].confidence, 0.45)


if __name__ == "__main__":
    unittest.main()
