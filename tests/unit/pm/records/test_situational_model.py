"""CSM mechanics (A19F step 2): decay, merge, boost, prune, and snapshot. No model calls."""
from __future__ import annotations

import random
import unittest

from pm.constants import (CSM_BOOST, CSM_CAPACITY, CSM_DECAY, CSM_MIN_ACTIVATION)
from pm.records.situational_model import SituationalItem, SituationalModel
from pm.records.workspace import ShardKey

# Record 1 and 2 are near duplicates. Record 3 points in another direction.
EMBEDDINGS = {1: [1.0, 0.0, 0.0], 2: [0.99, 0.05, 0.0], 3: [0.0, 1.0, 0.0], 4: [0.0, 0.0, 1.0]}


def embedding_of(record_id: int) -> list[float]:
    return EMBEDDINGS[record_id]


class DecayTests(unittest.TestCase):
    def test_neutral_base_decays_by_csm_decay(self):
        model = SituationalModel()
        model.add_or_boost(SituationalItem(record_id=1, activation=1.0), "retrieval", embedding_of)
        model.decay_step()
        self.assertAlmostEqual(model.items[1].activation, CSM_DECAY)
        self.assertEqual(model.items[1].ticks_present, 1)

    def test_high_base_decays_slower_than_low_base(self):
        model = SituationalModel(items={
            3: SituationalItem(record_id=3, activation=1.0, base_activation=0.1),
            4: SituationalItem(record_id=4, activation=1.0, base_activation=0.9),
        })
        model.decay_step()
        self.assertLess(model.items[3].activation, CSM_DECAY)
        self.assertGreater(model.items[4].activation, CSM_DECAY)


class MergeTests(unittest.TestCase):
    def test_similar_record_merges_and_boosts(self):
        model = SituationalModel()
        model.add_or_boost(SituationalItem(record_id=1, activation=0.4, entered_tick=1), "retrieval", embedding_of)
        held = model.add_or_boost(SituationalItem(record_id=2, activation=0.9, entered_tick=2), "BoredomThresholdCross",
                                  embedding_of)
        self.assertEqual(list(model.items), [1])
        self.assertIs(held, model.items[1])
        self.assertEqual(held.merged_ids, [2])
        self.assertAlmostEqual(held.activation, 0.4 + CSM_BOOST)
        self.assertEqual(held.last_boost_tick, 2)

    def test_dissimilar_record_enters_as_new_item(self):
        model = SituationalModel()
        model.add_or_boost(SituationalItem(record_id=1, activation=0.4), "retrieval", embedding_of)
        model.add_or_boost(SituationalItem(record_id=3, activation=0.4, shard_hint=ShardKey(kind="entity", ref="2")),
                           "retrieval", embedding_of)
        self.assertEqual(sorted(model.items), [1, 3])
        self.assertEqual(model.items[3].shard_hint, ShardKey(kind="entity", ref="2"))

    def test_several_endorsers_raise_activation_and_are_recorded(self):
        model = SituationalModel()
        model.add_or_boost(SituationalItem(record_id=3, activation=0.2), "retrieval", embedding_of)
        for endorser in ("AbsurditySampler", "HumorousRemarkPlanner", "AbsurditySampler"):
            model.add_or_boost(SituationalItem(record_id=3), endorser, embedding_of)
        item = model.items[3]
        self.assertEqual(item.endorsers, ["retrieval", "AbsurditySampler", "HumorousRemarkPlanner"])
        self.assertAlmostEqual(item.activation, min(1.0, 0.2 + 3 * CSM_BOOST))
        self.assertGreater(item.base_activation, SituationalItem().base_activation)
        self.assertEqual(item.merged_ids, [])


class PruneTests(unittest.TestCase):
    def test_low_items_leave_unless_protected(self):
        weak = CSM_MIN_ACTIVATION / 2
        model = SituationalModel(items={
            1: SituationalItem(record_id=1, activation=weak),
            3: SituationalItem(record_id=3, activation=weak, merged_ids=[9]),
            4: SituationalItem(record_id=4, activation=0.8),
        })
        removed = model.prune(protected_ids={9})
        self.assertEqual(removed, [1])
        self.assertEqual(sorted(model.items), [3, 4])

    def test_capacity_drops_lowest_unprotected_items(self):
        count = CSM_CAPACITY + 3
        model = SituationalModel(items={
            index: SituationalItem(record_id=index, activation=0.2 + index / (2 * count)) for index in range(count)
        })
        removed = model.prune(protected_ids={0})
        self.assertEqual(removed, [1, 2, 3])
        self.assertEqual(len(model.items), CSM_CAPACITY)
        self.assertIn(0, model.items)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.model = SituationalModel(items={
            1: SituationalItem(record_id=1, activation=0.5),
            3: SituationalItem(record_id=3, activation=0.5, salience=1.0, valence=-0.8),
            4: SituationalItem(record_id=4, activation=0.5),
        })

    def test_exploit_mode_orders_by_relevance_and_respects_budget(self):
        view = self.model.snapshot_for([1.0, 0.1, 0.0], embedding_of, lambda record_id: 10,
                                       attention_bias=(0.0, 1.0), max_tokens=20, rng=random.Random(1))
        self.assertEqual(view, [1, 3])

    def test_precursor_bonus_lifts_an_irrelevant_item(self):
        view = self.model.snapshot_for([1.0, 0.0, 0.0], embedding_of, lambda record_id: 10,
                                       attention_bias=(0.0, 1.0), max_tokens=100, rng=random.Random(1),
                                       precursor_ids={4})
        self.assertEqual(view[:2], [1, 4])

    def test_sampling_mode_is_repeatable_with_a_seed(self):
        views = [self.model.snapshot_for([0.5, 0.5, 0.5], embedding_of, lambda record_id: 1,
                                         attention_bias=(0.0, 0.2), max_tokens=100, rng=random.Random(7))
                 for _ in range(2)]
        self.assertEqual(views[0], views[1])
        self.assertEqual(sorted(views[0]), [1, 3, 4])


class RoundTripTests(unittest.TestCase):
    def test_json_round_trip_restores_int_keys(self):
        model = SituationalModel()
        model.add_or_boost(SituationalItem(record_id=3, activation=0.5, shard_hint=ShardKey(kind="workflow", ref="minecraft")),
                           "retrieval", embedding_of)
        restored = SituationalModel.model_validate(model.model_dump(mode="json"))
        self.assertEqual(restored, model)
        self.assertEqual(list(restored.items), [3])


if __name__ == "__main__":
    unittest.main()
