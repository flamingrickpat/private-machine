"""ALPHA RUN ONLY: tests of the temporary failsafe clamps.

Delete this file together with the ALPHA block in ``pm.constants``.
"""
from __future__ import annotations

import unittest

from pm.constants import ALPHA_MAX_RATING_STEP, ALPHA_MAX_STATE_STEP, ALPHA_MAX_STIMULUS_DELTA
from pm.records.codelet_records import CodeletRuntimeState
from pm.records.mental_state import (
    VectorModelReservedSize,
    VectorModelRelationshipEntityID,
    FullMentalState,
    StateCore,
    StateEmotions,
    StateRelationship,
    alpha_clamp_delta,
    alpha_limit_state_step,
)
from workflows.main_cognition.codelet_registry import add_rating

VALENCE = StateCore.model_fields["valence"].json_schema_extra["vector_position"]
BOREDOM = StateEmotions.model_fields["boredom"].json_schema_extra["vector_position"]


class AlphaClampTests(unittest.TestCase):
    def test_stimulus_delta_is_clamped_but_entity_id_stays(self):
        vector = [0.0] * VectorModelReservedSize
        vector[VALENCE] = -1.8
        vector[BOREDOM] = 0.9
        vector[VectorModelRelationshipEntityID] = 7
        clamped = alpha_clamp_delta(vector)
        self.assertAlmostEqual(clamped[VALENCE], -ALPHA_MAX_STIMULUS_DELTA)
        self.assertAlmostEqual(clamped[BOREDOM], ALPHA_MAX_STIMULUS_DELTA)
        self.assertEqual(clamped[VectorModelRelationshipEntityID], 7)

    def test_state_step_is_limited_for_valence_and_boredom(self):
        previous = FullMentalState()
        previous.state_core.valence = 0.78
        state = FullMentalState()
        state.state_core.valence = -1.0
        state.state_emotions.boredom = 0.95
        alpha_limit_state_step(previous, state)
        self.assertAlmostEqual(state.state_core.valence, 0.78 - ALPHA_MAX_STATE_STEP)
        self.assertAlmostEqual(state.state_emotions.boredom, ALPHA_MAX_STATE_STEP)

    def test_relationship_is_limited_only_for_the_same_partner(self):
        previous = FullMentalState(state_relationship=StateRelationship(entity_id=2, trust=0.0))
        same = FullMentalState(state_relationship=StateRelationship(entity_id=2, trust=1.0))
        other = FullMentalState(state_relationship=StateRelationship(entity_id=3, trust=1.0))
        alpha_limit_state_step(previous, same)
        alpha_limit_state_step(previous, other)
        self.assertAlmostEqual(same.state_relationship.trust, ALPHA_MAX_STATE_STEP)
        self.assertEqual(same.state_relationship.entity_id, 2)
        self.assertEqual(other.state_relationship.trust, 1.0)

    def test_selector_rating_moves_a_third_per_cycle(self):
        runtime = CodeletRuntimeState(name="BoredomThresholdCross")
        self.assertEqual(add_rating(runtime, 0.1), 0.1)
        self.assertAlmostEqual(add_rating(runtime, 0.95), 0.1 + ALPHA_MAX_RATING_STEP)
        self.assertAlmostEqual(add_rating(runtime, 0.0), 0.1)


if __name__ == "__main__":
    unittest.main()
