"""The EMA mental state over feature deltas (D27)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from pm.records.enums import FeatureType
from pm.records.memory_records import Feature
from pm.records.mental_state import (
    VectorModelRelationshipEntityID,
    MentalStateSample,
    PersonalityProfile,
    StateEmotions,
    StateNeeds,
    StateRelationship,
    compute_mental_state,
    compute_partner_expectation_profile,
    compute_relationship_state,
    create_empty_ms_vector,
    derive_expression,
    feature_state_sample,
    features_to_history,
    reference_half_life_seconds,
)

T0 = datetime(2026, 1, 1)
JOY = StateEmotions.model_fields["joy"].json_schema_extra["vector_position"]
SADNESS = StateEmotions.model_fields["sadness"].json_schema_extra["vector_position"]
CONNECTION = StateNeeds.model_fields["connection"].json_schema_extra["vector_position"]
TRUST = StateRelationship.model_fields["trust"].json_schema_extra["vector_position"]
HOUR = 3600.0


def _noise(count: int, *, entity: int | None = None) -> list[MentalStateSample]:
    """Return small alternating deltas, one every ten minutes. They set the deviation scale."""
    return [MentalStateSample(timestamp=T0 + timedelta(minutes=10 * index),
                              entity_ids=[entity] if entity is not None else [],
                              delta={JOY: 0.05 * (-1) ** index, TRUST: 0.05 * (-1) ** index})
            for index in range(count)]


def _state(samples, now, half_life=12 * HOUR):
    return compute_mental_state(samples, now=now, half_life_s=half_life, personality=PersonalityProfile())


class EmaStateTests(unittest.TestCase):
    def test_empty_history_reads_the_field_defaults(self):
        state = _state([], T0)
        self.assertEqual(state.state_emotions.joy, 0.0)
        self.assertEqual(state.state_needs.connection, 0.5)
        self.assertEqual(state.appraisal_general.certainty, 0.5)
        self.assertIsNone(state.state_relationship)

    def test_a_new_event_raises_its_axis_over_a_small_history(self):
        history = _noise(12)
        event = T0 + timedelta(hours=3)
        joyful = [*history, MentalStateSample(timestamp=event, delta={JOY: 0.5, SADNESS: -0.2})]
        state = _state(joyful, event)
        # One delta of 0.5 over the scale floor reads as tanh(0.5 / (2 * floor)).
        # ALPHA RUN: the floor is 0.35 (was 0.25), so about 0.61 (was 0.76).
        self.assertGreater(state.state_emotions.joy, 0.55)
        # An emotion cannot fall below its default of 0.
        self.assertEqual(state.state_emotions.sadness, 0.0)
        # An axis without deltas stays at its default.
        self.assertEqual(state.state_needs.connection, 0.5)

    def test_a_negative_need_delta_reads_below_the_need_default(self):
        event = T0 + timedelta(hours=3)
        samples = [*_noise(12), MentalStateSample(timestamp=event, delta={CONNECTION: -0.4})]
        self.assertLess(_state(samples, event).state_needs.connection, 0.5)

    def test_elapsed_time_lets_the_feeling_fade(self):
        event = T0 + timedelta(hours=3)
        samples = [*_noise(12), MentalStateSample(timestamp=event, delta={JOY: 0.5})]
        values = [_state(samples, event + timedelta(hours=hours)).state_emotions.joy for hours in (0, 6, 24, 96)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreater(values[0], 0.55)  # ALPHA RUN: sigma floor 0.35.
        self.assertLess(values[-1], 0.1)

    def test_an_event_after_a_long_gap_still_shows(self):
        # The old order updated the EMA after the delta. Then a gap of several
        # half-lives erased the new event at once.
        event = T0 + timedelta(days=30)
        samples = [*_noise(12), MentalStateSample(timestamp=event, delta={JOY: 0.5})]
        self.assertGreater(_state(samples, event).state_emotions.joy, 0.55)  # ALPHA RUN: sigma floor 0.35.

    def test_a_longer_half_life_keeps_the_feeling_longer(self):
        event = T0 + timedelta(hours=3)
        samples = [*_noise(12), MentalStateSample(timestamp=event, delta={JOY: 0.5})]
        later = event + timedelta(hours=12)
        short = _state(samples, later, half_life=6 * HOUR).state_emotions.joy
        long = _state(samples, later, half_life=48 * HOUR).state_emotions.joy
        self.assertLess(short, long)

    def test_the_half_life_ignores_ego_strength_for_now(self):
        self.assertEqual(reference_half_life_seconds(None), 24 * HOUR)
        self.assertEqual(reference_half_life_seconds(0.5), 24 * HOUR)

    def test_personality_comes_from_the_persona(self):
        profile = PersonalityProfile(neuroticism=0.9)
        state = compute_mental_state([], now=T0, half_life_s=HOUR, personality=profile)
        self.assertEqual(state.state_personality, profile)


class LatentStateTests(unittest.TestCase):
    def test_appraisal_and_cognition_deltas_do_not_add_up(self):
        ego = 202  # StateCognition.ego_strength
        samples = [MentalStateSample(timestamp=T0 + timedelta(minutes=index), delta={ego: 0.1}) for index in range(5)]
        state = _state(samples, T0 + timedelta(minutes=5))
        self.assertEqual(state.state_cognition.ego_strength, 0.5)


class PersonalityTests(unittest.TestCase):
    def test_persona_vector_round_trips_and_defaults_to_neutral_traits(self):
        from pm.records.cycle_state import PersonaConfig
        profile = PersonalityProfile(neuroticism=0.9, humor_playfulness=0.2)
        self.assertEqual(PersonalityProfile.init_from_vector(profile.to_vector()), profile)
        config = PersonaConfig(user_name="User", companion_name="Companion")
        self.assertEqual(PersonalityProfile.init_from_vector(config.personality_profile), PersonalityProfile())


class ExpressionTests(unittest.TestCase):
    def test_expression_follows_the_state(self):
        state = _state([], T0)
        state.state_emotions.anger = 0.9
        state.state_core.arousal = 0.9
        state.state_core.dominance = 0.8
        derive_expression(state)
        self.assertGreater(state.state_expression.aggressiveness, 0.0)
        self.assertGreater(state.state_expression_ext.confrontive_push, 0.0)


class RelationshipTests(unittest.TestCase):
    def test_each_entity_reads_only_its_own_features(self):
        event = T0 + timedelta(hours=3)
        history = [*_noise(12, entity=1), *_noise(12, entity=2),
                   MentalStateSample(timestamp=event, entity_ids=[1], delta={TRUST: 0.4}),
                   MentalStateSample(timestamp=event, entity_ids=[2], delta={TRUST: -0.4})]
        first = compute_relationship_state(history, 1, now=event, half_life_s=12 * HOUR)
        second = compute_relationship_state(history, 2, now=event, half_life_s=12 * HOUR)
        self.assertEqual(first.entity_id, 1)
        self.assertGreater(first.trust, 0.5)
        self.assertEqual(second.entity_id, 2)
        self.assertLess(second.trust, -0.5)
        nobody = compute_relationship_state(history, 3, now=event, half_life_s=12 * HOUR)
        self.assertEqual(nobody.trust, 0.0)

    def test_partner_profile_reads_the_relationship_deltas(self):
        good = [MentalStateSample(timestamp=T0 + timedelta(minutes=index), entity_ids=[1], delta={TRUST: 0.5})
                for index in range(5)]
        profile = compute_partner_expectation_profile(good, 1)
        self.assertEqual(profile.sample_count, 5)
        self.assertGreater(profile.baseline_quality, 0.0)
        self.assertEqual(compute_partner_expectation_profile(good, 2).sample_count, 0)


class FeatureSampleTests(unittest.TestCase):
    def test_sample_keeps_the_delta_the_entity_and_the_partner(self):
        appraisal = create_empty_ms_vector()
        appraisal[VectorModelRelationshipEntityID] = 1
        delta = create_empty_ms_vector()
        delta[JOY] = 0.3
        delta[VectorModelRelationshipEntityID] = 0.4
        feature = Feature(content="x", feature_type=FeatureType.StateUpdate, causal=True, entity_id=2,
                          timestamp_world_begin=datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=2))),
                          mental_state_appraisal_vec=appraisal, mental_state_delta_vec=delta)
        sample = feature_state_sample(feature)
        self.assertEqual(sample.delta, {JOY: 0.3})
        self.assertEqual(sample.entity_ids, [2, 1])
        self.assertEqual(sample.timestamp, datetime(2026, 1, 1, 10))

    def test_history_holds_only_causal_features_oldest_first(self):
        late = Feature(content="late", feature_type=FeatureType.Dialogue, causal=True,
                       timestamp_world_begin=T0 + timedelta(hours=1))
        early = Feature(content="early", feature_type=FeatureType.Dialogue, causal=True, timestamp_world_begin=T0)
        hidden = Feature(content="hidden", feature_type=FeatureType.Dialogue, causal=False, timestamp_world_begin=T0)
        history = features_to_history([late, hidden, early])
        self.assertEqual([item.timestamp for item in history], [T0, T0 + timedelta(hours=1)])


if __name__ == "__main__":
    unittest.main()
