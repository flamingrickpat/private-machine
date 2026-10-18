"""Affect of the main cycle: appraisal state, EMA mental state, regulation, and words (D27).

No live model. The pure functions run directly. The cycle tests run the real
kernel and the real module adapter with a scripted model.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.app.lifecycle_records import record_shutdown, record_startup
from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, new_id
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.store import MemoryStore
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureSubtype, FeatureType
from pm.records.memory_records import ActionSchemaRecord, Feature
from pm.records.mental_state import (
    VectorModelRelationshipEntityID,
    AppraisalGeneral,
    StateNeeds,
    FullMentalState,
    MentalStateSample,
    StateEmotions,
    StateRelationship,
    compute_mental_state,
    create_empty_state,
    derive_expression,
    feature_state_sample,
    reference_half_life_seconds,
    sparse_delta,
)
from pm.records.workspace import ShardKey
from pm.runs.workflow_module import PROJECT_ROOT
from tests.support import ManualClock
from tests.unit.workflows.main_cognition.cycle_support import ScriptedModel, scripted_tools
from workflows.main_cognition.affect import (
    REGULATION_BLENDED,
    REGULATION_REACTIVE,
    REGULATION_REFLECTIVE,
    appraise_state,
    describe_mental_state,
    regulation_mode,
)
from workflows.main_cognition.codelet_schedule import state_signals
from workflows.main_cognition.cycle_work import CycleRequest, CycleWork
from workflows.main_cognition.host import MainCogLoop
from workflows.main_cognition.phases.appraise import update_affect
from workflows.main_cognition.schemas import AppraisalResult

REAL_MANIFEST = PROJECT_ROOT / "workflows" / "main_cognition" / "main_cognition.workflow.md"
JOY = StateEmotions.model_fields["joy"].json_schema_extra["vector_position"]
TRUST = StateRelationship.model_fields["trust"].json_schema_extra["vector_position"]
CONNECTION = StateNeeds.model_fields["connection"].json_schema_extra["vector_position"]
CERTAINTY = AppraisalGeneral.model_fields["certainty"].json_schema_extra["vector_position"]


def _small_ghost():
    from tests.unit.test_design_contracts import small_ghost
    return small_ghost()


def _store(ghost, content: str) -> int:
    return ghost.store_knoxel(Feature(
        content=content, feature_type=FeatureType.Dialogue, causal=True,
        override_source=ghost.get_user_name(),
    )).id


def _reports(stimulus_id: int, *, appraisal=None, emotion=None, need=None) -> dict:
    appraisal = appraisal or {}
    return {
        "triage": {"items": [{"knoxel_id": stimulus_id, "category": "Moderate"}], "summary": "t"},
        "appraisal": {
            "feeling": appraisal.get("feeling", "steady"),
            "appraisal": dict(appraisal.get("appraisal", {})),
            "expectation_error": float(appraisal.get("expectation_error", 0.0)),
            "emotion_delta": dict(emotion or {}),
            "need_delta": dict(need or {}),
            "reasoning": "evidence",
            "evidence_ids": [stimulus_id],
        },
        "context": {"memories": [], "candidate_ids": [], "summary": "c"},
        "codelets": {"proposals": [], "runs": [], "summary": "codelets"},
        "attention": {"selected_ids": [stimulus_id], "nominations": [], "summary": "n"},
        "goals": {"changes": [], "summary": "g"},
        "workspace": {"frame_reason": "reply", "focus_ids": [stimulus_id],
                      "control_mode": "reflective", "summary": "w"},
        "action_selection": {
            "actions": [{"action_type": "Reply", "workflow_name": "dialogue_story",
                         "content": "reply to Jesse", "arguments": {}, "depends_on": []}],
            "summary": "s",
        },
    }


def _module(ghost, reports, tmp):
    model = ScriptedModel(reports=reports)
    bus = BusSystem()
    errors: list[BaseException] = []
    module = MainCogLoop(
        "main_cog", bus, errors.append, ManualClock(), None,
        working_target=MemoryBashTarget(ghost, "read"),
        workflow=REAL_MANIFEST,
        tools_factory=scripted_tools(model),
        runtime_root=Path(tmp) / "runtime",
        ghost=ghost,
    )
    bus.subscribe(module.module_id, module.accepts, module.deliver)
    return module, bus, errors, model


def _cycle(ghost, reports, stimulus_id, *, source="user_input", topic="user.input"):
    with TemporaryDirectory() as tmp:
        module, bus, errors, model = _module(ghost, reports, tmp)
        bus.put(Envelope(id=new_id(), source=source, kind="data", topic=topic,
                         correlation_id="corr-affect", payload={"knoxel_id": stimulus_id}))
        assert module.step(block=False)
        assert errors == [], errors
        return CycleWork.model_validate(module.last_result.report), model


def _appraisal_features(ghost) -> list[Feature]:
    return [item for item in ghost.all_features if item.feature_subtype == FeatureSubtype.Appraisal]


class AppraisalResultTests(unittest.TestCase):
    def test_axes_cast_into_the_typed_appraisal_models(self):
        result = AppraisalResult(appraisal={"goal_congruence": 0.6, "perceived_warmth": 0.8})
        self.assertEqual(result.general().goal_congruence, 0.6)
        self.assertEqual(result.general().certainty, 0.5)
        self.assertEqual(result.social().perceived_warmth, 0.8)

    def test_unknown_names_and_out_of_bounds_axes_are_invalid_model_output(self):
        for bad in ({"appraisal": {"excitement": 0.5}}, {"emotion_delta": {"excitement": 0.5}},
                    {"need_delta": {"snacks": 0.5}}, {"appraisal": {"certainty": 1.5}}):
            with self.assertRaises(ValueError):
                AppraisalResult.model_validate(bad)


class AppraiseStateTests(unittest.TestCase):
    def test_axes_derive_emotions_and_the_delta_is_the_change(self):
        start = create_empty_state()
        good = AppraisalResult(appraisal={"goal_congruence": 0.9, "control_self": 0.8, "certainty": 0.8})
        state, delta = appraise_state(start, good, entity_id=None)
        self.assertGreater(state.state_emotions.joy, 0.3)
        self.assertAlmostEqual(delta[JOY], state.state_emotions.joy - start.state_emotions.joy)
        self.assertIsNone(state.state_relationship)

    def test_explicit_deltas_and_expectation_error_add_to_the_derived_values(self):
        start = FullMentalState()
        start.state_emotions.boredom = 0.3
        answer = AppraisalResult(emotion_delta={"boredom": 0.4}, need_delta={"connection": -0.3},
                                 expectation_error=-1.0)
        state, delta = appraise_state(start, answer, entity_id=None)
        # No appraisal axis derives boredom, so the explicit delta adds to the current value.
        self.assertAlmostEqual(state.state_emotions.boredom, 0.7)
        self.assertAlmostEqual(state.state_needs.connection, 0.2)
        self.assertLess(state.state_core.valence, 0.0)
        self.assertGreater(state.state_emotions.sadness, 0.0)
        boredom = StateEmotions.model_fields["boredom"].json_schema_extra["vector_position"]
        self.assertAlmostEqual(delta[boredom], 0.4)

    def test_values_stay_inside_their_bounds(self):
        answer = AppraisalResult(emotion_delta={"joy": 0.9, "propriety_inhibition": -3.0})
        state, _ = appraise_state(create_empty_state(), answer, entity_id=None)
        self.assertLessEqual(state.state_emotions.joy, 1.0)
        self.assertEqual(state.state_emotions.propriety_inhibition, -1.0)
        FullMentalState.model_validate(state.model_dump())

    def test_social_appraisal_names_the_partner_but_the_delta_has_no_id(self):
        social = AppraisalResult(appraisal={"norm_violation": 0.9, "contempt": 0.8, "agency_other": 1.0})
        state, delta = appraise_state(create_empty_state(), social, entity_id=3)
        self.assertEqual(state.state_relationship.entity_id, 3)
        self.assertLess(state.state_relationship.trust, 0)
        self.assertGreater(state.state_relationship.conflict_tension, 0)
        self.assertEqual(state.to_list()[VectorModelRelationshipEntityID], 3)
        self.assertEqual(delta[VectorModelRelationshipEntityID], 0.0)


class RegulationTests(unittest.TestCase):
    def test_danger_preempts_and_sadness_reflects(self):
        calm = create_empty_state()
        danger, _ = appraise_state(calm, AppraisalResult(appraisal={"bodily_threat": 0.95}), entity_id=None)
        self.assertEqual(regulation_mode(danger, calm), REGULATION_REACTIVE)
        sad = create_empty_state()
        sad.state_emotions.sadness = 0.8
        self.assertEqual(regulation_mode(calm, sad), REGULATION_REFLECTIVE)
        self.assertEqual(regulation_mode(calm, calm), REGULATION_BLENDED)


class DescribeTests(unittest.TestCase):
    def test_text_names_the_strongest_feelings_without_numbers(self):
        state = FullMentalState()
        state.state_emotions.joy = 0.8
        state.state_emotions.curiosity = 0.45
        state.state_emotions.boredom = 0.2
        state.state_core.valence = 0.5
        state.state_needs.connection = 0.1
        state.state_relationship = StateRelationship(entity_id=1, affection=0.6)
        text = describe_mental_state(state, name="Companion", partner_name="User")
        self.assertIn("Companion feels strong joy, along with moderate curiosity and mild boredom.", text)
        self.assertIn("good spirits", text)
        self.assertIn("Toward User there is moderate affection.", text)
        self.assertIn("longs for connection", text)
        self.assertNotRegex(text, r"\d")

    def test_neutral_state_reads_calm(self):
        state = FullMentalState()
        derive_expression(state)
        text = describe_mental_state(state, name="Companion")
        self.assertEqual(text, "Companion feels calm and emotionally even.")

    def test_text_names_the_urge_to_act_and_the_attention_drift(self):
        state = FullMentalState()
        state.state_emotions.anger = 0.9
        state.state_emotions.fear = 0.6
        state.state_core.valence = -0.6
        state.state_core.dominance = -0.5
        state.state_relationship = StateRelationship(entity_id=1, trust=-0.8, aversive_association=0.7)
        derive_expression(state)
        text = describe_mental_state(state, name="Companion", partner_name="User")
        self.assertIn("Companion wants to go quiet and pull back.", text)
        self.assertIn("Companion's attention drifts toward threat and loss.", text)
        self.assertIn("Toward User there is strong distrust", text)


class SchedulerSignalTests(unittest.TestCase):
    def test_signals_read_the_typed_state_and_the_attention_bias(self):
        state = FullMentalState()
        state.state_emotions.sadness = 0.6
        state.state_core.valence = -0.4
        derive_expression(state)
        flat = state_signals(state)
        self.assertEqual(flat["emotion.sadness"], 0.6)
        self.assertEqual(flat["core.negative_valence"], 0.4)
        self.assertEqual(flat["emotion.negative_strongest"], 0.6)
        self.assertLess(flat["bias.valence_focus"], 0.0)
        self.assertIn("bias.salience", flat)
        self.assertIn("expression.disengage_silence", flat)
        self.assertNotIn("relationship.trust", flat)
        state.state_relationship = StateRelationship(entity_id=4, trust=0.5)
        flat = state_signals(state)
        self.assertEqual(flat["relationship.trust"], 0.5)
        self.assertNotIn("relationship.entity_id", flat)


class UpdateAffectTests(unittest.TestCase):
    def _work(self, history, *, entity=True):
        request = CycleRequest(companion_name="Companion", user_name="User", user_entity_id=1,
                               correlation_id="c", cycle_time=datetime(2026, 1, 2), state_history=history)
        work = CycleWork(request=request)
        key = ShardKey(kind="entity", ref="1") if entity else ShardKey(kind="self")
        work.state.stimulus_groups = [StimulusGroup(key=key, correlation_id="c")]
        return work

    def test_the_new_feature_enters_the_ema_state_with_the_partner_relationship(self):
        history = [MentalStateSample(timestamp=datetime(2026, 1, 1, hour), entity_ids=[1],
                                     delta={JOY: 0.05 * (-1) ** hour, TRUST: 0.02 * (-1) ** hour})
                   for hour in range(10)]
        work = self._work(history)
        answer = AppraisalResult(appraisal={"goal_congruence": 0.9, "control_self": 0.9, "certainty": 0.9,
                                            "trust_cues": 0.9, "fairness": 0.9})
        affect = update_affect(work, answer)
        state = work.state.mental_state
        self.assertGreater(state.state_emotions.joy, 0.5)
        self.assertEqual(state.state_relationship.entity_id, 1)
        self.assertGreater(state.state_relationship.trust, 0.0)
        self.assertEqual(affect.entity_id, 1)
        self.assertGreater(affect.delta_vec[JOY], 0.0)
        self.assertIn("joy", affect.felt)
        self.assertIn("Toward User", affect.felt)

    def test_elapsed_time_between_cycles_lets_an_untouched_feeling_fade(self):
        event = datetime(2026, 1, 1, 12)
        history = [*(MentalStateSample(timestamp=datetime(2026, 1, 1, hour), delta={CONNECTION: 0.02 * (-1) ** hour})
                     for hour in range(10)),
                   MentalStateSample(timestamp=event, delta={CONNECTION: -0.4})]
        soon = self._work(history, entity=False)
        soon.request.cycle_time = event
        later = self._work(history, entity=False)
        later.request.cycle_time = datetime(2026, 1, 4, 12)
        # The appraisal names no need, so the raw need level stays. Only the
        # EMA baseline moves with the elapsed time.
        update_affect(soon, AppraisalResult())
        update_affect(later, AppraisalResult())
        self.assertLess(soon.state.mental_state.state_needs.connection, 0.2)
        self.assertGreater(later.state.mental_state.state_needs.connection,
                           soon.state.mental_state.state_needs.connection)

    def test_the_delta_is_measured_from_the_current_readout(self):
        history = [MentalStateSample(timestamp=datetime(2026, 1, 1, hour), delta={JOY: 0.3}) for hour in range(2)]
        work = self._work(history, entity=False)
        current = compute_mental_state(history, now=work.request.cycle_time,
                                       half_life_s=reference_half_life_seconds(), personality=work.request.personality)
        affect = update_affect(work, AppraisalResult(emotion_delta={"joy": 0.2}))
        # The neutral axes derive no joy, so the explicit delta sets it to
        # 0.2. The delta is the step from the current readout to that value.
        self.assertAlmostEqual(affect.appraisal_state.state_emotions.joy, 0.2)
        self.assertAlmostEqual(affect.delta_vec[JOY], 0.2 - current.state_emotions.joy)
        self.assertEqual(affect.delta_vec[CERTAINTY], 0.0)

    def test_the_persona_personality_reaches_the_state(self):
        work = self._work([])
        work.request.personality.neuroticism = 0.9
        affect = update_affect(work, AppraisalResult())
        self.assertEqual(work.state.mental_state.state_personality.neuroticism, 0.9)
        self.assertEqual(affect.appraisal_state.state_personality.neuroticism, 0.9)

    def test_a_self_group_has_no_relationship(self):
        work = self._work([], entity=False)
        affect = update_affect(work, AppraisalResult(appraisal={"novelty": 0.8}))
        self.assertIsNone(work.state.mental_state.state_relationship)
        self.assertIsNone(affect.entity_id)

    def test_affect_survives_the_phase_state_round_trip(self):
        work = self._work([])
        work.request.state_history = [MentalStateSample(timestamp=datetime(2026, 1, 1), delta={JOY: 0.8})]
        work.affect = update_affect(work, AppraisalResult(emotion_delta={"joy": -0.7}))
        self.assertTrue(any(value < 0 for value in work.affect.delta_vec))
        restored = CycleWork.model_validate_json(work.model_dump_json())
        self.assertEqual(restored.affect.delta_vec, work.affect.delta_vec)
        self.assertEqual(restored.state.mental_state.to_list(), work.state.mental_state.to_list())


class CycleTests(unittest.TestCase):
    def test_appraisal_feature_carries_both_vectors_and_the_committed_state_is_the_ema_state(self):
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "I let you down.")
        reports = _reports(stimulus_id, appraisal={
            "feeling": "hurt", "appraisal": {"goal_congruence": -0.7, "certainty": 0.3},
            "expectation_error": -0.6})
        work, _ = _cycle(ghost, reports, stimulus_id)
        feature = _appraisal_features(ghost)[0]
        self.assertGreater(feature.mental_state_appraisal.state_emotions.sadness, 0.0)
        self.assertEqual(feature.mental_state_appraisal_vec, work.affect.appraisal_state.to_list())
        self.assertEqual(feature.mental_state_delta_vec, work.affect.delta_vec)
        self.assertTrue(feature.causal)
        self.assertEqual(feature.metadata["regulation"], work.affect.regulation)
        self.assertEqual(feature.metadata["affect_summary"], work.affect.felt)
        committed = ghost.states[-1].mental_state
        self.assertEqual(committed.to_list(), work.state.mental_state.to_list())
        self.assertGreater(committed.state_emotions.sadness, 0.0)
        action = [item for item in ghost.all_knoxels.values() if isinstance(item, ActionSchemaRecord)][-1]
        self.assertEqual(action.metadata["regulation"], feature.metadata["regulation"])
        self.assertEqual(action.metadata["affect_summary"], work.affect.felt)

    def test_the_next_cycle_reads_the_committed_feature_in_its_history(self):
        ghost = _small_ghost()
        first = _store(ghost, "good news")
        _cycle(ghost, _reports(first, emotion={"joy": 0.5}), first)
        feature = _appraisal_features(ghost)[0]
        second = _store(ghost, "more news")
        work, _ = _cycle(ghost, _reports(second), second)
        samples = work.request.state_history
        self.assertIn(feature_state_sample(feature), samples)
        self.assertEqual(sparse_delta(feature.mental_state_delta_vec)[JOY], 0.5)

    def test_danger_makes_the_frame_reactive_and_later_prompts_read_it(self):
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "the kitchen is on fire")
        reports = _reports(stimulus_id, appraisal={
            "feeling": "alarmed", "appraisal": {"bodily_threat": 0.9, "goal_congruence": -0.8}})
        work, model = _cycle(ghost, reports, stimulus_id)
        self.assertEqual(work.affect.regulation, REGULATION_REACTIVE)
        goals = model.messages[model.calls.index("goals")][-1].content.text
        self.assertIn(REGULATION_REACTIVE, goals)
        self.assertIn(work.affect.felt, goals)
        frame = ghost.all_attention_frames[-1]
        self.assertEqual(frame.control_mode, REGULATION_REACTIVE)
        self.assertIn("preempt", frame.frame_reason.lower())

    def test_explicit_need_delta_reaches_the_ema_state(self):
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "I am here with you.")
        _cycle(ghost, _reports(stimulus_id, need={"connection": -0.3}), stimulus_id)
        self.assertLess(ghost.states[-1].mental_state.state_needs.connection, 0.5)

    def test_affect_survives_a_checkpoint_restart(self):
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ghost.json"
            ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=checkpoint)
            stimulus_id = _store(ghost, "That failure was not your fault.")
            _cycle(ghost, _reports(stimulus_id, emotion={"sadness": 0.7}), stimulus_id)
            before = ghost.states[-1].mental_state
            after = MemoryStore.from_checkpoint(checkpoint).states[-1].mental_state
            self.assertAlmostEqual(after.state_emotions.sadness, before.state_emotions.sadness)
            self.assertGreater(after.state_emotions.sadness, 0.0)

    def test_service_failure_startup_event_is_appraised_not_neutral(self):
        ghost = _small_ghost()
        boot = record_startup(ghost, boot_id="boot-affect")
        record_shutdown(ghost, boot_id="boot-affect")
        reports = _reports(boot["knoxel_id"], appraisal={
            "feeling": "uncertain", "appraisal": {"certainty": 0.1, "novelty": 0.6}})
        _cycle(ghost, reports, boot["knoxel_id"], source="system", topic="system.lifecycle")
        feature = _appraisal_features(ghost)[0]
        self.assertLess(feature.mental_state_appraisal.appraisal_general.certainty, 0.5)
        self.assertGreater(feature.mental_state_appraisal.state_emotions.surprise, 0.0)


if __name__ == "__main__":
    unittest.main()
