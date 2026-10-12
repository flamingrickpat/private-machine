"""A07 affect dynamics tests: decay, persona contrast, traces, and regulation.

No live model, module thread, or Minecraft server. The pure mathematics runs
directly, and the real parent kernel and the real module adapter run with a
deterministic ``run_stage`` stub.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.memory.store import MemoryStore
from workflows.main_cognition.affect_dynamics import REGULATION_REACTIVE, REGULATION_REFLECTIVE, decay_toward_baseline, personality_gains, project_affect
from pm.records.cycle_state import PersonaConfig
from pm.records.memory_records import ActionSchemaRecord, Feature
from pm.records.enums import FeatureSubtype, FeatureType
from pm.records.mental_state import FullMentalState, create_empty_state, make_personality_profile

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope, new_id
from modules import MainCogLoop
from pm.app.lifecycle_records import record_shutdown, record_startup
from tests.support import ManualClock, Recorder
from pm.runs.workflow_module import PROJECT_ROOT
from workflows.main_cognition.schemas import AppraisalResult
from pm.memory.bash_view import MemoryBashTarget

REAL_MANIFEST = PROJECT_ROOT / "workflows" / "main_cognition" / "main_cognition.workflow.md"


def _small_ghost():
    from tests.test_design_contracts import small_ghost
    return small_ghost()


def _store(ghost, content: str) -> int:
    return ghost.store_knoxel(Feature(
        content=content, feature_type=FeatureType.Dialogue, causal=True,
        override_source=ghost.get_user_name(),
    )).id


def _reports(stimulus_id: int, *, appraisal=None, emotion=None, need=None, actions=None) -> dict:
    return {
        "triage": {"items": [{"knoxel_id": stimulus_id, "category": "Moderate"}], "summary": "t"},
        "appraisal": {
            "feeling": (appraisal or {}).get("feeling", "steady"),
            "appraisal": dict((appraisal or {}).get("appraisal", {})),
            "expectation_error": float((appraisal or {}).get("expectation_error", 0.0)),
            "emotion_delta": dict(emotion or {}),
            "need_delta": dict(need or {}),
            "reasoning": "evidence",
            "evidence_ids": [stimulus_id],
            "summary": "a",
        },
        "context": {"memories": [], "candidate_ids": [], "summary": "c"},
        "attention": {"selected_ids": [stimulus_id], "nominations": [], "summary": "n"},
        "goals": {"changes": [], "summary": "g"},
        "workspace": {"frame_reason": "reply", "focus_ids": [stimulus_id],
                      "control_mode": "reflective", "summary": "w"},
        "action_selection": {
            "actions": list(actions or [{
                "action_type": "Reply", "workflow_name": "dialogue_story",
                "content": "reply to Jesse", "arguments": {}, "depends_on": [],
            }]),
            "summary": "s",
        },
    }


def _module(ghost, reports, tmp, module_id="main_cog"):
    payloads: dict[str, dict] = {}

    def run_stage(name, payload):
        payloads[name] = dict(payload)
        return dict(reports[name])

    bus = BusSystem()
    errors: list[BaseException] = []
    module = MainCogLoop(
        module_id, bus, errors.append, ManualClock(), None,
        working_target=MemoryBashTarget(ghost, "read"),
        workflow=REAL_MANIFEST,
        tools_factory=lambda environment, batch: {"run_stage": run_stage},
        runtime_root=Path(tmp) / "runtime",
        ghost=ghost,
    )
    bus.subscribe(module.module_id, module.accepts, module.deliver)
    return module, bus, errors, payloads


class AffectMathTests(unittest.TestCase):
    def test_decay_returns_mood_toward_baseline(self):
        state = create_empty_state()
        state.state_emotions.joy = 0.9
        state.state_emotions.sadness = 0.7
        state.state_core.valence = 0.8
        decayed = decay_toward_baseline(state, elapsed_seconds=3 * 60 * 60)
        self.assertLess(decayed.state_emotions.joy, 0.9)
        self.assertLess(decayed.state_emotions.sadness, 0.7)
        self.assertLess(decayed.state_core.valence, 0.8)
        self.assertGreaterEqual(decayed.state_emotions.joy, 0.0)
        self.assertLessEqual(decayed.state_emotions.joy, 1.0)

    def test_mixed_state_and_accumulation_stay_bounded(self):
        previous = create_empty_state().to_list()
        positive = AppraisalResult(feeling="proud", emotion_delta={"joy": 0.9, "pride": 0.8})
        negative = AppraisalResult(feeling="hurt", emotion_delta={"sadness": 0.9, "anger": 0.5})
        first = project_affect(previous, positive, elapsed_seconds=0.0)
        second = project_affect(first.state.to_list(), negative, elapsed_seconds=0.0)
        # Two simultaneous emotions can coexist.
        self.assertGreater(second.state.state_emotions.joy, 0.5)
        self.assertGreater(second.state.state_emotions.sadness, 0.5)
        for name in ("joy", "pride", "sadness", "anger"):
            value = float(getattr(second.state.state_emotions, name))
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_contrasting_personas_differ_on_the_same_facts(self):
        previous = create_empty_state().to_list()
        facts = AppraisalResult(
            feeling="let down",
            appraisal={"goal_congruence": -0.7, "certainty": 0.2, "agency_other": 0.8},
            expectation_error=-0.8,
        )
        cheerful = make_personality_profile(big5=(0.7, 0.6, 0.8, 0.9, 0.1), affiliation=0.9, warmth=0.9)
        dramatic = make_personality_profile(big5=(0.5, 0.4, 0.3, 0.3, 0.95), attach_anx=0.9, affiliation=0.8, warmth=0.3)
        calm = project_affect(previous, facts, elapsed_seconds=60.0, personality=cheerful)
        intense = project_affect(previous, facts, elapsed_seconds=60.0, personality=dramatic)
        self.assertGreater(intense.state.state_emotions.sadness, calm.state.state_emotions.sadness)
        self.assertGreater(intense.state.state_emotions.sadness, 0.0)

    def test_personality_gains_stay_bounded(self):
        extreme = make_personality_profile(big5=(1.0, 1.0, 1.0, 1.0, 1.0), bas_drive=1.0, bis=1.0,
                                           attach_anx=1.0, attach_avoid=1.0, affiliation=1.0,
                                           self_control=0.0, warmth=1.0, humor=1.0, blunt=1.0)
        for value in personality_gains(extreme).values():
            self.assertGreaterEqual(value, 0.5)
            self.assertLessEqual(value, 1.5)

    def test_danger_preempts_slow_regulation(self):
        previous = create_empty_state().to_list()
        danger = AppraisalResult(feeling="terrified", appraisal={"bodily_threat": 0.95, "goal_congruence": -0.9})
        sad = AppraisalResult(feeling="low", emotion_delta={"sadness": 0.9}, appraisal={"goal_congruence": -0.5})
        danger_projection = project_affect(previous, danger, elapsed_seconds=0.0)
        sad_projection = project_affect(previous, sad, elapsed_seconds=0.0)
        self.assertEqual(danger_projection.regulation, REGULATION_REACTIVE)
        self.assertEqual(sad_projection.regulation, REGULATION_REFLECTIVE)

    def test_negative_expectation_changes_mood_without_an_observed_fact(self):
        previous = create_empty_state().to_list()
        projected = project_affect(previous, None, elapsed_seconds=0.0, expectation_error=-1.0)
        self.assertLess(projected.state.state_core.valence, 0.0)
        self.assertGreater(projected.state.state_emotions.sadness, 0.0)
        self.assertEqual(projected.trace["expectation_error"], -1.0)


class PipelineTraceTests(unittest.TestCase):
    def test_pipeline_trace_links_event_to_affect_and_regulation(self):
        payloads: dict[str, dict] = {}

        def run_stage(name, payload):
            payloads[name] = dict(payload)
            return dict(_reports(11, appraisal={
                "feeling": "alarmed",
                "appraisal": {"bodily_threat": 0.9, "goal_congruence": -0.8},
            })[name])

        from workflows.main_cognition.pipeline import run_cycle
        request = {
            "companion_name": "Spamuel", "user_name": "Jesse",
            "mental_state_latent_vec": create_empty_state().to_list(),
            "mental_state_age_seconds": 120.0,
            "stimuli": [{"knoxel_id": 11, "kind": "user.input", "text": "the kitchen is on fire"}],
        }
        report = run_cycle(request, run_stage)
        self.assertEqual(report.regulation, REGULATION_REACTIVE)
        self.assertIn("elapsed_seconds", report.affect_trace)
        self.assertIn("behavior", report.affect_trace)
        # Later stages see the accepted affect block.
        self.assertIn("affect", payloads["attention"])
        self.assertEqual(payloads["attention"]["affect"]["regulation"], REGULATION_REACTIVE)
        # Danger forces the reactive control mode even when the model chose reflective.
        self.assertIn("workspace", payloads)
        self.assertEqual(report.control_mode, REGULATION_REACTIVE)
        self.assertIn("preempt", report.frame_reason.lower())

    def test_report_carries_the_projected_state(self):
        def run_stage(name, payload):
            return dict(_reports(11, emotion={"joy": 0.5})[name])

        from workflows.main_cognition.pipeline import run_cycle
        report = run_cycle({"stimuli": [{"knoxel_id": 11, "kind": "user.input"}]}, run_stage)
        mental = FullMentalState.init_from_list(report.mental_state_latent_vec)
        self.assertGreater(mental.state_emotions.joy, 0.0)
        self.assertTrue(report.affect_summary)


class ModuleCommitTests(unittest.TestCase):
    def test_appraisal_feature_carries_trace_and_behavior(self):
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "I let you down.")
        reports = _reports(stimulus_id, appraisal={
            "feeling": "hurt",
            "appraisal": {"goal_congruence": -0.7, "certainty": 0.3},
            "expectation_error": -0.6,
        })
        with TemporaryDirectory() as tmp:
            module, bus, errors, _ = _module(ghost, reports, tmp)
            bus.put(Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                             correlation_id="corr-a07", payload={"knoxel_id": stimulus_id}))
            self.assertTrue(module.step(block=False))
            self.assertEqual(errors, [])
            feature = [
                knoxel for knoxel in ghost.all_features
                if knoxel.feature_subtype == FeatureSubtype.Appraisal
            ][0]
            trace = feature.metadata["affect_trace"]
            self.assertIn("personality_gains", trace)
            self.assertIn("behavior", trace)
            self.assertEqual(feature.metadata["regulation"], trace["regulation"])
            self.assertGreater(feature.mental_state_appraisal.state_emotions.sadness, 0.0)
            # The committed cycle uses the same projected vector the stages saw.
            committed = ghost.states[-1].mental_state_latent_vec
            self.assertEqual(committed, feature.mental_state_appraisal_vec)
            # The later choice keeps the accepted affect beside its own identity.
            action = [
                knoxel for knoxel in ghost.all_knoxels.values()
                if isinstance(knoxel, ActionSchemaRecord)
            ][-1]
            self.assertEqual(action.metadata["regulation"], feature.metadata["regulation"])
            self.assertTrue(action.metadata["affect_summary"])

    def test_explicit_need_delta_applies_at_full_weight(self):
        ghost = _small_ghost()
        stimulus_id = _store(ghost, "I am here with you.")
        reports = _reports(stimulus_id, need={"connection": 0.3})
        with TemporaryDirectory() as tmp:
            module, bus, errors, _ = _module(ghost, reports, tmp)
            bus.put(Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                             correlation_id="corr-need", payload={"knoxel_id": stimulus_id}))
            self.assertTrue(module.step(block=False))
            self.assertEqual(errors, [])
            mental = FullMentalState.init_from_list(ghost.states[-1].mental_state_latent_vec)
            self.assertAlmostEqual(mental.state_needs.connection, 0.3)

    def test_affect_survives_a_checkpoint_restart(self):
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ghost.json"
            ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=checkpoint)
            stimulus_id = _store(ghost, "That failure was not your fault.")
            reports = _reports(stimulus_id, emotion={"sadness": 0.7})
            module, bus, errors, _ = _module(ghost, reports, tmp)
            bus.put(Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                             correlation_id="corr-save", payload={"knoxel_id": stimulus_id}))
            self.assertTrue(module.step(block=False))
            self.assertEqual(errors, [])
            before = FullMentalState.init_from_list(ghost.states[-1].mental_state_latent_vec)

            reopened = MemoryStore.from_checkpoint(checkpoint)
            after = FullMentalState.init_from_list(reopened.states[-1].mental_state_latent_vec)
            self.assertAlmostEqual(after.state_emotions.sadness, before.state_emotions.sadness)
            # A restart does not reset the persona to neutral.
            self.assertGreater(after.state_emotions.sadness, 0.0)

    def test_service_failure_startup_event_is_appraised_not_neutral(self):
        ghost = _small_ghost()
        boot = record_startup(ghost, boot_id="boot-a07")
        reports = _reports(boot["knoxel_id"], appraisal={
            "feeling": "uncertain", "appraisal": {"certainty": 0.1, "novelty": 0.6},
        })
        with TemporaryDirectory() as tmp:
            module, bus, errors, _ = _module(ghost, reports, tmp)
            envelope = Envelope(
                id=new_id(), source="system", kind="data", topic="system.lifecycle",
                correlation_id="", payload={"knoxel_id": boot["knoxel_id"]},
            )
            knoxel_id, text, kind = module._stimulus_from(envelope)
            self.assertEqual(kind, "system.lifecycle")
            self.assertIn("uncertain", text)
            record_shutdown(ghost, boot_id="boot-a07")

            bus.put(envelope)
            self.assertTrue(module.step(block=False))
            self.assertEqual(errors, [])
            feature = [
                knoxel for knoxel in ghost.all_features
                if knoxel.feature_subtype == FeatureSubtype.Appraisal
            ][0]
            self.assertLess(feature.mental_state_appraisal.appraisal_general.certainty, 0.5)


if __name__ == "__main__":
    unittest.main()