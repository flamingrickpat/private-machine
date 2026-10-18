"""Codelet mechanics of A19F step 5. No model calls except the scripted ones.

Covers the activation score and cooldown, base-level activation, pathway
boosts across unregistered codelets, the output variant count, the
mini-appraisal of percepts, and the CSM boost from two endorsers.
"""
from __future__ import annotations

import random
import unittest
from unittest import mock

from pm.constants import (
    CODELET_BASE_ACTIVATION,
    CODELET_BASE_LEARN_RATE,
    CODELET_COOLDOWN_FACTOR,
    CODELET_W_BASE,
    CSM_BOOST,
    PATHWAY_BOOST_SCALE,
    PATHWAY_SKIP_DECAY,
)
from pm.records.codelet_records import CodeletRegistryState, CodeletRuntimeState
from pm.records.mental_state import AppraisalGeneral, AppraisalSocial, FullMentalState, ema_axis_positions
from pm.records.situational_model import SituationalItem, SituationalModel

from tests.contract.main_cognition.test_cycle_contract import answers, run_cycle
from tests.unit.workflows.main_cognition.cycle_support import ScriptedModel, default_percepts, phase_of
from workflows.main_cognition.affect import percept_affect
from workflows.main_cognition.codelet_pathways import CodeletActivation
from workflows.main_cognition.codelet_percepts import CodeletFamily, MODEL_BY_TYPE, CodeletPerceptType
from workflows.main_cognition.codelet_registry import (
    CodeletDefinition, activation_score, apply_pathway_boost, decay_registry, load_definitions, mark_fired,
    pathway_boosts, reinforce, wave_sizes,
)
from workflows.main_cognition.phases.attention import FocusEvidence

DEFINITION = CodeletDefinition(name="BoredomThresholdCross", families=[CodeletFamily.Appraisals],
                               stimulus_kinds=["user.input"])


def score(runtime: CodeletRuntimeState, tick: int = 10):
    """Score the test definition without noise."""
    rng = mock.Mock(random=lambda: 0.0)
    return activation_score(DEFINITION, runtime, tick=tick, stimulus_kinds=["user.input"],
                            state=FullMentalState(), similarity=0.5, rng=rng)


class ActivationTests(unittest.TestCase):
    def test_cooldown_lowers_the_score_after_a_run(self):
        fresh = score(CodeletRuntimeState(name=DEFINITION.name))
        runtime = CodeletRuntimeState(name=DEFINITION.name)
        mark_fired(runtime, 9, [-2])
        cooling = score(runtime, tick=10)
        self.assertTrue(cooling.cooling)
        self.assertLess(cooling.score, fresh.score * CODELET_COOLDOWN_FACTOR + 1e-9)
        # After the cooldown the factor no longer applies. The recency penalty is small.
        later = score(runtime, tick=9 + runtime.cooldown_ticks)
        self.assertFalse(later.cooling)
        self.assertGreater(later.score, cooling.score)
        self.assertEqual(runtime.total_runs, 1)
        self.assertEqual(runtime.activation, 0.0)

    def test_stored_activation_and_stimulus_match_raise_the_score(self):
        low = score(CodeletRuntimeState(name=DEFINITION.name))
        high = score(CodeletRuntimeState(name=DEFINITION.name, activation=0.2))
        self.assertAlmostEqual(high.score - low.score, 0.2, places=3)
        unmatched = activation_score(DEFINITION, CodeletRuntimeState(), tick=10, stimulus_kinds=["thought.created"],
                                     state=FullMentalState(), similarity=0.5, rng=mock.Mock(random=lambda: 0.0))
        self.assertEqual(unmatched.stimulus, 0.0)
        self.assertLess(unmatched.score, low.score)

    def test_state_match_reads_the_families(self):
        state = FullMentalState()
        state.state_emotions.boredom = 0.9
        bored = activation_score(DEFINITION, CodeletRuntimeState(), tick=10, stimulus_kinds=["user.input"],
                                 state=state, similarity=0.5, rng=mock.Mock(random=lambda: 0.0))
        self.assertAlmostEqual(bored.state, 0.9)

    def test_base_level_activation_rises_on_entry_and_decays_slowly(self):
        registry = CodeletRegistryState()
        runtime = registry.runtime(DEFINITION.name)
        self.assertEqual(runtime.base_activation, CODELET_BASE_ACTIVATION)
        before = score(runtime).score
        reinforce(runtime)
        expected = CODELET_BASE_ACTIVATION + CODELET_BASE_LEARN_RATE * (1 - CODELET_BASE_ACTIVATION)
        self.assertAlmostEqual(runtime.base_activation, expected)
        self.assertAlmostEqual(score(runtime).score - before, CODELET_W_BASE * (expected - CODELET_BASE_ACTIVATION),
                               places=3)
        runtime.activation = 0.4
        decay_registry(registry)
        self.assertLess(runtime.base_activation, expected)
        self.assertGreater(runtime.base_activation, expected * 0.99)
        self.assertLess(runtime.activation, 0.4)

    def test_waves_split_the_budget(self):
        self.assertEqual(wave_sizes(4, 2), [2, 2])
        self.assertEqual(wave_sizes(5, 2), [3, 2])
        self.assertEqual(wave_sizes(1, 2), [1])
        self.assertEqual(wave_sizes(12, 2), [6, 6])


class PathwayTests(unittest.TestCase):
    CHAINS = {"test": [["A", CodeletActivation(1.6), "X", CodeletActivation(1.9), "Y", CodeletActivation(1.5), "B"],
                       ["B", CodeletActivation(1.2), "A"]]}

    def test_boost_skips_unregistered_codelets_with_decay(self):
        boosts = pathway_boosts("A", {"A", "B"}, self.CHAINS)
        self.assertEqual(len(boosts), 1)
        self.assertEqual(boosts[0].target, "B")
        self.assertEqual(boosts[0].skipped, ["X", "Y"])
        self.assertAlmostEqual(boosts[0].strength, 1.6 * PATHWAY_SKIP_DECAY ** 2)

    def test_direct_successor_gets_the_full_step_strength_and_precursors(self):
        boosts = pathway_boosts("B", {"A", "B"}, self.CHAINS)
        self.assertEqual([(item.target, item.strength, item.skipped) for item in boosts], [("A", 1.2, [])])
        registry = CodeletRegistryState()
        boosts[0].precursor_ids = [-4, -5]
        apply_pathway_boost(registry, boosts[0])
        self.assertAlmostEqual(registry.runtime("A").activation, PATHWAY_BOOST_SCALE * 1.2)
        self.assertEqual(registry.runtime("A").precursor_ids, [-4, -5])

    def test_no_registered_successor_gives_no_boost(self):
        self.assertEqual(pathway_boosts("A", {"A"}, self.CHAINS), [])

    def test_shipped_pathways_link_the_frustration_head(self):
        names = set(load_definitions().by_name())
        boosts = pathway_boosts("BlockedGoalFrustration", names)
        self.assertIn("DisrespectAttribution", [item.target for item in boosts])


class DefinitionTests(unittest.TestCase):
    def test_twelve_definitions_load_with_their_modes(self):
        loaded = load_definitions()
        names = loaded.by_name()
        self.assertEqual(len(names), 12)
        agents = sorted(name for name, item in names.items() if item.mode() == "agent")
        self.assertEqual(agents, ["AbandonmentFearAppraisal", "AnalogousEpisodeRecall", "IdentitySpecialNarrative",
                                  "ToMVignetteSimulator"])
        text = " ".join(item.prompt + item.description + " ".join(item.data_aquisation) for item in names.values())
        for name in ("Spamuel", "Jesse", "{user_name}"):
            self.assertNotIn(name, text)
        for softener in ("constructive redirection", "avoiding overt aggression", "neutral observation instead",
                         "Avoid any potentially offensive"):
            self.assertNotIn(softener, text)

    def test_an_omitted_mode_follows_the_first_family(self):
        memory = CodeletDefinition(name="Recall", families=[CodeletFamily.MemoryAccessors, CodeletFamily.Appraisals])
        appraisal = CodeletDefinition(name="Spike", families=[CodeletFamily.Appraisals, CodeletFamily.MemoryAccessors])
        self.assertEqual((memory.mode(), appraisal.mode()), ("agent", "chain"))

    def test_every_percept_type_has_its_own_model(self):
        self.assertEqual(set(MODEL_BY_TYPE), set(CodeletPerceptType))
        self.assertEqual(MODEL_BY_TYPE[CodeletPerceptType.FeelingSocial].__name__, "FeelingSocialPercept")


class MiniAppraisalTests(unittest.TestCase):
    def test_percept_stores_both_vectors_and_keeps_the_state(self):
        state = FullMentalState()
        before = state.model_dump()
        general = AppraisalGeneral(goal_congruence=-0.8, agency_other=0.9, certainty=0.9)
        social = AppraisalSocial(fairness=0.0)
        appraisal_state, delta = percept_affect(state, general, social, entity_id=None)
        self.assertEqual(state.model_dump(), before)
        self.assertEqual(appraisal_state.appraisal_general.goal_congruence, -0.8)
        self.assertGreater(appraisal_state.state_emotions.anger, 0.0)
        axes = ema_axis_positions()
        self.assertTrue(any(abs(value) > 0 for value in delta))
        self.assertTrue(all(value == 0.0 for position, value in enumerate(delta) if position not in axes))


class CsmEndorsementTests(unittest.TestCase):
    def test_two_codelets_endorsing_one_record_raise_its_activation(self):
        model = SituationalModel()
        embeddings = {7: [1.0, 0.0]}
        model.add_or_boost(SituationalItem(record_id=7, activation=0.5), "retrieval", embeddings.__getitem__)
        model.add_or_boost(SituationalItem(record_id=7), "BoredomThresholdCross", embeddings.__getitem__)
        model.add_or_boost(SituationalItem(record_id=7), "HumorousRemarkPlanner", embeddings.__getitem__)
        self.assertAlmostEqual(model.items[7].activation, 0.5 + 2 * CSM_BOOST)
        self.assertEqual(model.items[7].endorsers, ["retrieval", "BoredomThresholdCross", "HumorousRemarkPlanner"])

    def test_cited_records_get_codelet_endorsements_in_a_cycle(self):
        script = answers()
        script["codelet_percepts"] = lambda messages: {**default_percepts(messages), "about_ids": [11]}
        work = run_cycle(ScriptedModel(script), "codelets-endorse")
        stimulus = work.state.csm.items[11]
        self.assertGreaterEqual(len(stimulus.endorsers), 3)
        self.assertEqual(stimulus.endorsers[0], "retrieval")
        self.assertEqual(stimulus.activation, 1.0)


class CycleTests(unittest.TestCase):
    def test_the_cycle_runs_its_budget_in_two_waves_with_one_variant_each(self):
        script = answers()
        script["triage"] = {"items": [{"knoxel_id": 11, "category": "Critical"}], "codelet_budget": 3}
        model = ScriptedModel(script)
        work = run_cycle(model, "codelets-waves")
        result = work.codelets
        self.assertEqual(result.budget, 3)
        self.assertEqual([run.wave for run in result.runs], [1, 1, 2])
        self.assertEqual(len({run.codelet for run in result.runs}), 3)
        self.assertTrue(all(len(output.variants) == 1 for output in result.outputs))
        # One selector call, then call 1 (or the agent) and call 2 per run.
        self.assertEqual(model.calls.count("codelet_select"), 1)
        self.assertEqual(model.calls.count("codelet_percepts"), 3)
        for output in result.outputs:
            percept = output.variants[0].percepts[0]
            self.assertLess(percept.record_id, -1)
            self.assertEqual(len(percept.appraisal_vec), len(percept.delta_vec))
            self.assertIn(percept.record_id, work.records)
        registry = work.state.codelets
        self.assertEqual(sorted(name for name, item in registry.codelets.items() if item.total_runs),
                         sorted(run.codelet for run in result.runs))

    def test_the_variant_count_follows_the_constant(self):
        with mock.patch("workflows.main_cognition.phases.codelets.CODELET_OUTPUT_VARIANTS", 2):
            model = ScriptedModel(answers())
            work = run_cycle(model, "codelets-variants")
        self.assertTrue(all(len(output.variants) == 2 for output in work.codelets.outputs))
        self.assertEqual(model.calls.count("codelet_percepts"), 2 * len(work.codelets.runs))

    def test_every_codelet_call_shares_the_prefix_and_ends_with_its_instruction(self):
        model = ScriptedModel(answers())
        run_cycle(model, "codelets-prefix")
        codelet_calls = [turns for turns in model.messages if turns and phase_of(turns).startswith("codelet_")]
        self.assertEqual(len(codelet_calls), 1 + model.calls.count("codelet_story") + model.calls.count("codelet_percepts"))
        prefixes = {tuple(turn.content.text for turn in turns[:3]) for turns in codelet_calls}
        self.assertEqual(len(prefixes), 1)
        system, story, ack = next(iter(prefixes))
        self.assertIn("expert story writer", system)
        self.assertIn("BEGIN CHARACTER", story)
        self.assertIn("BEGIN WHAT IS ON", story)
        self.assertIn("the kitchen is on fire", story)

    def test_focus_evidence_holds_codelet_notes_only(self):
        self.assertIn("codelet_notes", FocusEvidence.model_fields)


class CommitTests(unittest.TestCase):
    def test_commit_stages_percepts_and_persists_the_registry_and_the_csm(self):
        from tempfile import TemporaryDirectory

        from pm.bus.envelopes import Envelope, new_id
        from pm.records.codelet_records import CodeletPercept
        from tests.unit.workflows.main_cognition.test_cognition_state import _module, _reports, _small_ghost, _store

        ghost = _small_ghost()
        stimulus_id = _store(ghost, "you dodged it again")
        with TemporaryDirectory() as tmp:
            module, bus, errors = _module(ghost, _reports(stimulus_id), tmp)
            bus.put(Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                             correlation_id="corr-codelets", payload={"knoxel_id": stimulus_id}))
            self.assertTrue(module.step(block=False))
        self.assertEqual(errors, [])
        state = ghost.states[-1]
        percepts = [item for item in ghost.all_knoxels.values() if isinstance(item, CodeletPercept)]
        self.assertTrue(percepts)
        self.assertTrue(all(any(item.mental_state_delta_vec) for item in percepts
                            if item.metadata["percept_type"] != "Recommendation"))
        # Pending ids are gone: every CSM item and every last percept id names a record.
        ids = set(ghost.all_knoxels)
        self.assertTrue(set(state.csm.items) <= ids)
        self.assertTrue(all(item.record_id > 0 for item in state.csm.items.values()))
        fired = {name: item for name, item in state.codelets.codelets.items() if item.total_runs}
        self.assertEqual(len(fired), 4)
        self.assertTrue(all(item.last_fired_tick == state.tick_id for item in fired.values()))
        self.assertTrue(all(set(item.last_percept_ids) <= ids for item in fired.values()))
        self.assertIn(stimulus_id, state.csm.items)
        frame = ghost.all_attention_frames[-1]
        self.assertEqual(sorted(frame.metadata["percept_ids"]), sorted(item.id for item in percepts))
        self.assertTrue(all(key > 0 for boost in frame.metadata["pathway_boosts"] for key in boost["precursor_ids"]))


if __name__ == "__main__":
    unittest.main()
