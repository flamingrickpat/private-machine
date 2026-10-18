"""Pure codelet scheduling and attention competition.

No model, kernel, or MemoryStore. These tests fix the rules of the scheduler
and the competition. They do not show that the selected codelets produce
useful proposals; the live A15 gate checks that with real inference.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from workflows.main_cognition.attention_competition import compete
from workflows.main_cognition.codelet_schedule import (
    AttentionPolicy, load_catalogue, request_signals, schedule, state_signals,
)
from workflows.main_cognition.host import stale_nominations
from workflows.main_cognition.schemas import AppraisalResult, AttentionNomination, CodeletHistory, Stimulus
from pm.records.mental_state import create_empty_state


def write_catalogue(directory: str, codelets: list[dict], **policy) -> Path:
    path = Path(directory) / "catalogue.yaml"
    path.write_text(yaml.safe_dump({"version": "t1", "policy": {"main": policy}, "codelets": codelets}),
                    encoding="utf-8")
    return path


def signals_for(stimuli=(), *, idle=None, appraisal=None) -> dict[str, float]:
    """Return the event signals of a cycle with these inputs and no goals or memories."""
    return request_signals([Stimulus.model_validate(item) for item in stimuli], active_goal_count=0,
                           selected_memory_count=0, seconds_since_user_input=idle, regulation="",
                           appraisal=AppraisalResult.model_validate(appraisal or {}))


def llm(name: str, base: float, **extra) -> dict:
    return {"name": name, "family": "appraisal", "kind": "llm", "proposal": "appraisal",
            "reality": "inferred", "base_activation": base, "instructions": "Do one thing.", **extra}


class DefaultCatalogueTests(unittest.TestCase):
    def test_default_catalogue_loads_with_a_stable_digest(self):
        first, second = load_catalogue(), load_catalogue()
        self.assertEqual(first.digest, second.digest)
        self.assertIn("main", first.catalogue.policy)
        self.assertIn("inner", first.catalogue.policy)

    def test_llm_codelet_without_instructions_is_invalid(self):
        with TemporaryDirectory() as directory:
            path = write_catalogue(directory, [llm("x", 0.9, instructions="")])
            with self.assertRaises(ValueError):
                load_catalogue(path)


class SignalTests(unittest.TestCase):
    def test_thought_and_user_stimuli_are_separate_signals(self):
        thought = signals_for([{"knoxel_id": 1, "kind": "thought.created"}])
        user = signals_for([{"knoxel_id": 1, "kind": "user.input", "source": "user_input"}])
        self.assertEqual((thought["stimulus.thought"], thought["stimulus.user"]), (1.0, 0.0))
        self.assertEqual((user["stimulus.thought"], user["stimulus.user"]), (0.0, 1.0))

    def test_idle_seconds_become_hours_and_negative_valence_is_positive(self):
        signals = signals_for(idle=7200)
        self.assertEqual(signals["time.idle_hours"], 2.0)
        state = create_empty_state()
        state.state_core.valence = -0.4
        state.state_emotions.sadness = 0.6
        flat = state_signals(state)
        self.assertEqual(flat["core.negative_valence"], 0.4)
        self.assertEqual(flat["emotion.strongest"], 0.6)
        self.assertEqual(flat["emotion.negative_strongest"], 0.6)

    def test_missing_axes_derive_inverse_signals_for_gates_and_boosts(self):
        signals = signals_for(appraisal={"appraisal": {
            "fairness": 0.2, "inclusion_exclusion": 0.3, "certainty": 0.25,
        }})
        self.assertEqual(signals["appraisal.unfairness"], 0.8)
        self.assertEqual(signals["appraisal.exclusion"], 0.7)
        self.assertEqual(signals["appraisal.uncertainty"], 0.75)
        # The underlying axis keeps its own direction.
        self.assertEqual(signals["appraisal.fairness"], 0.2)

    def test_a_partner_state_supplies_relationship_signals(self):
        state = create_empty_state()
        flat = state_signals(state)
        self.assertNotIn("relationship.conflict_tension", flat)
        from pm.records.mental_state import StateRelationship
        state.state_relationship = StateRelationship(entity_id=7, conflict_tension=0.6)
        flat = state_signals(state)
        self.assertEqual(flat["relationship.conflict_tension"], 0.6)

    def test_unknown_proposal_kinds_origins_and_realities_are_invalid(self):
        with TemporaryDirectory() as directory:
            bad_kind = write_catalogue(directory, [llm("x", 0.9, proposal="vibes")])
            with self.assertRaises(ValueError):
                load_catalogue(bad_kind)
        with TemporaryDirectory() as directory:
            bad_origin = write_catalogue(directory, [llm("x", 0.9, origin="elsewhere")])
            with self.assertRaises(ValueError):
                load_catalogue(bad_origin)
        with TemporaryDirectory() as directory:
            bad_reality = write_catalogue(directory, [llm("x", 0.9, reality="certain")])
            with self.assertRaises(ValueError):
                load_catalogue(bad_reality)

    def test_the_shipped_catalogue_declares_context_and_consequence(self):
        for codelet in load_catalogue().catalogue.codelets:
            self.assertTrue(codelet.context, codelet.name)
            self.assertTrue(codelet.consequence, codelet.name)


class PathwayBoostTests(unittest.TestCase):
    def test_a_pathway_signal_lets_a_followup_codelet_run(self):
        from workflows.main_cognition.pathways import PathwayDefinition, pathway_map
        followup = llm("followup", 0.1, pathway="curiosity",
                       when=[{"signal": "stimulus.user", "op": ">=", "value": 1}],
                       boost=[{"signal": "pathway.curiosity", "weight": 0.8}])
        pathways = [PathwayDefinition(name="curiosity", version="1", members=["followup"])]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "catalogue.yaml"
            path.write_text(yaml.safe_dump({
                "version": "t1", "policy": {"main": {}}, "codelets": [followup],
                "pathways": [item.model_dump() for item in pathways],
            }), encoding="utf-8")
            loaded = load_catalogue(path)
        decisions, _ = schedule(loaded, {"stimulus.user": 1.0, "pathway.curiosity": 0.9}, {})
        self.assertTrue(decisions[0].run)
        self.assertIn("pathway.curiosity=0.9", decisions[0].cause)


class ScheduleTests(unittest.TestCase):
    def test_gate_cooldown_threshold_and_budget_each_have_a_reason(self):
        codelets = [
            llm("gated", 0.9, when=[{"signal": "stimulus.thought", "op": ">=", "value": 1}]),
            llm("cooling", 0.9, cooldown_ticks=2),
            llm("weak", 0.1),
            llm("strong", 0.6, boost=[{"signal": "emotion.sadness", "weight": 0.5}]),
            llm("second", 0.8),
            llm("third", 0.7),
        ]
        with TemporaryDirectory() as directory:
            loaded = load_catalogue(write_catalogue(directory, codelets, max_llm_codelets=2))
        decisions, _ = schedule(loaded, {"emotion.sadness": 0.8}, {"cooling": CodeletHistory(cycles_since=1)})
        reasons = {item.definition.name: (item.run, item.reason) for item in decisions}
        self.assertEqual(reasons["gated"], (False, "gate"))
        self.assertEqual(reasons["cooling"], (False, "cooldown"))
        self.assertEqual(reasons["weak"], (False, "threshold"))
        self.assertEqual(reasons["strong"], (True, "scheduled"))
        self.assertEqual(reasons["second"], (True, "scheduled"))
        self.assertEqual(reasons["third"], (False, "budget"))

    def test_cooldown_ends_after_the_declared_cycles(self):
        with TemporaryDirectory() as directory:
            loaded = load_catalogue(write_catalogue(directory, [llm("cooling", 0.9, cooldown_ticks=2)]))
        decisions, _ = schedule(loaded, {}, {"cooling": CodeletHistory(cycles_since=2)})
        self.assertTrue(decisions[0].run)

    def test_danger_runs_pure_codelets_only(self):
        pure = {"name": "direct_stimulus", "family": "attention", "kind": "pure", "proposal": "attention",
                "reality": "observed", "base_activation": 1.0}
        with TemporaryDirectory() as directory:
            loaded = load_catalogue(write_catalogue(directory, [pure, llm("slow", 0.9)]))
        decisions, _ = schedule(loaded, {"appraisal.bodily_threat": 0.8}, {})
        reasons = {item.definition.name: (item.run, item.reason) for item in decisions}
        self.assertEqual(reasons["direct_stimulus"], (True, "scheduled"))
        self.assertEqual(reasons["slow"], (False, "danger_budget"))

    def test_emotional_pain_with_reactive_regulation_is_not_danger(self):
        with TemporaryDirectory() as directory:
            loaded = load_catalogue(write_catalogue(directory, [llm("echo", 0.9)]))
        signals = {"regulation.reactive": 1.0, "appraisal.aversive_pressure": 0.8, "emotion.sadness": 0.8,
                   "emotion.fear": 0.9}
        decisions, _ = schedule(loaded, signals, {})
        self.assertTrue(decisions[0].run)


def nomination(key: str, codelet: str, urgency: float, relevance: float, targets=(1,)) -> AttentionNomination:
    return AttentionNomination(nomination_id=key, codelet=codelet, target_ids=list(targets), urgency=urgency,
                               novelty=0.5, goal_relevance=relevance, reportability=0.5)


class CompetitionTests(unittest.TestCase):
    def test_reserved_capacity_and_minimum_score(self):
        policy = AttentionPolicy(capacity=2, min_score=0.3, reserved_codelets={"direct_stimulus": 1})
        items = [
            nomination("n1", "direct_stimulus", 0.1, 0.1, targets=(1,)),
            nomination("n2", "relationship_check", 0.9, 0.9, targets=(2,)),
            nomination("n3", "curiosity_probe", 0.8, 0.8, targets=(3,)),
            nomination("n4", "curiosity_probe", 0.0, 0.0, targets=(4,)),
        ]
        result = compete(items, policy)
        status = {item.nomination_id: item.status for item in result.nominations}
        self.assertEqual(status, {"n1": "selected", "n2": "selected", "n3": "deferred", "n4": "deferred"})
        reasons = {item.nomination_id: item.reason for item in result.nominations}
        self.assertTrue(reasons["n3"].startswith("capacity"))
        self.assertTrue(reasons["n4"].startswith("below_min_score"))
        self.assertEqual(result.selected_ids, [1, 2])

    def test_danger_bonus_lets_urgency_outrank_relevance(self):
        policy = AttentionPolicy(capacity=1, min_score=0.0)
        calm = compete([nomination("a", "x", 0.7, 0.0), nomination("b", "y", 0.0, 1.0)], policy)
        danger = compete([nomination("a", "x", 0.7, 0.0), nomination("b", "y", 0.0, 1.0)], policy, danger=True)
        self.assertEqual([item.nomination_id for item in calm.nominations if item.status == "selected"], ["b"])
        self.assertEqual([item.nomination_id for item in danger.nominations if item.status == "selected"], ["a"])

    def test_unscored_nomination_cannot_win(self):
        item = nomination("n1", "x", 1.0, 1.0)
        item.status, item.reason = "rejected", "unscored"
        result = compete([item], AttentionPolicy(min_score=0.0))
        self.assertEqual(result.nominations[0].status, "rejected")
        self.assertEqual(result.selected_ids, [])


class StaleTests(unittest.TestCase):
    def test_changed_or_removed_source_makes_a_nomination_stale(self):
        same = AttentionNomination(nomination_id="n1", source_revisions={"5": 2})
        changed = AttentionNomination(nomination_id="n2", source_revisions={"5": 1})
        removed = AttentionNomination(nomination_id="n3", source_revisions={"9": 1})
        self.assertEqual(stale_nominations([same, changed, removed], {5: 2}), {"n2", "n3"})


if __name__ == "__main__":
    unittest.main()
