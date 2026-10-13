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
from workflows.main_cognition.schemas import AttentionNomination
from pm.records.mental_state import create_empty_state


def write_catalogue(directory: str, codelets: list[dict], **policy) -> Path:
    path = Path(directory) / "catalogue.yaml"
    path.write_text(yaml.safe_dump({"version": "t1", "policy": {"main": policy}, "codelets": codelets}),
                    encoding="utf-8")
    return path


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
        thought = request_signals({"stimuli": [{"knoxel_id": 1, "kind": "thought.created"}]})
        user = request_signals({"stimuli": [{"knoxel_id": 1, "kind": "user.input", "source": "user_input"}]})
        self.assertEqual((thought["stimulus.thought"], thought["stimulus.user"]), (1.0, 0.0))
        self.assertEqual((user["stimulus.thought"], user["stimulus.user"]), (0.0, 1.0))

    def test_idle_seconds_become_hours_and_negative_valence_is_positive(self):
        signals = request_signals({"seconds_since_user_input": 7200})
        self.assertEqual(signals["time.idle_hours"], 2.0)
        state = create_empty_state()
        state.state_core.valence = -0.4
        state.state_emotions.sadness = 0.6
        flat = state_signals(state)
        self.assertEqual(flat["core.negative_valence"], 0.4)
        self.assertEqual(flat["emotion.strongest"], 0.6)


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
        decisions, _ = schedule(loaded, {"emotion.sadness": 0.8}, {"cooling": {"cycles_since": 1}})
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
        decisions, _ = schedule(loaded, {}, {"cooling": {"cycles_since": 2}})
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
