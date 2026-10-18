"""Pathway mechanics: precursor evidence, decay, and catalogue validation.

No model, kernel, or MemoryStore. These tests fix the rules that turn
accepted precursor proposals into later activation. Live pathway behavior is
checked by the A16 gate with real inference.
"""
from __future__ import annotations

import unittest

from workflows.main_cognition.codelet_schedule import load_catalogue
from workflows.main_cognition.pathways import (
    PathwayDefinition, pathway_map, pathway_signals,
)
from workflows.main_cognition.schemas import PathwayPrecursor


def pathway(name: str, members: list[str], **extra) -> PathwayDefinition:
    return PathwayDefinition(name=name, version="1", members=members, **extra)


def precursor(codelet: str, tick: int, **extra) -> PathwayPrecursor:
    return PathwayPrecursor.model_validate({"codelet": codelet, "tick_id": tick, "selected": True,
                                            "record_id": tick, "pathways": [], **extra})


class PathwaySignalTests(unittest.TestCase):
    def setUp(self):
        self.pathways = pathway_map([
            pathway("curiosity", ["curiosity_probe", "question_move"],
                    window_ticks=4, decay_per_tick=0.5),
            pathway("anger", ["refusal_edge"], window_ticks=2, decay_per_tick=0.5),
        ])

    def test_a_fresh_precursor_activates_its_pathway(self):
        signals = pathway_signals([precursor("curiosity_probe", 9)],
                                  self.pathways, latest_tick=10)
        self.assertAlmostEqual(signals["pathway.curiosity"], 0.5)
        self.assertNotIn("pathway.anger", signals)

    def test_age_decays_and_old_precursors_leave_the_window(self):
        fresh = pathway_signals([precursor("curiosity_probe", 9)], self.pathways, latest_tick=10)
        older = pathway_signals([precursor("curiosity_probe", 8)], self.pathways, latest_tick=10)
        gone = pathway_signals([precursor("curiosity_probe", 4)], self.pathways, latest_tick=10)
        self.assertAlmostEqual(fresh["pathway.curiosity"], 0.5)
        self.assertAlmostEqual(older["pathway.curiosity"], 0.25)
        self.assertNotIn("pathway.curiosity", gone)

    def test_several_precursors_add_up_but_cap_at_one(self):
        entries = [precursor("curiosity_probe", 10), precursor("question_move", 9),
                   precursor("curiosity_probe", 9)]
        signals = pathway_signals(entries, self.pathways, latest_tick=10)
        # 1.0 + 0.5 + 0.5 would be 2.0; the cap keeps activation in 0..1.
        self.assertAlmostEqual(signals["pathway.curiosity"], 1.0)

    def test_unselected_and_foreign_records_never_count(self):
        lost = precursor("curiosity_probe", 10, selected=False)
        signals = pathway_signals([lost, precursor("refusal_edge", 10, selected=False)],
                                  self.pathways, latest_tick=10)
        self.assertEqual(signals, {})

    def test_the_window_is_measured_from_the_latest_tick_not_the_precursor(self):
        # A single precursor from two cycles ago keeps its true age of two.
        signals = pathway_signals([precursor("question_move", 8)], self.pathways, latest_tick=10)
        self.assertAlmostEqual(signals["pathway.curiosity"], 0.25)

    def test_duplicate_pathway_names_stop_the_load(self):
        with self.assertRaises(ValueError):
            pathway_map([pathway("play", []), pathway("play", [])])


class CataloguePathwayTests(unittest.TestCase):
    def test_the_shipped_catalogue_declares_every_codelet_pathway(self):
        loaded = load_catalogue()
        names = {item.name for item in loaded.catalogue.codelets}
        for pathway in loaded.catalogue.pathways:
            self.assertTrue(pathway.members, pathway.name)
            self.assertTrue(set(pathway.members) <= names, pathway.name)
            self.assertTrue(pathway.emotions or pathway.description, pathway.name)
        for codelet in loaded.catalogue.codelets:
            if codelet.pathway:
                self.assertIn(codelet.pathway, {p.name for p in loaded.catalogue.pathways})

    def test_every_pathway_member_boosts_on_its_own_signal(self):
        """A pathway is wired when at least one member or follow-up reads it.

        Members connect to follow-up codelets through the ``pathway.<name>``
        boost. A pathway with no reader anywhere in the catalogue could never
        change behavior, so the catalogue must not declare one.
        """
        loaded = load_catalogue()
        readers = {boost.signal for item in loaded.catalogue.codelets for boost in item.boost}
        for pathway in loaded.catalogue.pathways:
            self.assertIn("pathway." + pathway.name, readers, pathway.name)

    def test_a_codelet_with_an_unknown_pathway_is_invalid(self):
        from workflows.main_cognition.codelet_schedule import Catalogue, AttentionPolicy
        from pydantic import ValidationError
        catalogue = {
            "version": "bad",
            "policy": {"main": AttentionPolicy().model_dump()},
            "pathways": [{"name": "play", "version": "1", "members": ["play_urge"]}],
            "codelets": [{
                "name": "play_urge", "family": "needs", "kind": "llm",
                "proposal": "action_tendency", "reality": "inferred",
                "pathway": "missing", "instructions": "Play.",
            }],
        }
        with self.assertRaises(ValidationError):
            Catalogue.model_validate(catalogue)

    def test_a_pathway_member_without_a_codelet_is_invalid(self):
        from workflows.main_cognition.codelet_schedule import Catalogue, AttentionPolicy
        from pydantic import ValidationError
        catalogue = {
            "version": "bad",
            "policy": {"main": AttentionPolicy().model_dump()},
            "pathways": [{"name": "play", "version": "1", "members": ["ghost_codelet"]}],
            "codelets": [],
        }
        with self.assertRaises(ValidationError):
            Catalogue.model_validate(catalogue)


if __name__ == "__main__":
    unittest.main()
