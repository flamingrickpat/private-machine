"""Check publication repair against original failed native receipts."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dev.work_items.alpha_a13_minecraft_gate import route_trial
from workflows.minecraft_tasks.schemas import PlanProposal, RevisionProposal, TaskTree
from workflows.minecraft_tasks.steps import compact_plan_context, leaf_project_context, project_check_context, revision_candidate
from workflows.minecraft_tasks.tree import TreeStore, digest

CAPTURE = Path(__file__).resolve().parents[1] / "test_data/captures/alpha/a13-publication-repair.json"


class PublicationContextTests(unittest.TestCase):
    def test_failed_publication_survives_schema_feedback_and_static_acceptance(self):
        capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
        self.assertFalse(capture["project_check_in_original_planner"])
        self.assertTrue(capture["planner_payload"]["latest_verification"]["verified"])
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            original = capture["publication_checks"][-1]
            relative = "verification/project-check-0003.json"
            store.write(relative, original)
            store.state.update(project_check=relative, goal=capture["tree"]["nodes"][0]["instruction"],
                               feedback=capture["planner_payload"]["feedback"])
            before = copy.deepcopy(store.state)
            check = project_check_context(store)
            self.assertEqual(check["errors"], original["result"]["errors"])
            self.assertFalse(check["checks"]["tested_saved_route"])
            self.assertEqual(check["original_digest"], digest(original))
            self.assertEqual(compact_plan_context({"feedback":store.state["feedback"], "latest_project_check":check})["latest_project_check"], check)
            self.assertEqual(leaf_project_context(store, [])["latest_project_check"], check)
            self.assertEqual(store.state, before)

    def test_original_restoration_only_proposal_preserves_the_complete_tree(self):
        capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
        original = RevisionProposal.model_validate(capture["native_proposal"])
        self.assertEqual(original.updates, [])
        previous = TaskTree.model_validate(capture["tree"])
        proposal = PlanProposal(status=original.status, summary=original.summary, restore_nodes=original.restore_nodes)
        candidate = revision_candidate(previous, proposal)
        self.assertEqual(candidate.model_dump(mode="json"), previous.model_dump(mode="json"))
        self.assertIsNot(candidate, previous)
        # A candidate preserves specifications; it does not repair missing
        # world effects or establish acceptance from the model's statement.
        self.assertFalse(capture["publication_checks"][-1]["result"]["checks"]["tested_saved_route"])

    def test_observed_small_trip_and_stationary_repairs_do_not_prove_an_excursion(self):
        capture = json.loads(CAPTURE.read_text(encoding="utf-8"))
        def observation(row):
            result = row["result"]
            return {"position":result["camera"]["feet_position"],
                    "safe":result["player"]["health"] > 0 and result["player"]["on_ground"] and not result["player"]["in_water"],
                    "home":True, "life":(result["body_session"],result["death_count"])}
        for attempt in capture["observations"]:
            observations = [observation(row) for row in attempt["observations"]]
            self.assertFalse(route_trial(observations[0], observations[2], observations[-1]))
        # These translated positions test numeric tolerance only. They are
        # derived operands and do not claim a successful real Minecraft trip.
        start = observation(capture["observations"][0]["observations"][0])
        returned = copy.deepcopy(start)
        for distance, expected in ((7.9,False),(8,True),(10.3,True),(12,True),(12.1,False)):
            excursion = copy.deepcopy(start)
            excursion["position"]["x"] += distance
            self.assertEqual(route_trial(start, excursion, returned), expected)
        excursion["position"]["x"] = start["position"]["x"] + 10
        self.assertTrue(route_trial(start, excursion, returned))
        returned["life"] = (start["life"][0], start["life"][1] + 1)
        self.assertFalse(route_trial(start, excursion, returned))


if __name__ == "__main__":
    unittest.main()
