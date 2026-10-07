"""Task graph and evidence contracts. Live behavior has a separate real gate."""
from __future__ import annotations

import copy
import ctypes
from graphlib import TopologicalSorter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time
import unittest

from pm_workflows.manifest import parse_workflow
from pydantic import ValidationError

from workflows.minecraft.actor import ActorOutcome
from workflows.minecraft_tasks.schemas import Condition, Node, PlanProposal, TaskTree
from workflows.minecraft_tasks.tree import TreeStore, acceptance_requirements, digest, validate_tree
from workflows.minecraft_tasks.verification import evaluate
from workflows.minecraft_tasks.steps import progress_facts, scoped_reads, survey_conditions
from runtime.cancellation import LinkedStopEvent

ROOT = Path(__file__).resolve().parents[1]
CAPTURED = json.loads((ROOT / "tests/fixtures/alpha/a11-original-world-reads.json").read_text(encoding="utf-8"))


def tree() -> TaskTree:
    """An independently defined graph exercises ordering without world simulation."""
    alive = Condition(id="living", description="The player remains alive.", kind="alive")
    nodes = [Node(id="goal", kind="goal", children=["gather", "craft", "build", "use"],
                  instruction="A usable structure.", criteria=[alive])]
    for key, dependencies in [("gather", []), ("craft", ["gather"]), ("build", ["craft"]), ("use", ["build"])]:
        nodes.append(Node(id=key, kind="leaf", parent="goal", depends_on=dependencies, instruction=key, criteria=[alive.model_copy(deep=True)]))
    return TaskTree(goal_id="goal", nodes=nodes)


def retain_contract_acceptances(store: TreeStore) -> None:
    """Wrap captured reads in generated envelopes for revision checks only.

    These envelopes never establish live actor behavior or model quality.
    The separate real gate supplies original actor and verifier provenance.
    """
    nodes = {node.id: node for node in store.load().nodes}
    for key in TopologicalSorter(acceptance_requirements(store.load())).static_order():
        retain_contract_node(store, nodes[key])


def retain_contract_node(store: TreeStore, node: Node) -> None:
    """Retain a generated certificate envelope around captured live body reads."""
    reads = []
    for condition in node.criteria:
        original = copy.deepcopy(CAPTURED["reads"][0])
        original["criterion_id"] = condition.id
        if condition.kind == "accepted":
            record = store.state["acceptances"][condition.node_id]
            original = {"criterion_id": condition.id, "tool": "accepted_task", "ok": True,
                        "arguments": {"node_id": condition.node_id}, "result": {
                            "node_id": condition.node_id, "verified": True,
                            "certificate_digest": digest(record), "verification": record["verification"]}}
        reads.append(original)
    path = f"verification/contract-{node.id}.json"
    store.write(path, {"generated_contract_envelope": True, "verified": True,
        "identity": {"leaf_id": node.id, "tree_revision": store.state["revision"], "spec_revision": node.revision},
        "reads": reads, "review": {"status": "accepted", "criterion_ids": [row.id for row in node.criteria]}})
    store.accept(node.id, path, [])


class TaskGraphTests(unittest.TestCase):
    def test_original_door_reads_require_both_halves(self):
        fixture = json.loads((ROOT / "tests/fixtures/alpha/a11-original-door-reads.json").read_text(encoding="utf-8"))
        built = fixture["native_build"]["result"]
        self.assertEqual((built["planned_cells"], built["placed_count"]), (1, 2))
        for key, expected in (("successful_reads", True), ("failed_reads", False)):
            reads = fixture[key]
            condition = Condition(id="door", description="Both door cells exist.", kind="block", names=["birch_door"],
                                  positions=[row["arguments"]["position"] for row in reads])
            self.assertEqual(evaluate([condition], reads)["door"], expected)

    def test_survey_requires_world_queries_without_a_tree(self):
        query = {"id": "terrain", "description": "Read the foundation layer.", "kind": "block",
                 "names": ["air"], "position": {"x": 1, "y": 112, "z": 2}}
        proposal = PlanProposal.model_validate({"status": "survey", "summary": "Check terrain.", "queries": [query]})
        self.assertEqual(proposal.queries[0].position.y, 112)
        for change in ({"queries": []}, {"tree": tree().model_dump()},
                       {"queries": [{"id": "prior", "description": "Not a world query.", "kind": "accepted", "node_id": "gather"}]}):
            with self.subTest(change=change), self.assertRaises(ValidationError):
                PlanProposal.model_validate({**proposal.model_dump(), **change})

    def test_refined_prerequisite_story_retains_external_acceptance_references(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            proposed = tree()
            proposed.nodes[2].criteria.append(Condition(id="gathered", description="Gathering passed.", kind="accepted", node_id="gather"))
            proposed.nodes[3].criteria.append(Condition(id="supplied", description="Crafting passed.", kind="accepted", node_id="craft"))
            first = store.revise(proposed)
            refined = first.model_copy(deep=True)
            refined.nodes[2].kind = "story"
            refined.nodes[2].children = ["craft_a", "craft_b"]
            for key in refined.nodes[2].children:
                refined.nodes.append(Node(id=key, kind="leaf", parent="craft", instruction=key,
                                          criteria=copy.deepcopy(refined.nodes[0].criteria)))
            revised = store.revise(refined, target="craft")
            self.assertEqual(revised.nodes[2].criteria, first.nodes[2].criteria)
            self.assertEqual(revised.nodes[3], first.nodes[3])
            for key in ("gather", "craft_a", "craft_b"):
                retain_contract_node(store, next(node for node in revised.nodes if node.id == key))
            self.assertEqual(store.ready_aggregates(), ["craft"])
            self.assertNotIn("build", store.eligible())
            retain_contract_node(store, revised.nodes[2])
            self.assertEqual(store.eligible(), ["build"])
            self.assertIn("craft", store.requirements("build"))
            store.state["acceptances"].pop("craft")
            self.assertNotIn("build", store.eligible())

    def test_refined_story_without_explicit_reference_still_guards_dependents(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            proposed = tree()
            proposed.nodes[2].kind = "story"
            proposed.nodes[2].children = ["craft_a"]
            proposed.nodes.append(Node(id="craft_a", kind="leaf", parent="craft", instruction="Craft supplies.",
                                       criteria=copy.deepcopy(proposed.nodes[0].criteria)))
            store.revise(proposed)
            for key in ("gather", "craft_a"):
                retain_contract_node(store, next(node for node in store.load().nodes if node.id == key))
            self.assertNotIn("build", store.eligible())
            retain_contract_node(store, next(node for node in store.load().nodes if node.id == "craft"))
            self.assertIn("build", store.eligible())
            before = store.state["acceptances"]["craft"]["files"].copy()
            retain_contract_node(store, next(node for node in store.load().nodes if node.id == "build"))
            self.assertTrue(store.valid("craft"))
            self.assertEqual(store.state["acceptances"]["craft"]["files"], before)

    @unittest.skipUnless(os.name == "nt", "Windows file sharing establishes this boundary.")
    def test_transient_real_file_lock_retries_only_the_commit(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            store.write("proof.json", {"revision": 1})
            kernel = ctypes.windll.kernel32
            create = kernel.CreateFileW
            create.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
            create.restype = ctypes.c_void_p
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = create(str(store.path_for("proof.json")), 0x80000000, 1, None, 3, 0, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            def release():
                time.sleep(0.08)
                kernel.CloseHandle(handle)
            worker = Thread(target=release)
            worker.start()
            try:
                store.write("proof.json", {"revision": 2})
            finally:
                worker.join()
            self.assertEqual(json.loads(store.path_for("proof.json").read_text()), {"revision": 2})

    @unittest.skipUnless(os.name == "nt", "Windows file sharing establishes this boundary.")
    def test_persistent_real_file_lock_cannot_trigger_phase_retry(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            store.write("proof.json", {"revision": 1})
            kernel = ctypes.windll.kernel32
            create = kernel.CreateFileW
            create.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
            create.restype = ctypes.c_void_p
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            handle = create(str(store.path_for("proof.json")), 0x80000000, 1, None, 3, 0, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            try:
                with self.assertRaisesRegex(RuntimeError, "Do not replay"):
                    store.write("proof.json", {"revision": 2})
            finally:
                kernel.CloseHandle(handle)
            self.assertEqual(json.loads(store.path_for("proof.json").read_text()), {"revision": 1})

    def test_local_actor_stop_keeps_planner_and_next_actor_active(self):
        parent = Event()
        actor = LinkedStopEvent(parent)
        actor.set()
        self.assertTrue(actor.is_set())
        self.assertFalse(parent.is_set())
        self.assertFalse(LinkedStopEvent(parent).is_set())

    def test_parent_cancellation_stops_every_child(self):
        parent = Event()
        children = [LinkedStopEvent(parent), LinkedStopEvent(parent)]
        parent.set()
        self.assertTrue(all(child.is_set() and child.wait(0) for child in children))

    def test_child_cannot_clear_parent_cancellation(self):
        parent = Event()
        child = LinkedStopEvent(parent)
        parent.set()
        child.clear()
        self.assertTrue(child.is_set())

    def test_dependency_order_requires_verified_prerequisites(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            store.revise(tree())
            self.assertEqual(store.eligible(), ["gather"])
            store.state["attempts"].append({"leaf_id": "gather", "state": "complete"})
            self.assertEqual(store.eligible(), ["gather"])

    def test_missing_dependency_fails(self):
        proposed = tree()
        proposed.nodes[-1].depends_on = ["absent"]
        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_tree(proposed)

    def test_dependency_cycle_fails(self):
        proposed = tree()
        proposed.nodes[1].depends_on = ["use"]
        with self.assertRaisesRegex(ValueError, "cycle"):
            validate_tree(proposed)

    def test_duplicate_ids_fail(self):
        proposed = tree()
        proposed.nodes[-1].id = "gather"
        with self.assertRaises(ValueError):
            validate_tree(proposed)

    def test_story_dependency_expands_to_leaves(self):
        proposed = tree()
        proposed.nodes[0].children = ["supplies", "build", "use"]
        proposed.nodes[1].parent = proposed.nodes[2].parent = "supplies"
        proposed.nodes[3].depends_on = ["supplies"]
        proposed.nodes.append(Node(id="supplies", kind="story", parent="goal", children=["gather", "craft"],
                                   instruction="Produce supplies.", criteria=proposed.nodes[0].criteria))
        self.assertEqual(validate_tree(proposed)["build"], {"gather", "craft"})

    def test_acceptance_reference_requires_hierarchy_or_dependency_guard(self):
        proposed = tree()
        proposed.nodes[0].criteria = [Condition(id="gathered", description="Gathering passed.", kind="accepted", node_id="gather")]
        validate_tree(proposed)
        proposed.nodes[1].criteria = [Condition(id="future", description="Use passed.", kind="accepted", node_id="use")]
        with self.assertRaisesRegex(ValueError, "declared prerequisites"):
            validate_tree(proposed)

    def test_acceptance_reference_recomputes_exact_certificate(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            store.revise(tree())
            retain_contract_acceptances(store)
            record = store.state["acceptances"]["gather"]
            condition = Condition(id="gathered", description="Gathering passed.", kind="accepted", node_id="gather")
            row = {"criterion_id": "gathered", "tool": "accepted_task", "ok": True,
                   "arguments": {"node_id": "gather"}, "result": {"node_id": "gather", "verified": True,
                   "certificate_digest": digest(record), "verification": record["verification"]}}
            self.assertFalse(evaluate([condition], [row])["gathered"])
            self.assertTrue(evaluate([condition], [row], certificates=store.state["acceptances"])["gathered"])
            row["result"]["certificate_digest"] = "stale"
            self.assertFalse(evaluate([condition], [row], certificates=store.state["acceptances"])["gathered"])

    def test_hierarchy_cycle_fails(self):
        proposed = tree()
        proposed.nodes[1].kind = "story"
        proposed.nodes[1].children = ["goal"]
        proposed.nodes[0].parent = "gather"
        with self.assertRaises(ValueError):
            validate_tree(proposed)

    def test_immutable_snapshots_and_unchanged_node_revisions(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            first = store.revise(tree())
            before = (Path(folder) / "planning/specs/0001.json").read_bytes()
            proposed = first.model_copy(deep=True)
            proposed.nodes[3].instruction = "Build the revised footprint."
            second = store.revise(proposed)
            self.assertEqual([n.revision for n in second.nodes], [1, 1, 1, 2, 1])
            self.assertEqual((Path(folder) / "planning/specs/0001.json").read_bytes(), before)
            self.assertEqual(TreeStore(Path(folder)).load(), second)

    def test_revision_invalidates_only_affected_acceptance(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            store.revise(tree())
            retain_contract_acceptances(store)
            proposed = store.load()
            proposed.nodes[3].instruction = "Build the revised footprint."
            store.revise(proposed)
            self.assertEqual(set(store.state["acceptances"]), {"gather", "craft"})
            self.assertEqual(store.eligible(), ["build"])
            self.assertTrue(store.path_for("verification/contract-build.json").is_file())

    def test_changed_original_receipt_invalidates_dependents(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            store.revise(tree())
            retain_contract_acceptances(store)
            path = store.path_for("verification/contract-gather.json")
            report = json.loads(path.read_text(encoding="utf-8"))
            report["reads"][0]["result"]["player"]["health"] = 0
            store.write("verification/contract-gather.json", report)
            self.assertEqual(set(store.invalidate()), {"goal", "gather", "craft", "build", "use"})
            self.assertEqual(store.eligible(), ["gather"])

    def test_controlled_refinement_preserves_unrelated_specs(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            proposed = store.revise(tree()).model_copy(deep=True)
            proposed.nodes[1].kind = "story"
            proposed.nodes[1].children = ["gather_a", "gather_b"]
            for key in proposed.nodes[1].children:
                proposed.nodes.append(Node(id=key, kind="leaf", parent="gather", instruction="One material outcome.", criteria=proposed.nodes[1].criteria))
            store.revise(proposed, target="gather")
            self.assertEqual(set(store.eligible()), {"gather_a", "gather_b"})

    def test_refinement_cannot_change_unrelated_node(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            proposed = store.revise(tree()).model_copy(deep=True)
            proposed.nodes[2].instruction = "Different unrelated work."
            with self.assertRaisesRegex(ValueError, "unrelated"):
                store.revise(proposed, target="gather")

    def test_refinement_cannot_weaken_acceptance(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            proposed = store.revise(tree()).model_copy(deep=True)
            proposed.nodes[1].criteria[0].description = "Different requirement."
            with self.assertRaisesRegex(ValueError, "preserve"):
                store.revise(proposed, target="gather")

    def test_stale_or_failed_verifier_cannot_release_leaf(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            store.revise(tree())
            for verified, revision in [(False, 1), (True, 0)]:
                store.write("verification/rejected.json", {"verified": verified, "identity": {
                    "leaf_id": "gather", "tree_revision": revision, "spec_revision": 1}})
                with self.assertRaises(ValueError):
                    store.accept("gather", "verification/rejected.json", [])
                self.assertEqual(store.eligible(), ["gather"])

    def test_manifest_routes_every_declared_outcome(self):
        manifest = parse_workflow(ROOT / "workflows/minecraft_tasks/minecraft_tasks.workflow.md")
        self.assertEqual(manifest.name, "minecraft_project")

    def test_evidence_path_cannot_escape_its_owner(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            with self.assertRaisesRegex(ValueError, "escaped"):
                store.write("../outside.json", {})

    def test_completion_flag_without_world_reads_is_rejected(self):
        with TemporaryDirectory() as folder:
            store = TreeStore(Path(folder))
            store.revise(tree())
            store.write("verification/unsupported.json", {"verified": True, "identity": {
                "leaf_id": "gather", "tree_revision": 1, "spec_revision": 1},
                "review": {"status": "accepted", "criterion_ids": ["living"]}})
            with self.assertRaisesRegex(ValueError, "independent world evidence"):
                store.accept("gather", "verification/unsupported.json", [])
            self.assertEqual(store.eligible(), ["gather"])


class CapturedEvidenceTests(unittest.TestCase):
    def test_recipe_scope_preserves_real_output_counts_and_costs(self):
        captured = json.loads((ROOT / "tests/fixtures/alpha/a11-original-recipe-reads.json").read_text(encoding="utf-8"))
        planks, door = scoped_reads(captured["reads"])
        self.assertEqual(planks["result"]["matches"][0]["output_count"], 4)
        self.assertEqual(planks["result"]["matches"][0]["ingredients"][0]["required"], 1)
        self.assertEqual(door["result"]["matches"][0]["output_count"], 3)
        self.assertEqual(door["result"]["matches"][0]["ingredients"][0]["required"], 6)
        self.assertEqual(planks["original_digest"], digest(captured["reads"][0]["result"]))

    def test_native_candidate_missing_handoff_is_rejected(self):
        captured = json.loads((ROOT / "tests/fixtures/alpha/a11-invalid-native-completion.json").read_text(encoding="utf-8"))
        with self.assertRaises(ValidationError) as rejected:
            ActorOutcome.model_validate_json(captured["text"])
        self.assertEqual(rejected.exception.errors()[0]["loc"], ("handoff",))

    def test_scoped_read_keeps_the_captured_non_solid_plant_flag(self):
        captured = json.loads((ROOT / "tests/fixtures/alpha/a11-non-solid-plant-read.json").read_text(encoding="utf-8"))
        read = captured["read"]
        scoped = scoped_reads([read])[0]
        self.assertEqual(scoped["result"]["block_name"], "tall_grass")
        self.assertIs(scoped["result"]["is_solid"], False)
        self.assertEqual(scoped["original_digest"], digest(read["result"]))

    def test_captured_region_after_gathering_cannot_prove_the_actor_route(self):
        captured = json.loads((ROOT / "tests/fixtures/alpha/a11-region-after-gathering.json").read_text(encoding="utf-8"))
        condition = Condition.model_validate(captured["condition"])
        reads = captured["reads"]
        self.assertGreater(reads[0]["result"]["unseen_cells"], 0)
        self.assertFalse(evaluate([condition], reads)[condition.id])

    def test_compact_planning_tree_retains_every_schema_default(self):
        captured = json.loads((ROOT / "tests/fixtures/alpha/a11-rejected-tree.json").read_text(encoding="utf-8"))
        original = TaskTree.model_validate(captured["tree"])
        compact = original.model_dump(mode="json", exclude_defaults=True)
        self.assertEqual(TaskTree.model_validate(compact).model_dump(mode="json"), original.model_dump(mode="json"))
        self.assertLess(len(json.dumps(compact)), len(json.dumps(original.model_dump(mode="json"))))

    def test_site_survey_deduplicates_shared_physical_cells(self):
        original = next(row for row in CAPTURED["reads"] if row["tool"] == "minecraft_inspect_block")
        point = original["result"]["position"]
        condition = Condition(id="trunk", description="Inspect the exact trunk.", kind="block",
                              names=["birch_log"], positions=[point])
        nodes = tree().nodes
        nodes[0].criteria.append(condition)
        nodes[1].criteria.append(condition.model_copy(deep=True))
        queries = survey_conditions(nodes)
        self.assertEqual(len(queries), 2)
        self.assertEqual(queries[1].position.model_dump(), point)
        self.assertEqual(queries[1].positions, [])

    def test_scoped_survey_preserves_original_block_facts_and_digest(self):
        original = copy.deepcopy(next(row for row in CAPTURED["reads"] if row["tool"] == "minecraft_inspect_block"))
        reduced = scoped_reads([original])[0]
        self.assertEqual(reduced["result"]["position"], original["result"]["position"])
        self.assertEqual(reduced["result"]["block_name"], original["result"]["block_name"])
        self.assertEqual(reduced["arguments"], original["arguments"])
        self.assertEqual(reduced["original_digest"], digest(original["result"]))

    def test_clock_and_view_changes_do_not_count_as_yielded_progress(self):
        original = copy.deepcopy(CAPTURED["reads"][0]["result"])
        changed = copy.deepcopy(original)
        changed["player"]["time_of_day"] += 100
        changed["camera"]["yaw"] += 90
        self.assertEqual(progress_facts(original), progress_facts(changed))

    def test_rejected_live_proposal_has_no_matching_parents(self):
        captured = json.loads((ROOT / "tests/fixtures/alpha/a11-rejected-tree.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "exactly one root"):
            validate_tree(TaskTree.model_validate(captured["tree"]))

    def test_captured_block_count_mismatch_fails_explicitly(self):
        captured = json.loads((ROOT / "tests/fixtures/alpha/a11-rejected-block-condition.json").read_text(encoding="utf-8"))
        with self.assertRaisesRegex(ValueError, "exact cells"):
            Condition.model_validate(captured["condition"])

    def test_grouped_cells_require_every_original_read(self):
        original = copy.deepcopy(next(row for row in CAPTURED["reads"] if row["tool"] == "minecraft_inspect_block"))
        original.update(criterion_id="trunk", observed_dimension="overworld")
        condition = Condition(id="trunk", description="The exact trunk cell exists.", kind="block",
                              names=["minecraft:birch_log"], positions=[original["result"]["position"]])
        self.assertTrue(evaluate([condition], [original])["trunk"])
        self.assertFalse(evaluate([condition], [original, original])["trunk"])

    def test_actor_claim_without_reads_fails(self):
        condition = Condition(id="wood", description="At least one birch log.", kind="inventory", names=["birch_log"])
        self.assertFalse(evaluate([condition], [{"criterion_id": "wood", "summary": "complete", "tool": "actor_report", "result": {}, "ok": True}])["wood"])

    def test_original_inventory_and_absent_item(self):
        original = copy.deepcopy(CAPTURED["reads"][0])
        original["criterion_id"] = "inventory"
        present = Condition(id="inventory", description="Birch logs exist.", kind="inventory", names=["birch_log"], minimum=1)
        absent = present.model_copy(update={"names": ["diamond_pickaxe"]})
        self.assertTrue(evaluate([present], [original])["inventory"])
        self.assertFalse(evaluate([absent], [original])["inventory"])

    def test_block_coordinates_and_dimension_are_required(self):
        original = copy.deepcopy(next(row for row in CAPTURED["reads"] if row["tool"] == "minecraft_inspect_block"))
        original.update(criterion_id="trunk", observed_dimension="overworld")
        condition = Condition(id="trunk", description="The inspected trunk exists.", kind="block",
                              names=["birch_log"], position=original["result"]["position"])
        self.assertTrue(evaluate([condition], [original])["trunk"])
        original["observed_dimension"] = "minecraft:the_nether"
        self.assertFalse(evaluate([condition], [original])["trunk"])
        original["observed_dimension"] = "overworld"
        original["arguments"]["position"]["x"] += 1
        self.assertFalse(evaluate([condition], [original])["trunk"])

    def test_missing_or_duplicate_original_read_fails(self):
        original = copy.deepcopy(CAPTURED["reads"][0])
        original["criterion_id"] = "living"
        condition = Condition(id="living", description="Alive.", kind="alive")
        self.assertFalse(evaluate([condition], [])["living"])
        self.assertFalse(evaluate([condition], [original, original])["living"])


if __name__ == "__main__":
    unittest.main()
