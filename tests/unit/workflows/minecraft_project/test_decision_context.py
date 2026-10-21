"""Check captured context size and exact material partitions."""
import copy
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory

from pm.llm.object_rendering import render_object_for_llm
from workflows.minecraft_project.planning.decisions import decision_context, native_decision_context
from workflows.minecraft_project.schemas import Condition, Node, TaskTree
from workflows.minecraft_project.tree_store import TreeStore, validate_tree
from workflows.minecraft_project.acceptance import evaluate


ROOT = Path(__file__).resolve().parents[4]
SCHEMA = ROOT / "workflows/minecraft_project/resources/llm_context.json"


class DecisionContextTests(unittest.TestCase):
    def test_initial_context_has_no_actor_attempt(self):
        source = {"latest_attempt": None, "previous_tree": None, "identity": {"tree_revision": 0}}
        self.assertEqual(decision_context(source), source)

    def test_captured_repair_uses_projection_even_without_overflow(self):
        capture = json.loads((ROOT / "test_data/captures/alpha/a13-review-context.json").read_text(encoding="utf-8"))
        source = json.loads(capture["repair_budget"]["original_input"]["messages"][1]["content"])
        before = copy.deepcopy(source)
        selected = decision_context(source)
        rendered = render_object_for_llm(selected, SCHEMA)
        self.assertNotIn("persona_context", selected)
        self.assertNotIn('"original_digest"', rendered)
        self.assertEqual(selected["previous_tree"], source["previous_tree"])
        self.assertLess(len(rendered), len(json.dumps(source, separators=(",", ":"))) * 0.85)
        native = native_decision_context(source)
        self.assertNotIn("nodes", native["previous_tree"])
        self.assertEqual(native["project_initial_observation"]["inventory"], source["project_initial_observation"]["inventory"])
        self.assertEqual(native["project_initial_observation"]["camera"]["feet_position"], source["project_initial_observation"]["camera"]["feet_position"])
        self.assertEqual(native["execution_context"]["recovery"], source["gameplay_history"]["recovery"])
        self.assertLess(len(render_object_for_llm(native, SCHEMA)), len(json.dumps(source, separators=(",", ":"))) * 0.25)
        self.assertEqual(source, before)

    def test_marker_exception_cannot_permit_stone_in_every_wall_cell(self):
        cells = [{"x": x, "y": y, "z": z} for y in range(105, 108)
                 for x in range(28, 33) for z in range(2, 9) if x in (28, 32) or z in (2, 8)]
        self.assertEqual(len(cells), 60)
        mixed = Condition(id="walls", description="Wooden walls with one stone marker.", kind="block",
                          names=["birch_planks", "cobblestone"], positions=cells)
        leaf = Node(id="walls", kind="leaf", parent="goal", instruction="Build the walls.", criteria=[mixed])
        root = Node(id="goal", kind="goal", children=["walls"], instruction="Build the house.",
                    criteria=[Condition(id="wall-proof", description="Walls exist.", kind="accepted", node_id="walls")])
        tree = TaskTree(goal_id="goal", nodes=[root, leaf])
        baseline = tree.model_copy(deep=True)
        baseline.nodes[1].criteria[0].names = ["birch_planks"]
        with TemporaryDirectory() as directory:
            store = TreeStore(Path(directory))
            store.revise(baseline)
            with self.assertRaisesRegex(ValueError, "multiple materials"):
                store.check_revision(tree)
        marker = {"x": 32, "y": 105, "z": 5}
        leaf.criteria = [mixed.model_copy(update={"names": ["birch_planks"],
                                                  "positions": [point for point in mixed.positions if point.model_dump() != marker]}),
                         Condition(id="marker", description="The marker stays stone.", kind="block",
                                   names=["cobblestone"], positions=[marker])]
        validate_tree(tree)
        with TemporaryDirectory() as directory:
            store = TreeStore(Path(directory))
            store.revise(baseline)
            store.check_revision(tree)
        reads = [{"criterion_id": condition.id, "tool": "minecraft_inspect_block", "ok": True,
                  "observed_dimension": "overworld", "arguments": {"position": point.model_dump()},
                  "result": {"position": point.model_dump(), "block_name": "cobblestone"}}
                 for condition in leaf.criteria for point in condition.positions]
        checks = evaluate(leaf.criteria, reads)
        self.assertFalse(checks["walls"])
        self.assertTrue(checks["marker"])

    def test_air_variants_remain_valid_for_multiple_empty_cells(self):
        condition = Condition(id="empty", description="The interior is empty.", kind="block",
                              names=["air", "cave_air", "void_air"],
                              positions=[{"x": 0, "y": 10, "z": 0}, {"x": 1, "y": 10, "z": 0}])
        leaf = Node(id="clear", kind="leaf", parent="goal", instruction="Clear the interior.", criteria=[condition])
        root = Node(id="goal", kind="goal", children=["clear"], instruction="Make an empty interior.", criteria=[condition])
        validate_tree(TaskTree(goal_id="goal", nodes=[root, leaf]))

