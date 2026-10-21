"""Exercise draft tools over real MCP with a captured retained project tree."""
import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastmcp import Client

from pm.host.mcp_listeners import serve_services
from workflows.minecraft_project.planning.draft import TaskDraft
from workflows.minecraft_project.schemas import TaskTree
from workflows.minecraft_project.tree_store import TreeStore

ROOT = Path(__file__).resolve().parents[4]


class TaskDraftTests(unittest.TestCase):
    def test_real_tools_preserve_captured_ids_and_keep_edits_uncommitted(self):
        capture = json.loads((ROOT / "test_data/captures/alpha/project-stable-child-rejection.json").read_text(encoding="utf-8"))
        original = TaskTree.model_validate(capture["tree"])
        with TemporaryDirectory() as directory:
            store = TreeStore(Path(directory))
            retained = store.revise(original)
            before = store.path.read_bytes()
            snapshot = store.path_for("planning/specs/0001.json").read_bytes()
            draft = TaskDraft(store, retained.nodes[0].instruction,
                lambda value: store.write("draft.json", value), lambda: None, base=retained)

            async def exercise(descriptors):
                async with Client(descriptors["draft"]["url"]) as client:
                    added = await client.call_tool("task_add", {"kind": "leaf",
                        "instruction": "Lay the new five by seven floor.", "specialist": "builder"})
                    node_id = added.data["node_id"]
                    self.assertEqual(added.data["parent"], "root_goal")
                    first = await client.call_tool("plan_check", {})
                    self.assertFalse(first.data["accepted"])
                    fixed = await client.call_tool("condition_set", {"node_id": node_id,
                        "condition": {"kind": "block", "names": ["birch_planks"],
                            "box": {"min": {"x": 0, "y": 10, "z": 0}, "max": {"x": 4, "y": 10, "z": 6}}}})
                    self.assertEqual(fixed.data["exact_cells"], 35)
                    leaf = await client.call_tool("task_get", {"node_id": node_id})
                    self.assertEqual(len(leaf.data["task"]["criteria"][0]["positions"]), 35)
                    for corner in ({"x": 0, "y": 10, "z": 0}, {"x": 4, "y": 10, "z": 6}):
                        self.assertIn(corner, leaf.data["task"]["criteria"][0]["positions"])
                    copied = await client.call_tool("condition_copy", {"target_id": retained.goal_id,
                        "source_id": node_id})
                    self.assertEqual(len(copied.data["copied_conditions"]), 1)
                    accepted = await client.call_tool("plan_check", {})
                    self.assertTrue(accepted.data["accepted"], accepted.data["feedback"])

            with serve_services({"draft": draft.server()}) as descriptors:
                asyncio.run(exercise(descriptors))
            final = draft.validate("Ready.").value.tree
            nodes = {node.id: node for node in final.nodes}
            for node in retained.nodes:
                if node.id != retained.goal_id:
                    self.assertEqual(nodes[node.id], node)
            self.assertTrue(set(retained.nodes[0].children) <= set(nodes[retained.goal_id].children))
            self.assertEqual(store.path.read_bytes(), before)
            self.assertEqual(store.path_for("planning/specs/0001.json").read_bytes(), snapshot)
            self.assertFalse(store.path_for("planning/specs/0002.json").exists())

    def test_invalid_condition_does_not_replace_a_valid_edit(self):
        with TemporaryDirectory() as directory:
            store = TreeStore(Path(directory))
            draft = TaskDraft(store, "A real draft boundary.", lambda value: None, lambda: None)
            draft.condition("goal", {"id": "alive", "kind": "alive"})
            before = json.dumps(draft.nodes, sort_keys=True)
            rejected = draft._call(lambda: draft.condition("goal", {"id": "alive", "kind": "block"}))
            self.assertFalse(rejected["ok"])
            self.assertEqual(json.dumps(draft.nodes, sort_keys=True), before)
            self.assertEqual(store.state["revision"], 0)


if __name__ == "__main__":
    unittest.main()
