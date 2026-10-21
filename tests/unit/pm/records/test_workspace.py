"""Workspace state models and the requirement file (A19F step 2). No model calls."""
from __future__ import annotations

import unittest
from pathlib import Path

from pydantic import ValidationError

from pm.memory.yaml_codec import KNOXEL_TYPE_REGISTRY
from pm.records.workspace import (GlobalWorkspace, SELF_SHARD, ShardKey, WorkspaceItem,
                                  load_workspace_requirements)

ROOT = Path(__file__).resolve().parents[4]
REQUIREMENTS = ROOT / "workflows" / "main_cognition" / "workspace.yaml"


class ShardTests(unittest.TestCase):
    def test_ensure_shard_creates_entity_and_workflow_shards_once(self):
        workspace = GlobalWorkspace()
        entity = workspace.ensure_shard(ShardKey(kind="entity", ref="2"))
        entity.items.append(WorkspaceItem(record_id=10, tokens=5))
        self.assertIs(workspace.ensure_shard(ShardKey(kind="entity", ref="2")), entity)
        workspace.ensure_shard(ShardKey(kind="workflow", ref="minecraft")).tail_ids.append(11)
        self.assertEqual(entity.requirement_set, "entity")
        self.assertIs(workspace.shard(SELF_SHARD), workspace.self_shard)
        self.assertEqual([shard.key.kind for shard in workspace.all_shards()], ["self", "entity", "workflow"])
        self.assertEqual(workspace.item_ids(), [10])
        self.assertEqual(entity.content_tokens(), 5)

    def test_missing_shard_raises(self):
        with self.assertRaises(KeyError):
            GlobalWorkspace().shard(ShardKey(kind="workflow", ref="minecraft"))

    def test_shard_key_is_hashable(self):
        self.assertEqual({ShardKey(kind="entity", ref="1"), ShardKey(kind="entity", ref="1")},
                         {ShardKey(kind="entity", ref="1")})

    def test_json_round_trip_restores_int_entity_keys(self):
        workspace = GlobalWorkspace()
        workspace.ensure_shard(ShardKey(kind="entity", ref="7")).items.append(
            WorkspaceItem(record_id=3, entered_tick=4, activation_at_entry=0.6, required_by="entity_narrative"))
        workspace.ensure_shard(ShardKey(kind="workflow", ref="minecraft"))
        dumped = workspace.model_dump(mode="json")
        self.assertEqual(list(dumped["entity_shards"]), ["7"])
        restored = GlobalWorkspace.model_validate(dumped)
        self.assertEqual(restored, workspace)
        self.assertEqual(list(restored.entity_shards), [7])


class RequirementTests(unittest.TestCase):
    def test_main_cognition_requirements_load(self):
        config = load_workspace_requirements(REQUIREMENTS)
        self.assertEqual(sorted(config.shards), ["entity", "self", "workflow"])
        self_names = [item.name for item in config.shards["self"].requirements]
        self.assertIn("self_narrative", self_names)
        narrative = config.shards["entity"].requirements[0]
        self.assertEqual((narrative.select.entity, narrative.min_items), ("shard", 1))
        self.assertEqual([item.name for item in config.csm.requirements],
                         ["current_stimuli", "self_narratives", "group_narratives"])

    def test_selectors_name_real_record_classes(self):
        known = {cls.__name__ for cls in KNOXEL_TYPE_REGISTRY.values()}
        config = load_workspace_requirements(REQUIREMENTS)
        sets = [*config.shards.values(), config.csm]
        names = {name for requirement_set in sets for item in requirement_set.requirements for name in item.select.record}
        self.assertTrue(names)
        self.assertEqual(names - known, set())

    def test_unknown_shard_kind_is_rejected(self):
        from pm.records.workspace import WorkspaceRequirementConfig
        with self.assertRaises(ValidationError):
            WorkspaceRequirementConfig.model_validate({"shards": {"room": {"requirements": []}}})


if __name__ == "__main__":
    unittest.main()
