"""Capability overlay built from the live application context.

The repository is the real one. The records are real committed records.
"""
from __future__ import annotations

from pathlib import Path
import unittest

from pm.memory.store import MemoryStore
from pm.records.cognitive_records import BeliefRecord
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import new_feature

from workflows.introspection.snapshot import (
    build_capability_overlay,
    enabled_capability_names,
    observed_capability_receipts,
    overlay_public,
)


def small_ghost() -> MemoryStore:
    return MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))


class EnabledMappingTests(unittest.TestCase):
    def test_module_ids_map_to_capabilities(self):
        names = enabled_capability_names({"inner_thought", "dialogue_story"})
        self.assertIn("source_inspection", names)
        self.assertIn("private_thought", names)
        self.assertIn("public_dialogue", names)
        self.assertNotIn("minecraft_body", names)


class ObservedReceiptTests(unittest.TestCase):
    def test_a_committed_private_thought_is_an_observation(self):
        ghost = small_ghost()
        new_feature(ghost, "I keep thinking about the story.", FeatureType.InnerMonologue,
                    source="Spamuel", interlocus=InterlocusType.PrivateInternal,
                    source_kind=SourceKind.Self, metadata={"loop": "inner"})
        rows = observed_capability_receipts(ghost)
        self.assertIn("private_thought", rows)
        self.assertTrue(rows["private_thought"][0]["reference"].startswith("feature:"))

    def test_a_capability_belief_is_a_source_inspection_observation(self):
        ghost = small_ghost()
        from pm.records.base import Provenance
        ghost.store_knoxel(BeliefRecord(content="I can inspect my source.", belief_key="k",
                                        provenance=Provenance(source_kind=SourceKind.Self, source_ref="test"),
                                        metadata={"memory_kind": "capability_belief"}))
        self.assertIn("source_inspection", observed_capability_receipts(ghost))


class OverlayBuildTests(unittest.TestCase):
    def test_overlay_reads_the_real_repository(self):
        ghost = small_ghost()
        overlay = build_capability_overlay(ghost=ghost, active_modules={"inner_thought"})
        self.assertIn("source_inspection", overlay.findings)
        finding = overlay.findings["source_inspection"]
        self.assertTrue(finding.has("implemented"))
        self.assertEqual(finding.strongest.value, "enabled")
        public = overlay_public(overlay)
        self.assertTrue(public["available"])
        names = {item["name"] for item in public["capabilities"]}
        self.assertIn("private_thought", names)

    def test_overlay_public_without_an_overlay_is_explicit(self):
        self.assertEqual(overlay_public(None),
                         {"available": False, "checked_at": None, "capabilities": []})


if __name__ == "__main__":
    unittest.main()
