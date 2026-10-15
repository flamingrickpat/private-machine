"""Revisable self-model belief contracts against the real commit boundary."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.memory.store import MemoryStore
from pm.records.cognitive_records import BeliefRecord, BeliefShiftRecord
from pm.records.enums import FeatureType

from pm.introspection.capabilities import CapabilityContext, CapabilityDeclaration, build_overlay
from pm.introspection.grants import SourceGrant, SourceInspector
from pm.introspection.self_model import (
    capability_belief_key,
    mark_stale_beliefs,
    record_capability_beliefs,
    record_narrative_belief,
)


def small_ghost() -> MemoryStore:
    from tests.unit.test_design_contracts import small_ghost as make
    return make()


def _overlay(tmp: str, *, name="feature"):
    root = Path(tmp) / "repo"
    (root / "workflows").mkdir(parents=True, exist_ok=True)
    source = root / "workflows" / "feature.py"
    if not source.exists():
        source.write_text("def run():\n    return 1\n", encoding="utf-8")
    inspector = SourceInspector(SourceGrant("test", roots=(root,)))
    declaration = CapabilityDeclaration(name=name, summary="A feature.",
                                        implemented_refs=("workflows/feature.py",))
    return build_overlay([declaration], CapabilityContext(inspector=inspector))


def _beliefs(ghost, key: str) -> list[BeliefRecord]:
    return [record for record in ghost.all_knoxels.values()
            if isinstance(record, BeliefRecord) and record.belief_key == key]


class CapabilityBeliefTests(unittest.TestCase):
    def test_first_inspection_creates_one_belief(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            batch = ghost.begin_batch("cap-1")
            receipt = record_capability_beliefs(batch, _overlay(tmp), companion_name="Spamuel",
                                                source_revision="rev1")
            ghost.commit(batch)
            self.assertEqual(len(receipt["accepted_ids"]), 1)
            stored = _beliefs(ghost, capability_belief_key("feature"))
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0].belief_kind, "capability")
            self.assertEqual(stored[0].metadata["state"], "implemented")

    def test_identical_repeat_does_not_stack_confidence(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            batch = ghost.begin_batch("cap-2")
            record_capability_beliefs(batch, _overlay(tmp), companion_name="Spamuel")
            ghost.commit(batch)
            first = _beliefs(ghost, capability_belief_key("feature"))[0]
            batch = ghost.begin_batch("cap-3")
            receipt = record_capability_beliefs(batch, _overlay(tmp), companion_name="Spamuel")
            ghost.commit(batch)
            self.assertEqual(receipt["accepted_ids"], [])
            self.assertEqual(receipt["retained_ids"], [first.id])
            self.assertEqual(len(_beliefs(ghost, capability_belief_key("feature"))), 1)

    def test_a_changed_finding_supersedes_and_keeps_history(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            batch = ghost.begin_batch("cap-4")
            record_capability_beliefs(batch, _overlay(tmp), companion_name="Spamuel")
            ghost.commit(batch)
            old = _beliefs(ghost, capability_belief_key("feature"))[0]
            (Path(tmp) / "repo" / "workflows" / "feature.py").write_text(
                "def run():\n    return 2\n", encoding="utf-8")
            batch = ghost.begin_batch("cap-5")
            receipt = record_capability_beliefs(batch, _overlay(tmp), companion_name="Spamuel")
            ghost.commit(batch)
            self.assertEqual(len(receipt["accepted_ids"]), 1)
            self.assertEqual(receipt["revised"], [old.id])
            superseded = ghost.get_knoxel_by_id(old.id)
            self.assertFalse(superseded.active)
            self.assertEqual(superseded.superseded_by_id, receipt["accepted_ids"][0])
            shifts = [record for record in ghost.all_knoxels.values()
                      if isinstance(record, BeliefShiftRecord) and record.target_belief_id == old.id]
            self.assertEqual(len(shifts), 1)
            self.assertEqual(shifts[0].shift_type, "supersede")

    def test_staleness_is_recorded_without_erasing_the_belief(self):
        ghost = small_ghost()
        with TemporaryDirectory() as tmp:
            batch = ghost.begin_batch("cap-6")
            record_capability_beliefs(batch, _overlay(tmp), companion_name="Spamuel")
            ghost.commit(batch)
            batch = ghost.begin_batch("cap-7")
            marked = mark_stale_beliefs(batch, ["feature"], revision="rev2",
                                        reason="the configuration revision changed")
            ghost.commit(batch)
            self.assertEqual(len(marked), 1)
            belief = _beliefs(ghost, capability_belief_key("feature"))[0]
            self.assertTrue(belief.active)
            self.assertTrue(belief.metadata["stale"])
            self.assertEqual(belief.metadata["stale_revision"], "rev2")


class NarrativeBeliefTests(unittest.TestCase):
    def test_narrative_revision_keeps_the_original_episode_reference(self):
        ghost = small_ghost()
        from pm.records.memory_records import new_feature
        from pm.records.enums import InterlocusType, SourceKind
        thought = new_feature(ghost, "I keep returning to the story.", FeatureType.InnerMonologue,
                              source="Spamuel", interlocus=InterlocusType.PrivateInternal,
                              source_kind=SourceKind.Self, metadata={"loop": "inner"})
        with TemporaryDirectory() as tmp:
            batch = ghost.begin_batch("narr-1")
            first = record_narrative_belief(batch, topic="the story", text="The story still matters to me.",
                                            evidence_ids=[thought.id], episode_ids=[thought.id],
                                            companion_name="Spamuel")
            ghost.commit(batch)
            first_id = first["accepted_ids"][0]
            batch = ghost.begin_batch("narr-2")
            second = record_narrative_belief(batch, topic="the story",
                                             text="I may have outgrown the story.",
                                             evidence_ids=[thought.id], episode_ids=[thought.id],
                                             companion_name="Spamuel")
            ghost.commit(batch)
            self.assertEqual(second["revised"], [first_id])
            old = ghost.get_knoxel_by_id(first_id)
            self.assertEqual(old.supporting_knoxel_ids, [thought.id])
            self.assertTrue(old.metadata["episode_ids"])
            self.assertFalse(old.active)

    def test_a_dropped_evidence_id_is_reported_not_linked(self):
        ghost = small_ghost()
        batch = ghost.begin_batch("narr-3")
        result = record_narrative_belief(batch, topic="missing", text="A reading.",
                                         evidence_ids=[999999], companion_name="Spamuel")
        self.assertEqual(result["dropped_ids"], [999999])
        belief = ghost.begin_batch("narr-4")
        created = record_narrative_belief(belief, topic="missing", text="A reading.",
                                          evidence_ids=[999999], companion_name="Spamuel")
        ghost.commit(belief)
        stored = ghost.get_knoxel_by_id(created["accepted_ids"][0])
        self.assertEqual(stored.supporting_knoxel_ids, [])


if __name__ == "__main__":
    unittest.main()
