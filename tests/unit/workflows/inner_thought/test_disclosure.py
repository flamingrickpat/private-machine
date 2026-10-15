"""Disclosure contract tests: selected private state enters public context.

No model or kernel. The real MemoryStore and the real public context builder
run, so the lane, the disclosure gate, and the revision guards are exercised
exactly as dialogue uses them.
"""
from __future__ import annotations

import unittest

from pm.records.enums import FeatureType, SourceKind
from pm.records.memory_records import new_feature
from pm.records.qualia_records import QualeRecord

from workflows.public_speech.public_context import build_public_context, memory_sections


def small_ghost():
    from tests.unit.test_design_contracts import small_ghost as make
    return make()


def private_thought(ghost, text, mode):
    return new_feature(
        ghost, text, FeatureType.InnerMonologue, source=ghost.get_companion_name(),
        source_kind=SourceKind.Self,
        metadata={"loop": "inner", "operation": "future_projection",
                  "disclosure": {"mode": mode, "shareable_summary": "", "reason": "test"}},
    )


def quale(ghost, reportable):
    return ghost.store_knoxel(QualeRecord(
        content="a warm pull toward the story", channel_id="affect.valence",
        position=0.4, velocity=0.05, acceleration=0.0, driver_ids=[1], reportable=reportable,
    ))


class DisclosureTests(unittest.TestCase):
    def test_candid_thought_and_reportable_quale_enter_the_private_lane(self):
        ghost = small_ghost()
        candid = private_thought(ghost, "I want to invite Jesse to co-write.", "candid")
        reportable = quale(ghost, True)
        context = build_public_context(ghost, {"stimuli": []})
        private = [item for item in context.evidence if item.get("lane") == "private_state"]
        ids = {item["knoxel_id"] for item in private}
        self.assertIn(candid.id, ids)
        self.assertIn(reportable.id, ids)
        self.assertIn(candid.id, context.revisions)
        self.assertIn(reportable.id, context.revisions)
        # The lane renders under its title in dialogue prompts.
        sections = memory_sections(context, [
            {"lane": "private_state",
             "title": "## What Spamuel privately noticed and chose to share"}])
        self.assertTrue(any("co-write" in section for section in sections))

    def test_withheld_thought_and_dormant_quale_stay_private(self):
        ghost = small_ghost()
        withheld = private_thought(ghost, "I resented the silence more than I admitted.", "withhold")
        dormant = quale(ghost, False)
        context = build_public_context(ghost, {"stimuli": []})
        texts = [str(item.get("text", "")) for item in context.evidence]
        self.assertFalse(any("resented" in text for text in texts))
        ids = {item["knoxel_id"] for item in context.evidence}
        self.assertNotIn(withheld.id, ids)
        self.assertNotIn(dormant.id, ids)


if __name__ == "__main__":
    unittest.main()
