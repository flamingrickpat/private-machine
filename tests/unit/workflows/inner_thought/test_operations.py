"""Old-catalogue mapping and delivered operation contracts.

No model, kernel, or MemoryStore. The mapping must cover all 37 historical
thought types so nothing silently disappears.
"""
from __future__ import annotations

import unittest

from workflows.inner_thought.operations import (
    GROUPS, OLD_TYPE_MAPPING, OPERATIONS, Operation, validate_operation,
)

HISTORICAL_TYPES = [
    "TH_PRESENT_APPRAISAL", "TH_IMAGINATIVE_PROJECTION", "TH_FEASIBILITY_CHECK", "TH_MEMORY_SEARCH",
    "TH_INTROSPECTIVE_RUMINATION", "TH_PLAN_NEXT_STEP", "TH_COMPOSE_REPLY", "TH_BACKTRACK_PIVOT",
    "TH_EPISODIC_SNAPSHOT", "TH_PATTERN_MINING", "TH_COUNTERFACTUAL_REWRITE", "TH_REGRET_AUDIT",
    "TH_GRATITUDE_RECALL", "TH_SKILL_ABSTRACTION", "TH_SOURCE_ATTRIBUTION", "TH_NARRATIVE_STITCHING",
    "TH_EMOTIONAL_DEBRIEF", "TH_MEMORY_GAP_PROBE", "TH_NEED_CHECK", "TH_BOUNDARY_GUARD",
    "TH_PERSPECTIVE_SHIFT", "TH_CURIOSITY_SPARK", "TH_SOCIAL_TEMPERATURE_READ", "TH_SYSTEMS_SCAN",
    "TH_VALUE_ALIGNMENT_CHECK", "TH_SELF_COMPASSION_TAKE", "TH_OPPORTUNITY_SPOTTING", "TH_SCENARIO_SKETCH",
    "TH_RISK_FORECAST", "TH_MICRO_PLAN", "TH_CONTINGENCY_BRANCHING", "TH_COMMITMENT_TEST",
    "TH_SOCIAL_BID_CRAFTING", "TH_LONG_HORIZON_WEAVE", "TH_WISHFUL_IDEATION", "TH_SIMULATED_FEEDBACK",
    "TH_EXIT_PIVOT_DECISION",
]


class MappingTests(unittest.TestCase):
    def test_every_historical_type_is_mapped(self):
        self.assertEqual(sorted(OLD_TYPE_MAPPING), sorted(HISTORICAL_TYPES))
        self.assertEqual(len(HISTORICAL_TYPES), 37)

    def test_every_mapped_successor_is_a_delivered_operation(self):
        for old_name, successor in OLD_TYPE_MAPPING.items():
            if successor is None:
                continue
            self.assertIn(successor, OPERATIONS, f"{old_name} maps to an unknown operation")

    def test_every_group_from_the_work_item_is_covered(self):
        groups = {operation.group for operation in OPERATIONS.values()}
        self.assertEqual(groups, set(GROUPS))
        for group in GROUPS:
            self.assertTrue(any(operation.group == group for operation in OPERATIONS.values()))


class OperationTests(unittest.TestCase):
    def test_operations_declare_mode_and_focus(self):
        for operation in OPERATIONS.values():
            self.assertIsInstance(operation, Operation)
            self.assertIn(operation.mode, {"generative", "discriminatory"})
            self.assertIn(operation.temporal_focus, {"past", "present", "future"})

    def test_validate_operation_rejects_unknown_names(self):
        with self.assertRaises(KeyError):
            validate_operation("TH_COMPOSE_REPLY")


if __name__ == "__main__":
    unittest.main()
