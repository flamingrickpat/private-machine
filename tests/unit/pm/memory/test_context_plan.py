"""Pure context planning: lanes, budgets, focus, boundaries, redistribution.

No model, no embeddings service, no MemoryStore. Records are serialized
dictionaries, vectors are small fixed lists, and every record costs a fixed
number of tokens, so each budget decision is easy to read.
"""
from __future__ import annotations

import unittest
from datetime import datetime

from pm.memory.context_plan import Focus, LaneSpec, LaneWeights, PlanSpec, plan_context

NOW = datetime(2026, 10, 13, 12, 0)


def record(kind: str, content: str, day: int, **data) -> dict:
    stamp = f"2026-05-{day:02d}T20:00:00"
    return {"__type__": kind, "data": {"content": content, "timestamp_world_begin": stamp,
                                       "timestamp_world_end": stamp, **data}}


def records() -> dict[int, dict]:
    return {
        1: record("Feature", "old line about soup", 1, feature_type="Dialogue"),
        2: record("Feature", "line about the plant", 20, feature_type="Dialogue"),
        3: record("Feature", "private thought about Jesse", 21, feature_type="Thought"),
        4: record("Feature", "newest line about crackers", 22, feature_type="Dialogue"),
        10: record("MemoryClusterKnoxel", "summary: the plant almost died", 5, cluster_type="temporal"),
        11: record("MemoryClusterKnoxel", "summary: crackers versus soup", 21, cluster_type="temporal"),
        12: record("MemoryClusterKnoxel", "summary: a quiet evening", 6, cluster_type="temporal"),
        20: record("DeclarativeFactKnoxel", "Jesse owns a plant", 3, importance=0.9),
        21: record("DeclarativeFactKnoxel", "Jesse dislikes homework", 3, importance=0.2),
        30: record("Narrative", "Spamuel sees himself as dramatic (v1)", 2, narrative_type="SelfImage", entity_id=1),
        31: record("Narrative", "Spamuel sees himself as dramatic and loyal (v2)", 9, narrative_type="SelfImage", entity_id=1),
    }


VECTORS = {10: [1.0, 0.0], 11: [0.0, 1.0], 12: [0.7, 0.7], 20: [1.0, 0.0], 21: [0.0, 1.0]}


def lane(name, share, select="score", **extra) -> LaneSpec:
    return LaneSpec(name=name, budget_share=share, select=select, **extra)


def run(spec: PlanSpec, focus: Focus, **kwargs):
    return plan_context(records(), spec, focus, vector_of=VECTORS.get, tokens_of=lambda key: 10, now=NOW, **kwargs)


class LaneTests(unittest.TestCase):
    def test_recent_lane_is_the_newest_contiguous_window_in_time_order(self):
        spec = PlanSpec(name="t", total_tokens=100, lanes=[
            lane("dialogue", 0.3, "recent", record_types=["Feature"], feature_types=["Dialogue", "Thought"])])
        plan = run(spec, Focus())
        self.assertEqual(plan.lane("dialogue").ids, [2, 3, 4])

    def test_history_stays_older_than_the_dialogue_window(self):
        spec = PlanSpec(name="t", total_tokens=100, lanes=[
            lane("dialogue", 0.2, "recent", record_types=["Feature"], feature_types=["Dialogue"]),
            lane("history", 0.5, record_types=["MemoryClusterKnoxel"], before_lane="dialogue")])
        plan = run(spec, Focus())
        self.assertEqual(plan.lane("dialogue").ids, [2, 4])
        self.assertNotIn(11, plan.lane("history").ids)  # day 21 overlaps the window start (day 20)
        self.assertEqual(set(plan.lane("history").ids), {10, 12})

    def test_score_lane_ranks_by_focus_and_records_who_lost(self):
        spec = PlanSpec(name="t", total_tokens=100, lanes=[
            lane("history", 0.2, order="score", record_types=["MemoryClusterKnoxel"])])
        plan = run(spec, Focus(texts=[(1.0, "the plant")], vectors=[(1.0, [1.0, 0.0])]))
        history = plan.lane("history")
        self.assertEqual(history.ids, [10, 12])
        self.assertEqual([item["id"] for item in history.competing], [11])
        self.assertGreater(history.scores[10], history.scores[11])

    def test_importance_weight_lifts_an_important_fact(self):
        spec = PlanSpec(name="t", total_tokens=100, lanes=[
            lane("facts", 0.1, order="score", record_types=["DeclarativeFactKnoxel"],
                 weights=LaneWeights(semantic=0.0, lexical=0.0, importance=1.0))])
        self.assertEqual(run(spec, Focus()).lane("facts").ids, [20])

    def test_focus_lane_holds_only_required_records_and_excluded_never_appear(self):
        spec = PlanSpec(name="t", total_tokens=200, lanes=[
            lane("dialogue", 0.2, "recent", record_types=["Feature"], feature_types=["Dialogue"]),
            lane("focus", 0.2, "focus"),
            lane("facts", 0.2, record_types=["DeclarativeFactKnoxel"])])
        plan = run(spec, Focus(required_ids=[1, 12, 21, 999], exclude_ids=[4]))
        # Record 1 is already in the dialogue window; a record fills one lane only.
        self.assertIn(1, plan.lane("dialogue").ids)
        self.assertEqual(plan.lane("focus").ids, [21, 12])
        self.assertNotIn(4, plan.ids())
        self.assertEqual(len(plan.ids()), len(set(plan.ids())))

    def test_near_duplicates_fill_one_place(self):
        spec = PlanSpec(name="t", total_tokens=100, lanes=[
            lane("facts", 0.5, record_types=["DeclarativeFactKnoxel", "MemoryClusterKnoxel"], max_similarity=0.95,
                 weights=LaneWeights(semantic=1.0, lexical=0.0))])
        # Records 10 and 20 have the same vector; only the stronger one stays.
        plan = run(spec, Focus(vectors=[(1.0, [1.0, 0.0])]))
        facts = plan.lane("facts")
        self.assertEqual(len({10, 20} & set(facts.ids)), 1)
        self.assertIn("similar", {item["reason"] for item in facts.competing})
        self.assertTrue(facts.exhausted)

    def test_latest_per_keeps_the_newest_narrative_refinement(self):
        spec = PlanSpec(name="t", total_tokens=100, lanes=[
            lane("narratives", 0.5, record_types=["Narrative"], latest_per=["narrative_type", "entity_id"])])
        self.assertEqual(run(spec, Focus()).lane("narratives").ids, [31])


class BudgetTests(unittest.TestCase):
    def test_spare_budget_moves_to_the_first_hungry_priority_lane(self):
        spec = PlanSpec(name="t", total_tokens=100, priority=["history"], lanes=[
            lane("facts", 0.5, record_types=["DeclarativeFactKnoxel"]),
            lane("history", 0.1, record_types=["MemoryClusterKnoxel"])])
        plan = run(spec, Focus())
        self.assertEqual(plan.lane("facts").used, 20)
        self.assertEqual(len(plan.lane("history").ids), 3)
        self.assertEqual(plan.rounds[0]["to"], "history")

    def test_unknown_lane_names_are_rejected(self):
        spec = PlanSpec(name="t", total_tokens=100, priority=["missing"], lanes=[lane("facts", 0.5)])
        with self.assertRaises(ValueError):
            run(spec, Focus())


if __name__ == "__main__":
    unittest.main()
