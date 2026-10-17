"""Focus advice tests: expiry, current caps, and rejection receipts.

No model endpoint. The focus resolver receives a deterministic completion
through the stage-support boundary; the memory store is real.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pm.memory.bash_view import MemoryBashEnvironment
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import new_feature

from tests.unit.test_design_contracts import small_ghost
from workflows.cognitive_control.attention import active_focus, attention_bias, resolve_focus
from workflows.main_cognition.schemas import AttentionNomination


class _Runtime:
    def __init__(self, replies, environment):
        self.tools = {"llm_complete": lambda messages, **generation: replies.pop(0)}
        self.environment = environment


class _Context:
    def __init__(self, replies):
        environment = MemoryBashEnvironment({}, {}, "read")
        self.runtime = _Runtime(replies, environment)
        self.kernel_data = Path(".")


def _focus_feature(ghost, *, strength=0.4, seconds=900):
    batch = ghost.begin_batch("focus-seed")
    feature = new_feature(ghost, "remember the story", FeatureType.StateUpdate,
        source=ghost.get_companion_name(), source_kind=SourceKind.Self,
        interlocus=InterlocusType.PrivateInternal, changes=batch,
        metadata={"control_kind": "attention_focus",
                  "focus": {"accept": True, "instruction": "keep the story in mind",
                            "promote_ids": [5], "reduce_ids": [], "strength": strength},
                  "expires_at": (datetime.utcnow() + timedelta(seconds=seconds)).isoformat()})
    ghost.commit(batch)
    return feature


class ActiveFocusTests(unittest.TestCase):
    def test_current_cap_clamps_stored_advice(self):
        ghost = small_ghost()
        _focus_feature(ghost, strength=0.4)
        focus = active_focus(ghost, maximum_strength=0.2)
        self.assertEqual(focus["strength"], 0.2)
        self.assertEqual(focus["promote_ids"], [5])
        self.assertFalse(focus["fresh"])

    def test_expired_advice_has_no_effect(self):
        ghost = small_ghost()
        _focus_feature(ghost, seconds=1)
        focus = active_focus(ghost, now=datetime.utcnow() + timedelta(seconds=5))
        self.assertEqual(focus, {})


class ResolveFocusTests(unittest.TestCase):
    def test_fresh_rejection_keeps_older_advice_and_lists_the_rejection(self):
        context = _Context(['{"accept": false, "reason": "the worry still matters"}'])
        request = {"active_focus": {"accept": True, "promote_ids": [5], "strength": 0.2,
                                     "fresh": False},
                   "focus_proposals": [{"record_id": 9, "focus": {
                       "instruction": "stop thinking about the message", "promote_ids": [], "reduce_ids": [5]}}]}
        result = resolve_focus(context, request)
        self.assertTrue(result["accept"])
        self.assertEqual(result["promote_ids"], [5])
        self.assertFalse(result["fresh"])
        self.assertEqual([item["proposal_id"] for item in result["rejections"]], [9])
        self.assertFalse(result["rejections"][0]["accept"])

    def test_accepted_proposal_becomes_the_fresh_advice(self):
        context = _Context(['{"accept": true, "reason": "it fits the moment"}'])
        request = {"active_focus": {"accept": True, "promote_ids": [5], "strength": 0.2},
                   "focus_proposals": [{"record_id": 9, "focus": {
                       "instruction": "remember the story", "promote_ids": [7], "reduce_ids": [5]}}]}
        result = resolve_focus(context, request)
        self.assertTrue(result["accept"])
        self.assertTrue(result["fresh"])
        self.assertEqual(result["proposal_id"], 9)
        self.assertEqual(result["promote_ids"], [7])


class AttentionBiasTests(unittest.TestCase):
    def test_reduction_does_not_defeat_urgent_or_dangerous_candidates(self):
        focus = {"accept": True, "strength": 0.3, "promote_ids": [], "reduce_ids": [5]}
        calm = AttentionNomination(nomination_id="a", target_ids=[5], urgency=0.1)
        urgent = AttentionNomination(nomination_id="b", target_ids=[5], urgency=0.9)
        biases = attention_bias([calm, urgent], focus, danger=False)
        self.assertLess(biases["a"], 0)
        self.assertEqual(biases["b"], 0.0)
        danger = attention_bias([calm], focus, danger=True)
        self.assertEqual(danger["a"], 0.0)

    def test_rejected_focus_applies_no_bias(self):
        focus = {"accept": False, "reduce_ids": [5], "strength": 0.3}
        calm = AttentionNomination(nomination_id="a", target_ids=[5], urgency=0.1)
        self.assertEqual(attention_bias([calm], focus, danger=False), {})


if __name__ == "__main__":
    unittest.main()
