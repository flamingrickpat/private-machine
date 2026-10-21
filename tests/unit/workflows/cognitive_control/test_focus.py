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
from workflows.cognitive_control.attention import active_focus, resolve_focus
from pydantic import BaseModel

from workflows.cognitive_control.schemas import FocusState
from workflows.main_cognition.schemas import FocusProposalRef


class _Evidence(BaseModel):
    goals: list[str] = []


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
        self.assertEqual(focus.strength, 0.2)
        self.assertEqual(focus.promote_ids, [5])
        self.assertFalse(focus.fresh)

    def test_expired_advice_has_no_effect(self):
        ghost = small_ghost()
        _focus_feature(ghost, seconds=1)
        focus = active_focus(ghost, now=datetime.utcnow() + timedelta(seconds=5))
        self.assertIsNone(focus)


class ResolveFocusTests(unittest.TestCase):
    def test_fresh_rejection_keeps_older_advice_and_lists_the_rejection(self):
        context = _Context(['{"accept": false, "reason": "the worry still matters"}'])
        active = FocusState(accept=True, promote_ids=[5], strength=0.2, fresh=False)
        proposals = [FocusProposalRef.model_validate({"record_id": 9, "focus": {
            "instruction": "stop thinking about the message", "promote_ids": [], "reduce_ids": [5]}})]
        result = resolve_focus(context, active, proposals, _Evidence())
        self.assertTrue(result.accept)
        self.assertEqual(result.promote_ids, [5])
        self.assertFalse(result.fresh)
        self.assertEqual([item.proposal_id for item in result.rejections], [9])
        self.assertFalse(result.rejections[0].accept)

    def test_accepted_proposal_becomes_the_fresh_advice(self):
        context = _Context(['{"accept": true, "reason": "it fits the moment"}'])
        active = FocusState(accept=True, promote_ids=[5], strength=0.2)
        proposals = [FocusProposalRef.model_validate({"record_id": 9, "focus": {
            "instruction": "remember the story", "promote_ids": [7], "reduce_ids": [5]}})]
        result = resolve_focus(context, active, proposals, _Evidence())
        self.assertTrue(result.accept)
        self.assertTrue(result.fresh)
        self.assertEqual(result.proposal_id, 9)
        self.assertEqual(result.promote_ids, [7])

    def test_a_rejection_without_older_advice_is_the_whole_result(self):
        context = _Context(['{"accept": false, "reason": "not now"}'])
        proposals = [FocusProposalRef.model_validate({"record_id": 9, "focus": {"instruction": "look away"}})]
        result = resolve_focus(context, None, proposals, _Evidence())
        self.assertFalse(result.accept)
        self.assertEqual(result.instruction, "look away")
        self.assertEqual([item.proposal_id for item in result.rejections], [9])
        self.assertIsNone(resolve_focus(context, None, [], _Evidence()))


if __name__ == "__main__":
    unittest.main()
