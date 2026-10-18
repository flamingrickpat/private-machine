"""Reply context planning against a real MemoryStore, without embeddings.

The store is small and synthetic. Without an embedding service the planner
scores lexically; the trace records that. These tests fix the integration
contract: the dialogue window holds verbatim speech and thoughts in order,
the current message never enters a lane, workspace targets are required, and
the public context carries lane-tagged evidence and the plan trace.
"""
from __future__ import annotations

import asyncio
import unittest

from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType
from pm.records.memory_records import new_feature
from workflows.main_cognition.schemas import ActionRequest, Stimulus
from workflows.public_speech.context_planning import FocusInput, plan_reply_context
from workflows.public_speech.public_context import build_public_context


class ContextPlanningTests(unittest.TestCase):
    def setUp(self):
        self.ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        self.say = lambda text, source="Jesse", kind=FeatureType.Dialogue: new_feature(self.ghost, text, kind, source=source)
        self.older = self.say("my plant has one dark leaf")
        self.say("then the leaf is filing a complaint", source="Spamuel")
        self.thought = self.say("he always comes back to the plant", source="Spamuel", kind=FeatureType.Thought)
        self.current = self.say("is the plant going to be ok?")

    def plan(self, plan_name="story", **focus):
        focus = FocusInput(current_message=self.current.content, exclude_ids=[self.current.id], **focus)
        return asyncio.run(plan_reply_context(self.ghost.snapshot(), plan_name, focus, None))

    def test_story_window_is_verbatim_speech_and_thoughts_without_the_current_message(self):
        planned = self.plan()
        self.assertEqual([(turn.speaker, turn.kind) for turn in planned.dialogue],
                         [("Jesse", "speech"), ("Spamuel", "speech"), ("Spamuel", "thought")])
        self.assertNotIn(self.current.id, [turn.knoxel_id for turn in planned.dialogue])
        self.assertEqual(planned.trace["semantic"], {"status": "disabled"})

    def test_workspace_targets_reach_the_focus_lane_and_the_public_context(self):
        fact = new_feature(self.ghost, "Jesse waters the plant on Sundays", FeatureType.WorldEvent, source="Jesse")
        planned = self.plan(required_ids=[fact.id])
        self.assertEqual([(item.knoxel_id, item.lane) for item in planned.evidence], [(fact.id, "focus")])
        context = build_public_context(self.ghost, ActionRequest(stimuli=[Stimulus(knoxel_id=self.current.id)]), planned=planned)
        lanes = {item.knoxel_id: item.lane for item in context.evidence}
        self.assertEqual(lanes[fact.id], "focus")
        self.assertIn(self.current.id, lanes)
        self.assertEqual(context.plan["plan"], "story")


if __name__ == "__main__":
    unittest.main()
