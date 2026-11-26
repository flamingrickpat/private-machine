"""Reply context planning against a real MemoryStore, without embeddings.

The planner uses its lexical score for these inputs. Spoken lines and private
history keep their source IDs and their prompt roles through final Story fitting.
"""
from __future__ import annotations

import asyncio
import unittest

from pm.memory.store import MemoryStore
from pm.context_render.turns import Role
from pm.records.base import RecordBase
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import FeatureType
from pm.records.memory_records import new_feature
from workflows.dialogue_story.context import select_story_prompt
from workflows.dialogue_story.schemas import StoryRequest
from workflows.dialogue_story.steps import load_prompts
from workflows.main_cognition.schemas import ActionRequest, Stimulus
from workflows.public_speech.context_planning import FocusInput, PlannedContext, plan_reply_context
from workflows.public_speech.public_context import PublicContext, build_public_context


class ContextPlanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        self.older = self.say("my plant has one dark leaf")
        self.reply = self.say("then the leaf is filing a complaint", source="Spamuel")
        self.thought = self.say("he always comes back to the plant", source="Spamuel", kind=FeatureType.Thought)
        self.current = self.say("is the plant going to be ok?")

    def say(self, text: str, source: str = "Jesse", kind: FeatureType = FeatureType.Dialogue) -> RecordBase:
        return new_feature(self.ghost, text, kind, source=source)

    def plan(self, required_ids: tuple[int, ...] = ()) -> PlannedContext:
        focus = FocusInput(current_message=self.current.content, exclude_ids=[self.current.id],
                           current_ids=(self.current.id,), required_ids=list(required_ids))
        return asyncio.run(plan_reply_context(self.ghost.snapshot(), "story", focus))

    def test_story_window_keeps_spoken_dialogue_and_private_thought_in_their_roles(self) -> None:
        planned = self.plan()
        self.assertEqual([(turn.speaker, turn.kind) for turn in planned.dialogue],
                         [("Jesse", "speech"), ("Spamuel", "speech")])
        self.assertNotIn(self.current.id, [turn.knoxel_id for turn in planned.dialogue])
        self.assertEqual(planned.trace.semantic.status, "disabled")
        context = PublicContext(companion_name="Spamuel", user_name="Jesse", temporal_history=planned.temporal_history)
        story = StoryRequest(companion_name="Spamuel", partner_name="Jesse", character_card_story="Spamuel.",
            prior_dialogue=planned.dialogue, current_user_message=self.current.content, current_message_id=self.current.id,
            input_budget_tokens=12000, public_context=context)
        selected = select_story_prompt(story, load_prompts())
        transcript = "\n".join(turn.content.text for turn in selected.selection.turns if turn.role is Role.Assistant)
        artifacts = "\n".join(turn.content.text for turn in selected.selection.turns if turn.role is Role.User)
        self.assertIn(self.reply.content, transcript)
        self.assertIn(self.thought.content, artifacts)
        self.assertNotIn(self.thought.content, transcript)

    def test_workspace_targets_reach_the_focus_lane_and_the_public_context(self) -> None:
        fact = new_feature(self.ghost, "Jesse waters the plant on Sundays", FeatureType.WorldEvent, source="Jesse")
        planned = self.plan(required_ids=(fact.id,))
        self.assertEqual([(item.knoxel_id, item.lane) for item in planned.evidence], [(fact.id, "focus")])
        context = build_public_context(self.ghost, ActionRequest(stimuli=[Stimulus(knoxel_id=self.current.id)]), planned=planned)
        lanes = {item.knoxel_id: item.lane for item in context.evidence}
        self.assertEqual(lanes[fact.id], "focus")
        self.assertIn(self.current.id, lanes)
        assert context.plan is not None
        self.assertEqual(context.plan.plan, "story")

    def test_empty_history_does_not_enter_the_standalone_transcript(self) -> None:
        blanks = [self.say(text, source="Spamuel", kind=kind).id
                  for text in ("", " \t\r\n", "\u00a0\u2003")
                  for kind in (FeatureType.Dialogue, FeatureType.Thought, FeatureType.InnerMonologue)]
        planned = self.plan()
        self.assertEqual([turn.knoxel_id for turn in planned.dialogue],
                         [self.older.id, self.reply.id])
        self.assertTrue(set(blanks).isdisjoint(turn.knoxel_id for turn in planned.dialogue))
        self.assertTrue(all(turn.text.strip() for turn in planned.dialogue))


if __name__ == "__main__":
    unittest.main()
