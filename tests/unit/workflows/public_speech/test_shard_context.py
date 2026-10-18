"""The reply context comes from the workspace shards of the choosing cycle.

No model. The tests build real records in a real MemoryStore and a typed
action request with a self shard and a partner shard.
"""
from __future__ import annotations

import unittest

from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import EntityClass, FeatureType
from pm.records.memory_records import Entity, new_feature
from pm.records.workspace import SELF_SHARD, ShardKey, WorkspaceItem, WorkspaceShard
from workflows.main_cognition.schemas import ActionRequest, RequestShard, Stimulus
from workflows.public_speech.public_context import build_public_context
from workflows.public_speech.shard_context import shard_context, shard_story


class ShardContextTests(unittest.TestCase):
    def setUp(self):
        self.ghost = MemoryStore(PersonaConfig(user_name="Uma", companion_name="Kit"))
        self.other = self.ghost.store_knoxel(Entity(content="Ollie", aliases=[], entity_class=EntityClass.AI))
        say = lambda text, source, kind=FeatureType.Dialogue: new_feature(self.ghost, text, kind, source=source)
        self.line = say("Can I pitch a third ending?", "Ollie")
        self.reply = say("Only if it is a good one.", "Kit")
        self.thought = say("Ollie always wants the last word.", "Kit", FeatureType.Thought)
        self.memory = say("Ollie once rewrote a whole chapter.", "Ollie", FeatureType.WorldEvent)
        self.current = say("Everyone dies, but politely.", "Ollie")
        partner = ShardKey(kind="entity", ref=str(self.other.id))
        self.request = ActionRequest(
            content="Tell Ollie that the polite deaths are lazy.", group=partner,
            partner_entity_id=self.other.id, partner_name="Ollie",
            stimuli=[Stimulus(knoxel_id=self.current.id, text=self.current.content, entity_id=self.other.id)],
            shards=[RequestShard(name="self", shard=WorkspaceShard(key=SELF_SHARD, tail_ids=[self.thought.id])),
                    RequestShard(name="Ollie", shard=WorkspaceShard(
                        key=partner, items=[WorkspaceItem(record_id=self.memory.id)],
                        tail_ids=[self.line.id, self.reply.id]))])

    def test_the_transcript_is_the_partner_tail_and_the_private_thoughts(self):
        planned = shard_context(self.ghost.snapshot(), self.request, "Kit")
        self.assertEqual([(turn.speaker, turn.kind) for turn in planned.dialogue],
                         [("Ollie", "speech"), ("Kit", "speech"), ("Kit", "thought")])
        self.assertNotIn(self.current.id, [turn.knoxel_id for turn in planned.dialogue])

    def test_the_story_renders_the_shard_items_without_the_verbatim_lines(self):
        planned = shard_context(self.ghost.snapshot(), self.request, "Kit")
        self.assertIn("Kit AND Ollie", planned.story)
        self.assertIn("in the story transcript", planned.story)
        self.assertIn("rewrote a whole chapter", planned.story)
        self.assertNotIn("third ending", planned.story)
        self.assertNotIn("[id", planned.story)
        self.assertIn("third ending", shard_story(self.ghost.snapshot(), self.request, "Kit"))

    def test_the_public_context_names_the_partner_and_keeps_the_shard_evidence(self):
        planned = shard_context(self.ghost.snapshot(), self.request, "Kit")
        context = build_public_context(self.ghost, self.request, planned=planned)
        self.assertEqual((context.partner_name, context.user_name), ("Ollie", "Uma"))
        lanes = {item.knoxel_id: item.lane for item in context.evidence}
        self.assertEqual(lanes[self.memory.id], "shard:Ollie")
        self.assertEqual(lanes[self.thought.id], "shard:self")
        self.assertIn(self.current.id, lanes)
        self.assertEqual(context.shard_story, planned.story)


if __name__ == "__main__":
    unittest.main()
