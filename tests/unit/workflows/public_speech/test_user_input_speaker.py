"""Give each human speaker of the dialogue input their own entity and exchange."""
from __future__ import annotations

import unittest

from pm.bus.bus import BusSystem
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import EntityClass, SourceKind
from pm.records.workspace import SELF_SHARD, ShardKey
from workflows.main_cognition.cycle_work import CycleRequest
from workflows.main_cognition.phases.intake import group_key
from workflows.main_cognition.schemas import Stimulus
from workflows.public_speech.publication import commit_public_reply, exchange_record
from workflows.public_speech.user_input import UserInputProxy


class SpeakerTests(unittest.TestCase):
    def setUp(self):
        self.ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Sunny"))
        self.proxy = UserInputProxy("user_input", BusSystem(), self.fail, lambda: 0, self.ghost, print)

    def test_the_default_speaker_is_the_user(self):
        for speaker in (None, "Jesse"):
            record = exchange_record(self.ghost, self.proxy.submit("hi", speaker=speaker))
            self.assertEqual((record.source, record.entity_id), ("Jesse", 0))
            self.assertEqual(record.provenance.source_kind, SourceKind.User)
        self.assertEqual(self.ghost.all_entities, [])

    def test_another_name_gets_one_entity_and_its_own_group(self):
        first = exchange_record(self.ghost, self.proxy.submit("hello", speaker="Alex"))
        second = exchange_record(self.ghost, self.proxy.submit("again", speaker="Alex"))
        entity_id = self.ghost.get_entity_id("Alex")
        self.assertIsNotNone(entity_id)
        self.assertEqual([entity.entity_class for entity in self.ghost.all_entities], [EntityClass.Human])
        self.assertEqual((first.entity_id, second.entity_id), (entity_id, entity_id))
        self.assertEqual((first.source, first.metadata["exchange_status"]), ("Alex", "pending"))
        # Main cognition sets the stimulus entity from the record; that entity owns the group.
        stimulus = Stimulus(knoxel_id=first.id, source="user_input", kind="user.input", entity_id=first.entity_id)
        self.assertEqual(group_key(stimulus, CycleRequest(user_entity_id=None)),
                         ShardKey(kind="entity", ref=str(entity_id)))
        user = Stimulus(knoxel_id=1, source="user_input", kind="user.input")
        self.assertEqual(group_key(user, CycleRequest(user_entity_id=None)), SELF_SHARD)

    def test_a_reply_resolves_the_exchange_of_another_speaker(self):
        correlation = self.proxy.submit("hello", speaker="Alex")
        commit_public_reply(self.ghost, "hi Alex", correlation, commit_id="reply",
                            addressee_entity_id=self.ghost.get_entity_id("Alex"))
        self.assertEqual(exchange_record(self.ghost, correlation).metadata["exchange_status"], "resolved")


if __name__ == "__main__":
    unittest.main()
