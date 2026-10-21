"""Dispatch recovery publishes the stored typed action request again.

No model. A real MemoryStore holds one pending action whose metadata holds
the request that main cognition published.
"""
from __future__ import annotations

import unittest

from pm.bus.bus import BusSystem
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ActionType
from pm.records.memory_records import ActionSchemaRecord
from pm.records.workspace import SELF_SHARD, ShardKey, WorkspaceShard
from tests.support import Recorder
from workflows.main_cognition.dispatch_recovery import replay_intents
from workflows.main_cognition.schemas import ActionRequest, RequestShard, Stimulus


class _Supervisor:
    """The two supervisor reads of ``replay_intents``: no run admitted the action."""

    def receipts(self) -> dict:
        return {}


class ReplayTests(unittest.TestCase):
    def test_a_pending_action_is_published_again_with_its_stored_request(self):
        ghost = MemoryStore(PersonaConfig(user_name="Uma", companion_name="Kit"))
        partner = ShardKey(kind="entity", ref="7")
        request = ActionRequest(
            action_id="act-1", workflow_name="dialogue_story", content="Tease her about the ending.",
            group=partner, partner_entity_id=7, partner_name="Ollie",
            stimuli=[Stimulus(knoxel_id=3, text="Everyone dies, but politely.")],
            shards=[RequestShard(name="self", shard=WorkspaceShard(key=SELF_SHARD)),
                    RequestShard(name="Ollie", shard=WorkspaceShard(key=partner, tail_ids=[3]))])
        action = ghost.store_knoxel(ActionSchemaRecord(
            content=request.content, action_type=ActionType.Reply,
            metadata={"action_id": "act-1", "dispatch_status": "pending", "dispatch_target": "dialogue_story",
                      "dispatch_correlation": "corr-1", "dispatch_payload": request.payload()}))
        bus = BusSystem()
        published = Recorder(bus, "action.request")
        handled = replay_intents(ghost, bus, _Supervisor())
        self.assertEqual(handled, {"corr-1"})
        self.assertEqual(len(published.events), 1)
        envelope = published.events[0]
        self.assertEqual((envelope.target, envelope.correlation_id), ("dialogue_story", "corr-1"))
        replayed = ActionRequest.model_validate(envelope.payload)
        self.assertEqual(replayed.knoxel_id, action.id)
        self.assertEqual(replayed.recovery_outcome, "intent_replayed")
        self.assertEqual(replayed.model_copy(update={"knoxel_id": None, "recovery_outcome": ""}), request)


if __name__ == "__main__":
    unittest.main()
