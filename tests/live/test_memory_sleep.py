"""Interrupt the real managed memory worker on a disposable character.

The local chat and embedding services in config.example.yaml must be running.
The fixture supplies two conversations separated by more than a session gap.
The accepted checkpoint must retain exactly the first complete conversation.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import time
import unittest
from uuid import uuid4

from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ActionType, ClusterType, EntityClass, FeatureType, InterlocusType
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord, Entity, Feature, MemoryClusterRecord
from pm.vclock import VirtualClock
from tests.live.configuration import live_model_config
from workflows.companion import build_app
from workflows.cognitive_control.schemas import WaitRequest
from workflows.main_cognition.schemas import ActionMetadata


class LiveMemorySleepTests(unittest.TestCase):
    def test_human_input_retains_one_complete_episode_from_a_running_worker(self) -> None:
        project = Path(__file__).resolve().parents[2]
        evidence = project / ".runtime" / "live-memory-sleep" / uuid4().hex
        evidence.mkdir(parents=True)
        config = live_model_config(project / "config.example.yaml", evidence)
        checkpoint = evidence / "ghost.pickle"
        ghost = MemoryStore(PersonaConfig(companion_name="Spamuel", user_name="Jesse"), checkpoint=checkpoint)
        companion = ghost.store_knoxel(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
        partner = ghost.store_knoxel(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
        day = datetime(2026, 11, 20)
        lighthouse = ("I put the lighthouse model on my desk.", "The blue lamp suits it.", "I named the lamp Meridian.")
        gardening = ("I painted the garden gate red.", "I can see it from the bridge.", "I planted bluebells beside it.")
        source_ids: list[list[int]] = []
        for index, conversation in enumerate((lighthouse, gardening)):
            scene: list[int] = []
            for turn, line in enumerate(conversation):
                stamp = day + timedelta(hours=16 + 3 * index, minutes=turn)
                speaker, listener = (companion, partner) if turn == 1 else (partner, companion)
                feature = ghost.store_knoxel(Feature(content=line, entity_id=speaker.id, addressee_entity_id=listener.id,
                    causal=True, feature_type=FeatureType.Dialogue, interlocus=InterlocusType.Public,
                    timestamp_world_begin=stamp, timestamp_world_end=stamp))
                scene.append(feature.id)
            source_ids.append(scene)
        ghost.accept_import()
        app = build_app(None, evidence / "runtime", ghost=ghost, clock=VirtualClock(day + timedelta(hours=22)),
                        config_path=str(config), cognition="real", enable_minecraft=False,
                        world_poll_seconds=None, record_lifecycle=False)
        control = app.cognitive_control
        assert control is not None
        try:
            batch = ghost.begin_batch("chosen-sleep", new_tick=True)
            action_data = ActionMetadata(action_id="sleep-decision")
            action = batch.create(ActionSchemaRecord(content="Sort through today's conversations while I rest.",
                action_type=ActionType.ToolCall, metadata=action_data.model_dump(exclude_unset=True)))  # pm-lint: allow PM007 -- persist typed action metadata
            batch.set_gwt(GwtTurnRecord(tick_id=batch.tick_id, knoxel_ids=[action.id]))
            control.commit(batch, WaitRequest(memory_action_id=action.id), control.generation)
            ghost.save()
            receipt = app.supervisor.start(action_id="sleep-decision", workflow_name="memory_reconsolidation",
                arguments={"reconsolidation_mode": "sleep", "_public_action_schema_id": action.id})  # pm-lint: allow PM005 -- managed action request wire fields
            deadline = time.monotonic() + 600
            while True:
                app.raise_if_error()
                with ghost.lock:
                    if app.memory_work.ready:
                        self.assertTrue(control.waiting())
                        break
                state = app.supervisor.status(receipt["run_id"])["state"]
                self.assertNotIn(state, {"completed", "cancelled", "failed", "interrupted"})
                if time.monotonic() >= deadline:
                    self.fail(f"No complete memory episode after ten minutes. Evidence at {evidence}.")
                time.sleep(.02)
            app.user_input.submit("Wake up, Spamuel!", delivery_id="wake")
            app.supervisor.join(receipt["run_id"], timeout=30)
            self.assertEqual(app.supervisor.status(receipt["run_id"])["state"], "interrupted")
            raw = MemoryStore.from_checkpoint(checkpoint, raw=True)
            self.assertFalse(any(row.content == "Wake up, Spamuel!" for row in raw.all_features))
            accepted = MemoryStore.from_checkpoint(checkpoint)
            topics = [row for row in accepted.all_knoxels.values()
                      if isinstance(row, MemoryClusterRecord) and row.cluster_type is ClusterType.Topical]
            self.assertEqual(len(topics), 1)
            self.assertEqual(topics[0].included_event_ids, ",".join(str(key) for key in source_ids[0]))
            self.assertFalse(control.waiting())
        finally:
            app.shutdown()
            app.join()
