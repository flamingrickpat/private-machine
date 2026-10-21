"""Typed cycle state round trip (A19F step 2). No model calls."""
from __future__ import annotations

import unittest

from pydantic import ValidationError

from pm.records.codelet_records import CodeletRegistryState, CodeletRuntimeState
from pm.records.cycle_records import ActionDecision, ActionDispatchRef, BroadcastRef, StimulusGroup
from pm.records.cycle_state import CycleState
from pm.records.global_workspace import GWT_TURN_SCHEMA, GwtTurnRecord, LeftItem, ShardChange
from pm.records.mental_state import FullMentalState
from pm.records.qualia_records import QualeChannelState
from pm.records.situational_model import SituationalItem
from pm.records.workspace import ShardKey, WorkspaceItem

USER = ShardKey(kind="entity", ref="1")


def full_cycle() -> CycleState:
    state = FullMentalState()
    state.state_emotions.boredom = 0.4
    cycle = CycleState(tick_id=5, mental_state=state, control_generation=3,
                       qualia_channels={"affect.valence": QualeChannelState(channel_id="affect.valence", position=0.2)})
    cycle.csm.items[9] = SituationalItem(record_id=9, activation=0.7, endorsers=["retrieval"], shard_hint=USER)
    cycle.workspace.ensure_shard(USER).items.append(WorkspaceItem(record_id=9, entered_tick=5, required_by="memories"))
    cycle.codelets = CodeletRegistryState(codelets={"BoredomThresholdCross": CodeletRuntimeState(
        name="BoredomThresholdCross", activation=0.6, last_fired_tick=4, total_runs=2, last_percept_ids=[9])})
    cycle.broadcasts.append(BroadcastRef(tick_id=4, summary="He dodged again.", experience_id=8))
    cycle.stimulus_groups.append(StimulusGroup(key=USER, stimulus_ids=[2], correlation_id="c1"))
    cycle.decisions.append(ActionDecision(group=USER, text="Tease him.", variants=["Tease him."], stimulus_ids=[2]))
    cycle.dispatches.append(ActionDispatchRef(action_id=12, group=USER, correlation_id="c1", workflow="dialogue_story"))
    return cycle


class CycleStateTests(unittest.TestCase):
    def test_json_round_trip_keeps_typed_state(self):
        cycle = full_cycle()
        restored = CycleState.model_validate_json(cycle.model_dump_json())
        self.assertEqual(restored, cycle)
        self.assertAlmostEqual(restored.mental_state.state_emotions.boredom, 0.4)
        self.assertEqual(list(restored.workspace.entity_shards), [1])
        self.assertEqual(restored.qualia_channels["affect.valence"].position, 0.2)

    def test_legacy_fields_are_rejected(self):
        with self.assertRaises(ValidationError):
            CycleState.model_validate({"tick_id": 1, "mental_state_latent_vec": [0.0], "internal_data": {}})

    def test_fractional_relationship_entity_id_reads_as_no_entity(self):
        vector = FullMentalState().to_list()
        vector[600], vector[601] = 1.38, 0.5
        state = FullMentalState.init_from_list(vector)
        self.assertEqual(state.state_relationship.entity_id, 0)
        restored = CycleState.model_validate_json(CycleState(tick_id=1, mental_state=state).model_dump_json())
        self.assertAlmostEqual(restored.mental_state.state_relationship.affection, 0.5)


class TurnRecordTests(unittest.TestCase):
    def test_v2_turn_round_trips_through_yaml(self):
        turn = GwtTurnRecord(tick_id=5, knoxel_ids=[9], summary="s", experience_id=8, shard_changes=[
            ShardChange(key=USER, entered=[WorkspaceItem(record_id=9)], left=[LeftItem(record_id=3, reason="decay")],
                        kept_ids=[4], tail_ids=[2])])
        restored = GwtTurnRecord.from_yaml(turn.to_yaml())
        self.assertEqual(restored, turn)
        self.assertEqual(restored.schema_id, GWT_TURN_SCHEMA)

    def test_v1_turn_loads_with_empty_v2_fields(self):
        # A v1 turn names an attention frame. The key is ignored; old frames are not migrated.
        turn = GwtTurnRecord.model_validate({"schema_id": "pm.gwt-turn.v1", "tick_id": 2, "attention_frame_id": 1})
        self.assertEqual((turn.schema_id, turn.shard_changes, turn.experience_id, turn.stimulus_ids),
                         ("pm.gwt-turn.v1", [], None, []))

    def test_old_ledger_fields_do_not_load(self):
        # dev/tools/migrate_cycle_ledger_a19f.py removes them from saved data.
        with self.assertRaises(ValueError):
            CycleState.model_validate({"tick_id": 1, "published_slots": ["main"]})

    def test_claimed_ids_follow_the_stimulus_groups(self):
        state = CycleState(tick_id=3, stimulus_groups=[StimulusGroup(key=USER, stimulus_ids=[4, 5]),
                                                       StimulusGroup(stimulus_ids=[6])])
        self.assertEqual(state.claimed_ids(), [4, 5, 6])


if __name__ == "__main__":
    unittest.main()
