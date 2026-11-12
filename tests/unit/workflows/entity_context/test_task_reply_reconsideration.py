"""Actual-store task-report recovery without executing or fabricating a task.

A cancelled receipt is explicit boundary input. These checks establish durable
reporting authority and reconsideration, not external task or model quality.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from typing import Literal

from pydantic import BaseModel, JsonValue, TypeAdapter

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.store import MemoryStore
from pm.memory.store_tools import build_tools
from pm.records.cycle_records import StimulusGroup
from pm.records.cycle_state import CycleState, PersonaConfig
from pm.records.enums import ActionType, EntityClass
from pm.records.global_workspace import GwtTurnRecord
from pm.records.memory_records import ActionSchemaRecord, Entity, Feature
from pm.runs.run_receipt import ManagedRunReceipt
from pm.runs.supervisor import ManagedRunSupervisor
from workflows.main_cognition.entity_updates import EntityActionMetadata
from workflows.main_cognition.host import MainCogLoop
from workflows.main_cognition.dispatch_recovery import replay_intents
from workflows.main_cognition.schemas import ActionRequest, PendingDispatch, Stimulus
from workflows.public_speech.publication import commit_public_reply, exchange_record
from workflows.public_speech.reply_recovery import ReplyRecovery
from workflows.public_speech.user_input import UserInputProxy


ROOT = Path(__file__).resolve().parents[4]


class CancelledEvidence(BaseModel):
    """Inspect the explicit cancellation fields of the existing JSON receipt."""

    state: Literal["cancelled"]
    task_verified: bool


class CapturedTaskReply(BaseModel):
    """Inspect typed authority without probing the legacy open receipt mapping."""

    action_origin: Literal["task_reply"]
    task_receipt: CancelledEvidence


class TaskReplyReconsiderationTests(unittest.TestCase):
    """Retain exact terminal evidence and one reconsideration identity on restart."""

    def main_host(self, store: MemoryStore, bus: BusSystem, runtime_root: Path) -> MainCogLoop:
        """Construct the production host without starting a generation worker."""
        return MainCogLoop("main_cog", bus, self.fail, time.monotonic, None,
            ghost=store, working_target=MemoryBashTarget(store),
            workflow=ROOT / "workflows/main_cognition/main_cognition.workflow.md",
            tools_factory=build_tools, runtime_root=runtime_root)

    def test_cancelled_task_reply_and_reconsideration_survive_clean_restart(self) -> None:
        with TemporaryDirectory() as directory:
            runtime_root = Path(directory)
            path = runtime_root / "ghost.pickle"
            store = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"), checkpoint=path)
            store.store_knoxel(Entity(content="Jesse", aliases=[], entity_class=EntityClass.Human))
            store.store_knoxel(Entity(content="Spamuel", aliases=[], entity_class=EntityClass.Self))
            store.accept_import()
            bus = BusSystem()
            events: list[Envelope] = []

            def collect(envelope: Envelope) -> None:
                events.append(envelope)

            bus.subscribe("task-report-evidence", lambda envelope: envelope.topic in {
                "action.request", "entity.action_reconsider"}, collect)
            proxy = UserInputProxy("user_input", bus, self.fail, time.monotonic, store, lambda text: None)
            correlation = proxy.submit("Please report the outcome of this selected task.")
            source = exchange_record(store, correlation)
            assert source is not None
            selected = store.begin_batch("selected-task-main", new_tick=True, embed=False)
            selected.depend_on([source.id])
            task = selected.create(ActionSchemaRecord(content="The selected external task.",
                action_type=ActionType.ToolCallAndReply,
                metadata={"action_id": "selected-task", "workflow_name": "note_task",
                          "dispatch_status": "admitted", "dispatch_correlation": correlation}))
            state = CycleState(tick_id=selected.tick_id,
                stimulus_groups=[StimulusGroup(stimulus_ids=[source.id])])
            selected.set_gwt(GwtTurnRecord(tick_id=selected.tick_id, stimulus_ids=[source.id],
                knoxel_ids=[source.id, task.id]), state)
            store.commit(selected)
            terminal = ManagedRunReceipt(run_id="cancelled-boundary-task", action_id="selected-task",
                workflow_name="note_task", parent_correlation=correlation,
                state="cancelled", latest_event="cancelled", artifact_path="", revision=1,
                authority_record_id=task.id, input_record_ids=[task.id],
                causal_commit_id="cancelled-task-receipt",
                arguments={"_public_reply_requested": True, "_public_action_schema_id": task.id})
            terminal_batch = store.begin_batch(terminal.causal_commit_id, embed=False)
            terminal_batch.depend_on([task.id], authority=task.id)
            receipt_json = TypeAdapter(ManagedRunReceipt).dump_json(terminal)
            receipt = TypeAdapter(dict[str, JsonValue]).validate_json(receipt_json)
            terminal_batch.set_run_receipt(terminal.run_id, receipt)
            store.commit(terminal_batch)
            supervisor = ManagedRunSupervisor(store, runtime_root=runtime_root, workflows={}, clock=time.monotonic)
            recovery = ReplyRecovery("reply_recovery", bus, self.fail, time.monotonic, None,
                                     ghost=store, supervisor=supervisor)
            recovery.reply_to_task(receipt)
            requests = [event for event in events if event.topic == "action.request"]
            self.assertEqual(len(requests), 1)
            request = ActionRequest.model_validate(requests[0].payload)
            self.assertEqual(request.action_origin, "task_reply")
            evidence = CapturedTaskReply.model_validate_json(request.model_dump_json())
            self.assertEqual(evidence.task_receipt.state, "cancelled")
            self.assertFalse(evidence.task_receipt.task_verified)
            assert request.knoxel_id is not None
            reply = store.snapshot({request.knoxel_id}).get(request.knoxel_id)
            retained = PendingDispatch.model_validate(reply.metadata).dispatch_payload
            self.assertEqual(retained, request)
            main = self.main_host(store, bus, runtime_root)
            main.defer_public_action(request)
            markers = [record for record in store.all_features if "entity_action_reconsideration" in record.metadata]
            self.assertEqual(len(markers), 1)
            marker = markers[0]
            self.assertEqual(EntityActionMetadata.model_validate(marker.metadata).entity_action_reconsideration.request, request)
            restored = MemoryStore.from_checkpoint(path)
            self.assertIsInstance(restored.snapshot({request.knoxel_id}).get(request.knoxel_id), ActionSchemaRecord)
            saved_marker = restored.snapshot({marker.id}).get(marker.id)
            self.assertIsInstance(saved_marker, Feature)
            saved = EntityActionMetadata.model_validate(saved_marker.metadata).entity_action_reconsideration
            self.assertEqual(saved.request.model_dump_json(), request.model_dump_json())
            self.assertEqual(saved.request.action_origin, "task_reply")
            self.assertEqual(saved.action_record_id, request.knoxel_id)
            self.assertEqual([key for cycle in restored.states for key in cycle.claimed_ids()].count(source.id), 1)
            restored_bus = BusSystem()
            replay: list[Envelope] = []

            def collect_replay(envelope: Envelope) -> None:
                replay.append(envelope)

            restored_bus.subscribe("reconsideration-evidence", lambda envelope: envelope.topic == "entity.action_reconsider", collect_replay)
            restored_main = self.main_host(restored, restored_bus, runtime_root)
            restored_main.resume_entity_actions()
            restored_main.defer_public_action(saved.request)
            self.assertEqual(len(replay), 1)
            self.assertEqual(replay[0].payload["knoxel_id"], marker.id)
            receipts = restored.read_run_receipts()
            self.assertEqual(receipts[terminal.run_id]["state"], "cancelled")
            self.assertEqual(len(receipts), 1)
            # An interim acknowledgement cannot erase the still-owed terminal
            # report when its replacement action is replayed after restart.
            commit_public_reply(restored, "I heard your request.", correlation,
                                commit_id="interim-answer", action_id=task.id)
            replacement_batch = restored.begin_batch("reconsidered-task-main", new_tick=True, embed=False)
            replacement_batch.depend_on([marker.id])
            replacement = replacement_batch.create(ActionSchemaRecord(content="Report the actual cancelled outcome.",
                action_type=ActionType.Reply,
                metadata={"action_id": "reconsidered-task-reply", "workflow_name": "dialogue_assistant",
                          "dispatch_status": "pending", "dispatch_target": "dialogue_assistant",
                          "dispatch_correlation": correlation}))
            replacement_request = saved.request.model_copy(update={"knoxel_id": replacement.id,
                "action_id": "reconsidered-task-reply", "gwt_tick": replacement_batch.tick_id,
                "stimuli": [Stimulus(knoxel_id=marker.id, text=marker.content,
                                     source="entity.action_reconsider", entity_id=request.partner_entity_id)]})
            replacement.metadata["dispatch_payload"] = replacement_request.payload()
            replacement_state = CycleState(tick_id=replacement_batch.tick_id,
                stimulus_groups=[StimulusGroup(stimulus_ids=[marker.id])])
            replacement_batch.set_gwt(GwtTurnRecord(tick_id=replacement_batch.tick_id,
                stimulus_ids=[marker.id], knoxel_ids=[marker.id, replacement.id]), replacement_state)
            restored.commit(replacement_batch)
            final_store = MemoryStore.from_checkpoint(path)
            final_bus = BusSystem()
            reports: list[Envelope] = []

            def collect_report(envelope: Envelope) -> None:
                reports.append(envelope)

            final_bus.subscribe("terminal-replay-evidence", lambda envelope: envelope.topic == "action.request", collect_report)
            final_supervisor = ManagedRunSupervisor(final_store, runtime_root=runtime_root, workflows={}, clock=time.monotonic)
            replay_intents(final_store, final_bus, final_supervisor)
            self.assertEqual(len(reports), 1)
            replayed = ActionRequest.model_validate(reports[0].payload)
            self.assertEqual(replayed.knoxel_id, replacement.id)
            self.assertEqual(replayed.action_origin, "task_reply")
            self.assertEqual(replayed.gwt_tick, replacement_batch.tick_id)
            self.assertFalse(CapturedTaskReply.model_validate_json(replayed.model_dump_json()).task_receipt.task_verified)


if __name__ == "__main__":
    unittest.main()
