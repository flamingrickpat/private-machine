"""Deterministic checks for the workflow routing boundary."""
from __future__ import annotations

import unittest
from time import time
from pathlib import Path
from tempfile import TemporaryDirectory

from pm.bus.bus import BusSystem
from pm.bus.envelopes import new_envelope
from pm.host.application import Application
from pm.memory.bash_view import MemoryBashTarget
from pm.memory.store import MemoryStore
from pm.records.base import Provenance
from pm.records.cycle_state import PersonaConfig
from pm.records.enums import ActionType, FeatureType, SourceKind
from pm.records.memory_records import ActionSchemaRecord, Feature
from pm.runs.supervisor import ManagedRunSupervisor
from pm.runs.workflow_runner import PROJECT_ROOT
from pm.runs.workflow_spec import WorkflowSpec
from workflows.dialogue_assistant.host import AssistantDialogue
from workflows.dialogue_assistant.schemas import AssistantRequest
from workflows.main_cognition.cycle_work import CycleRequest
from workflows.main_cognition.phases.intake import group_stimuli
from workflows.main_cognition.phases.route import body_lease
from workflows.main_cognition.schemas import ActionArguments, ActionDispatch, PendingDispatch, Stimulus, WorkflowOption
from workflows.public_speech.publication import commit_public_reply
from workflows.public_speech.reply_recovery import ReplyRecovery


class ManagedRoutingTests(unittest.TestCase):
    def test_internal_marker_keeps_the_latest_real_exchange(self) -> None:
        request = CycleRequest(stimuli=[
            Stimulus(knoxel_id=1, entity_id=20, correlation_id="earlier"),
            Stimulus(knoxel_id=2, entity_id=20, correlation_id="latest"),
            Stimulus(knoxel_id=3, entity_id=20, kind="entity.action_reconsider"),
        ])
        [group] = group_stimuli(request)
        self.assertEqual(group.stimulus_ids, [1, 2, 3])
        self.assertEqual(group.correlation_id, "latest")

    def test_custom_workflow_uses_registered_resource_for_start_only(self) -> None:
        options = [WorkflowOption(name="writing", exclusive_resource="editor")]
        start = ActionDispatch(workflow_name="writing", action_type="ToolCallAndReply")
        update = start.model_copy(update={"arguments": ActionArguments(operation="update", run_id="active-run")})
        self.assertEqual(body_lease(start, options), "editor")
        self.assertEqual(body_lease(update, options), "")

    def test_real_run_retains_separate_admission_and_terminal_reply_actions(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
            batch = ghost.begin_batch("request", embed=False)
            batch.create(Feature(content="Start the local worker.", feature_type=FeatureType.Dialogue,
                provenance=Provenance(source_kind=SourceKind.User, source_ref="Jesse"),
                metadata={"exchange_correlation": "exchange", "exchange_status": "pending"}))
            action = batch.create(ActionSchemaRecord(content="Run the local worker.",
                action_type=ActionType.ToolCallAndReply, metadata={"action_id": "worker-start"}))
            ghost.commit(batch)
            supervisor = ManagedRunSupervisor(ghost, runtime_root=root / "runtime", task_root=root / "tasks",
                workflows={"local_task": WorkflowSpec(
                    PROJECT_ROOT / "workflows/local_task/local_task.workflow.md", routine_start=True)}, clock=time)
            recovery = ReplyRecovery("reply_recovery", Application(BusSystem(), []), None,
                                     ghost=ghost, supervisor=supervisor)
            accepted = supervisor.start(action_id="worker-start", workflow_name="local_task",
                parent_correlation="exchange", instructions="Run the local worker.",
                arguments={"wait_for_cancel": True, "_public_reply_requested": True,
                           "_public_action_schema_id": action.id})
            run_id = accepted["run_id"]
            try:
                start = new_envelope("self_management", "telemetry", "action.receipt", "exchange",
                    {"workflow_name": "self_management", "operation": "start", "status": "accepted",
                     "public_reply_requested": True, "receipt": accepted})
                recovery.handle_batch([start, start])
            finally:
                supervisor.terminate(run_id)
                supervisor.join_all(timeout=10)
            recovery.reply_to_task(supervisor.status(run_id))
            replies = [record for record in ghost.all_knoxels.values()
                       if isinstance(record, ActionSchemaRecord) and record.action_type == ActionType.Reply]
            self.assertEqual(len(replies), 2)
            self.assertEqual([record.metadata["task_run_id"] for record in replies], [run_id, run_id])
            self.assertTrue(replies[0].metadata["action_id"].startswith("task-admission:"))
            self.assertTrue(replies[1].metadata["action_id"].startswith("task-reply:"))
            admission = PendingDispatch.model_validate(replies[0].metadata)
            terminal = PendingDispatch.model_validate(replies[1].metadata)
            assert admission.dispatch_payload.task_receipt is not None
            self.assertTrue(admission.dispatch_payload.task_receipt.admission_only)
            acknowledgment_id = commit_public_reply(ghost, "Worker accepted.", "exchange",
                commit_id="test-acknowledgment", action_id=replies[0].id, action_origin="task_reply")
            self.assertIsNotNone(acknowledgment_id)
            assistant = AssistantDialogue("assistant", Application(BusSystem(), []), None,
                working_target=MemoryBashTarget(ghost, "read"),
                workflow=PROJECT_ROOT / "workflows/dialogue_assistant/dialogue_assistant.workflow.md",
                runtime_root=root / "runtime", ghost=ghost, tools_factory=None)
            envelope = new_envelope("reply_recovery", "data", "action.request", "exchange",
                                    terminal.dispatch_payload.payload(), target="dialogue_assistant")
            self.assertEqual(assistant.stale_reply_reason(envelope, terminal.dispatch_payload), "")
            followup = ghost.begin_batch("test-followup", embed=False)
            followup.create(Feature(content="What else is on your mind?", feature_type=FeatureType.Dialogue,
                override_source="Jesse", provenance=Provenance(source_kind=SourceKind.User, source_ref="Jesse"),
                metadata={"exchange_correlation": "new-exchange", "exchange_status": "pending"}))  # pm-lint: allow PM005 -- serialized dialogue record metadata
            ghost.commit(followup)
            self.assertEqual(assistant.late_partner_lines(terminal.dispatch_payload), [])
            reply_input = AssistantRequest.model_validate(assistant.prepare_input([envelope]))
            self.assertTrue(reply_input.task_result_supplied)
            self.assertFalse(reply_input.tools)
            terminal_id = commit_public_reply(ghost, "Worker stopped.", "exchange",
                commit_id="test-terminal", action_id=replies[1].id, action_origin="task_reply")
            self.assertIsNotNone(terminal_id)
            self.assertIsNone(commit_public_reply(ghost, "Worker stopped.", "exchange",
                commit_id="test-terminal-replay", action_id=replies[1].id, action_origin="task_reply"))
