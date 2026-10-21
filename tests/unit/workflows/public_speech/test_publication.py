"""State, source, and publication boundaries. Live prose uses separate gates."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope
from workflows.main_cognition.schemas import ActionRequest, Stimulus
from workflows.public_speech.user_input import UserInputProxy
from pm.memory.snapshot import GhostConflict
from pm.memory.store import MemoryStore
from pm.records.cycle_state import PersonaConfig
from pm.records.memory_records import ActionSchemaRecord
from pm.records.enums import ActionType
from workflows.public_speech.publication import acknowledge_reply, commit_public_reply, exchange_record
from workflows.public_speech.reply_recovery import ReplyRecovery, task_evidence
from pm.runs.supervisor import ManagedRunSupervisor
from workflows.public_speech.public_context import build_public_context


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.ghost = MemoryStore(PersonaConfig(user_name="Jesse", companion_name="Spamuel"))
        self.bus = BusSystem()
        self.proxy = UserInputProxy("user_input", self.bus, self.fail, lambda: 0, self.ghost, lambda text: None)
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def supervisor(self, ghost):
        return ManagedRunSupervisor(ghost, runtime_root=Path(self.tmp.name), workflows={}, clock=lambda: 0)

    def test_stale_revision_cannot_publish_candidate(self):
        correlation = self.proxy.submit("Keep this original report.")
        source = exchange_record(self.ghost, correlation)
        context = build_public_context(self.ghost, ActionRequest(stimuli=[Stimulus(knoxel_id=source.id)]))
        changes = self.ghost.begin_batch("change-source")
        source.content = "The report changed before publication."
        changes.replace(source)
        self.ghost.commit(changes)
        with self.assertRaises(GhostConflict):
            commit_public_reply(self.ghost, "Unsupported stale candidate.", correlation, commit_id="stale", context=context)
        self.assertFalse(any(item.content == "Unsupported stale candidate." for item in self.ghost.all_features))
        self.assertEqual(exchange_record(self.ghost, correlation).metadata["exchange_status"], "pending")

    def test_new_input_does_not_cancel_frozen_reply(self):
        correlation = self.proxy.submit("First request.")
        context = build_public_context(self.ghost)
        self.proxy.submit("Second request.")
        reply_id = commit_public_reply(self.ghost, "Old reply.", correlation, commit_id="old", context=context)
        reply = self.ghost.get_knoxel_by_id(reply_id)
        self.assertEqual(reply.content, "Old reply.")
        self.assertEqual(reply.metadata["generation_user_frontier"], context.user_frontier)

    def test_source_conflict_prevents_semantic_and_protocol_recovery(self):
        from workflows.dialogue_story.host import StoryDialogue
        from pm.memory.bash_view import MemoryBashTarget
        from workflows.dialogue_story.standalone import MANIFEST

        for status in ("rejected", "failed"):
            with self.subTest(status=status), TemporaryDirectory() as tmp:
                correlation = self.proxy.submit("Keep this source unchanged.")
                source = exchange_record(self.ghost, correlation)
                module = StoryDialogue("dialogue", self.bus, self.fail, lambda: 0, None,
                    working_target=MemoryBashTarget(self.ghost, "read"),
                    workflow=MANIFEST, tools_factory=lambda environment, batch: {},
                    runtime_root=Path(tmp), ghost=self.ghost)
                module._public_context = build_public_context(self.ghost, ActionRequest(stimuli=[Stimulus(knoxel_id=source.id)]))
                module.invocation_id = "stale-" + status
                changes = self.ghost.begin_batch("changed-" + status)
                source.content = "Changed while the generation was running."
                changes.replace(source)
                self.ghost.commit(changes)
                with self.assertRaisesRegex(RuntimeError, "Invalid generation"):
                    module.reject_reply([Envelope(id="failure-" + status, source="operator", kind="data",
                        topic="action.request", correlation_id=correlation, payload={})], "Invalid generation.", status=status)
                self.assertEqual(exchange_record(self.ghost, correlation).metadata["exchange_status"], "pending")
                self.assertFalse(any(item.source == "Spamuel" for item in self.ghost.all_features))

    def test_protocol_failure_cannot_become_an_invented_user_report(self):
        from workflows.dialogue_story.host import StoryDialogue
        from pm.memory.bash_view import MemoryBashTarget
        from workflows.dialogue_story.standalone import MANIFEST

        with TemporaryDirectory() as tmp:
            correlation = self.proxy.submit("Reply after a transport error.")
            source = exchange_record(self.ghost, correlation)
            module = StoryDialogue("dialogue", self.bus, self.fail, lambda: 0, None,
                working_target=MemoryBashTarget(self.ghost, "read"), workflow=MANIFEST,
                tools_factory=lambda environment, batch: {}, runtime_root=Path(tmp), ghost=self.ghost)
            module._public_context = build_public_context(self.ghost, ActionRequest(stimuli=[Stimulus(knoxel_id=source.id)]))
            module.invocation_id = "protocol-failure"
            with self.assertRaisesRegex(RuntimeError, "HTTP 404"):
                module.reject_reply([Envelope(id="protocol-failure", source="operator", kind="data",
                    topic="action.request", correlation_id=correlation, payload={})],
                    "HTTP 404 from the original request.", status="failed")
            self.assertEqual(exchange_record(self.ghost, correlation).metadata["exchange_status"], "pending")
            self.assertEqual(len(self.ghost.all_features), 1)

    def test_resolution_and_replay_survive_checkpoint(self):
        with TemporaryDirectory() as tmp:
            self.ghost.checkpoint = Path(tmp) / "ghost.json"
            correlation = self.proxy.submit("One exchange.", session="original-session")
            reply_id = commit_public_reply(self.ghost, "One answer.", correlation, commit_id="reply")
            restored = MemoryStore.from_checkpoint(self.ghost.checkpoint)
            self.assertEqual(exchange_record(restored, correlation).metadata["public_reply_id"], reply_id)
            self.assertIsNone(commit_public_reply(restored, "Duplicate answer.", correlation, commit_id="duplicate"))
            recovery = ReplyRecovery("reply_recovery", self.bus, self.fail, lambda: 0, None,
                                     ghost=restored, supervisor=self.supervisor(restored))
            from tests.support import Recorder
            replies = Recorder(self.bus, "dialogue.reply")
            recovery.resume()
            self.assertEqual(replies.events[0].payload["knoxel_id"], reply_id)
            acknowledge_reply(restored, reply_id)
            recovery.resume()
            self.assertEqual(len(replies.events), 1)
            self.assertFalse(restored.get_knoxel_by_id(reply_id).metadata["factual_promotion"])

    def test_deception_requires_a_prior_purpose_and_retained_facts(self):
        correlation = self.proxy.submit("The factual codeword is copper lantern.")
        fact = exchange_record(self.ghost, correlation)
        changes = self.ghost.begin_batch("intent")
        action = changes.create(ActionSchemaRecord(content="A deliberate joke with a false codeword.", action_type=ActionType.Reply,
            metadata={"presentation": {"mode": "deceive", "purpose": "A declared teasing presentation.", "known_fact_ids": [fact.id]}}))
        self.ghost.commit(changes)
        context = build_public_context(self.ghost, ActionRequest(knoxel_id=action.id))
        reply_id = commit_public_reply(self.ghost, "Silver spoon.", correlation, commit_id="deceptive-reply", action_id=action.id, context=context)
        utterance = self.ghost.get_knoxel_by_id(reply_id)
        self.assertEqual(utterance.metadata["deception_intent_id"], action.id)
        self.assertEqual(utterance.metadata["known_fact_ids"], [fact.id])
        self.assertFalse(utterance.metadata["factual_promotion"])
        self.assertEqual(self.ghost.get_knoxel_by_id(fact.id).content, "The factual codeword is copper lantern.")

    def test_note_evidence_detects_changed_bytes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "outbox").mkdir()
            report = {"schema": "pm.note-verification.v1", "note_spec": {"path": "outbox/note.txt", "text": "copper lantern"}, "verified": True}
            (root / "outbox/report.yaml").write_text(yaml.safe_dump(report), encoding="utf-8")
            path = root / "outbox/note.txt"
            path.write_bytes(b"copper lantern")
            receipt = {"artifact_path": str(root), "state": "completed", "workflow_name": "note_task"}
            self.assertTrue(task_evidence(receipt)["task_verified"])
            path.write_bytes(b"copper lantern\n")
            self.assertFalse(task_evidence(receipt)["task_verified"])

    def test_restart_redispatches_pending_input_without_another_utterance(self):
        correlation = self.proxy.submit("A request interrupted before dispatch.")
        source = exchange_record(self.ghost, correlation)
        from tests.support import Recorder
        inputs = Recorder(self.bus, "user.input")
        recovery = ReplyRecovery("reply_recovery", self.bus, self.fail, lambda: 0, None,
                                 ghost=self.ghost, supervisor=self.supervisor(self.ghost))
        recovery.resume()
        self.assertEqual(inputs.events[0].correlation_id, correlation)
        self.assertEqual(inputs.events[0].payload["knoxel_id"], source.id)
        self.assertEqual(len(self.ghost.all_features), 1)

    def test_rejected_task_request_asks_for_agent_written_speech(self):
        correlation = self.proxy.submit("An unavailable task request.")
        recovery = ReplyRecovery("reply_recovery", self.bus, self.fail, lambda: 0, None,
                                 ghost=self.ghost, supervisor=self.supervisor(self.ghost))
        from tests.support import Recorder
        actions = Recorder(self.bus, "action.request")
        recovery.handle_batch([Envelope(id="rejected-task", source="self_management", kind="telemetry",
            topic="action.receipt", correlation_id=correlation, payload={"workflow_name": "self_management",
                "public_reply_requested": True, "status": "rejected", "summary": "The workflow is unavailable."})])
        source = exchange_record(self.ghost, correlation)
        self.assertEqual(source.metadata["exchange_status"], "pending")
        self.assertEqual(actions.events[0].target, "dialogue_assistant")
        self.assertIn("unavailable", actions.events[0].payload["task_receipt"]["summary"])
        self.assertEqual(len(self.ghost.all_features), 1)


if __name__ == "__main__":
    unittest.main()
