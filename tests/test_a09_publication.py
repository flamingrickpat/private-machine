"""State, source, and publication boundaries. Live prose uses separate gates."""
from __future__ import annotations

from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest

import yaml

from pm.bus.bus import BusSystem
from pm.bus.envelopes import Envelope
from modules import UserInputProxy
from pm.memory.snapshot import GhostConflict
from pm.memory.store import Ghost
from pm.records.cycle_state import GhostConfig
from pm.records.memory_records import ActionSchemaKnoxel
from pm.records.enums import ActionType
from runtime.publication import acknowledge_reply, commit_public_reply, exchange_record
from runtime.reply_recovery import ReplyRecovery, task_evidence
from pm.runs.supervisor import ManagedRunSupervisor
from workflows.public_context import build_public_context
from workflows.public_claims import ClaimReview, validate_claim_review


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.ghost = Ghost(GhostConfig(user_name="Jesse", companion_name="Spamuel"))
        self.bus = BusSystem()
        self.proxy = UserInputProxy("user_input", self.bus, self.fail, lambda: 0, self.ghost, lambda text: None)
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def supervisor(self, ghost):
        return ManagedRunSupervisor(ghost, runtime_root=Path(self.tmp.name), workflows={}, clock=lambda: 0)

    def test_stale_revision_cannot_publish_candidate(self):
        correlation = self.proxy.submit("Keep this original report.")
        source = exchange_record(self.ghost, correlation)
        context = build_public_context(self.ghost, {"stimuli": [{"knoxel_id": source.id}]})
        changes = self.ghost.begin_batch("change-source")
        source.content = "The report changed before publication."
        changes.replace(source)
        self.ghost.commit(changes)
        with self.assertRaises(GhostConflict):
            commit_public_reply(self.ghost, "Unsupported stale candidate.", correlation, commit_id="stale", context=context)
        self.assertFalse(any(item.content == "Unsupported stale candidate." for item in self.ghost.all_features))
        self.assertEqual(exchange_record(self.ghost, correlation).metadata["exchange_status"], "pending")

    def test_new_input_supersedes_frozen_reply(self):
        correlation = self.proxy.submit("First request.")
        context = build_public_context(self.ghost, {})
        self.proxy.submit("Second request.")
        with self.assertRaises(GhostConflict):
            commit_public_reply(self.ghost, "Old reply.", correlation, commit_id="old", context=context)

    def test_source_conflict_prevents_semantic_and_protocol_recovery(self):
        from modules import StoryDialogue
        from pm.memory.bash_view import GhostBashWrapper
        from workflows.dialogue_story.engine import MANIFEST

        for status in ("rejected", "failed"):
            with self.subTest(status=status), TemporaryDirectory() as tmp:
                correlation = self.proxy.submit("Keep this source unchanged.")
                source = exchange_record(self.ghost, correlation)
                module = StoryDialogue("dialogue", self.bus, self.fail, lambda: 0, None,
                    working_target=GhostBashWrapper(self.ghost, "read"),
                    workflow=MANIFEST, tools_factory=lambda environment, batch: {},
                    runtime_root=Path(tmp), ghost=self.ghost)
                module._public_context = build_public_context(self.ghost, {"stimuli": [{"knoxel_id": source.id}]})
                module.invocation_id = "stale-" + status
                module.grounding_recovery = lambda context: self.fail("Stale evidence reached model recovery.")
                changes = self.ghost.begin_batch("changed-" + status)
                source.content = "Changed while the generation was running."
                changes.replace(source)
                self.ghost.commit(changes)
                module.reject_reply([Envelope(id="failure-" + status, source="operator", kind="data",
                    topic="action.request", correlation_id=correlation, payload={})], "Invalid generation.", status=status)
                exchange = exchange_record(self.ghost, correlation)
                reply = self.ghost.get_knoxel_by_id(exchange.metadata["public_reply_id"])
                self.assertEqual(reply.metadata["recovery_outcome"], "stale")
                self.assertEqual(exchange.metadata["exchange_status"], "resolved")

    def test_protocol_failure_cannot_become_an_invented_user_report(self):
        from modules import StoryDialogue
        from pm.memory.bash_view import GhostBashWrapper
        from workflows.dialogue_story.engine import MANIFEST

        with TemporaryDirectory() as tmp:
            correlation = self.proxy.submit("Reply after a transport error.")
            source = exchange_record(self.ghost, correlation)
            module = StoryDialogue("dialogue", self.bus, self.fail, lambda: 0, None,
                working_target=GhostBashWrapper(self.ghost, "read"), workflow=MANIFEST,
                tools_factory=lambda environment, batch: {}, runtime_root=Path(tmp), ghost=self.ghost)
            module._public_context = build_public_context(self.ghost, {"stimuli": [{"knoxel_id": source.id}]})
            module.invocation_id = "protocol-failure"
            module.grounding_recovery = lambda context: self.fail("Protocol failure reached semantic recovery.")
            module.reject_reply([Envelope(id="protocol-failure", source="operator", kind="data",
                topic="action.request", correlation_id=correlation, payload={})],
                "HTTP 404 from the original request.", status="failed")
            exchange = exchange_record(self.ghost, correlation)
            reply = self.ghost.get_knoxel_by_id(exchange.metadata["public_reply_id"])
            self.assertEqual(exchange.metadata["exchange_status"], "resolved")
            self.assertEqual(reply.metadata["recovery_outcome"], "failed")
            self.assertFalse(reply.metadata["factual_promotion"])
            self.assertNotIn("reported", reply.content)

    def test_resolution_and_replay_survive_checkpoint(self):
        with TemporaryDirectory() as tmp:
            self.ghost.checkpoint = Path(tmp) / "ghost.json"
            correlation = self.proxy.submit("One exchange.", session="original-session")
            reply_id = commit_public_reply(self.ghost, "One answer.", correlation, commit_id="reply")
            restored = Ghost.from_checkpoint(self.ghost.checkpoint)
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
        action = changes.create(ActionSchemaKnoxel(content="A deliberate joke with a false codeword.", action_type=ActionType.Reply,
            metadata={"presentation": {"mode": "deceive", "purpose": "A declared teasing presentation.", "known_fact_ids": [fact.id]}}))
        self.ghost.commit(changes)
        context = build_public_context(self.ghost, {"knoxel_id": action.id})
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

    def test_rejected_task_request_closes_its_exchange(self):
        correlation = self.proxy.submit("An unavailable task request.")
        recovery = ReplyRecovery("reply_recovery", self.bus, self.fail, lambda: 0, None,
                                 ghost=self.ghost, supervisor=self.supervisor(self.ghost))
        recovery.handle_batch([Envelope(id="rejected-task", source="self_management", kind="telemetry",
            topic="action.receipt", correlation_id=correlation, payload={"workflow_name": "self_management",
                "public_reply_requested": True, "status": "rejected", "summary": "The workflow is unavailable."})])
        source = exchange_record(self.ghost, correlation)
        self.assertEqual(source.metadata["exchange_status"], "resolved")
        self.assertIn("unavailable", self.ghost.get_knoxel_by_id(source.metadata["public_reply_id"]).content)

    def test_captured_review_cannot_promote_companion_speech_to_an_event(self):
        capture = json.loads((Path(__file__).resolve().parents[1] / "test_data/captures/alpha/a09-companion-claim.json").read_text(encoding="utf-8"))
        context = build_public_context(self.ghost, {})
        context.evidence = [capture["evidence"]]
        review = ClaimReview.model_validate({"accepted": capture["accepted"], "claims": [capture["claim"]]})
        result = validate_claim_review(review, capture["claim"]["text"], context)
        self.assertFalse(result.accepted)
        self.assertEqual(result.claims[0].status, "unsupported")

    def test_captured_review_cannot_verify_unsourced_technical_advice(self):
        capture = json.loads((Path(__file__).resolve().parents[1] / "test_data/captures/alpha/a09-unsourced-claim.json").read_text(encoding="utf-8"))
        context = build_public_context(self.ghost, {})
        review = ClaimReview.model_validate({"accepted": capture["accepted"], "claims": [capture["claim"]]})
        result = validate_claim_review(review, capture["claim"]["text"], context)
        self.assertFalse(result.accepted)
        self.assertEqual(result.claims[0].status, "unsupported")

    def test_captured_review_cannot_invent_a_missing_report_as_knowledge_state(self):
        capture = json.loads((Path(__file__).resolve().parents[1] / "test_data/captures/alpha/a09-missing-report.json").read_text(encoding="utf-8"))
        context = build_public_context(self.ghost, {})
        review = ClaimReview.model_validate(capture["review"])
        result = validate_claim_review(review, capture["candidate"], context)
        self.assertFalse(result.accepted)
        self.assertEqual(result.claims[0].status, "unsupported")


if __name__ == "__main__":
    unittest.main()
