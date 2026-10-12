"""Resume selected task replies from durable run and exchange evidence."""
from __future__ import annotations

from pathlib import Path
import json

import yaml

from pm.bus.envelopes import Envelope, new_id
from pm.records.memory_records import ActionSchemaRecord
from pm.records.enums import ActionType
from runtime.publication import commit_public_reply, exchange_record, publish_reply
from pm.host.application import BusModule

TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})


def task_evidence(receipt: dict) -> dict:
    """Verify note bytes or retained Minecraft receipts before public dispatch."""
    result = dict(receipt)
    if not receipt.get("artifact_path"):
        result["task_verified"] = False
        return result
    root = Path(receipt["artifact_path"]).resolve()
    report_path = root / "outbox" / "report.yaml"
    if report_path.is_file():
        report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
        if isinstance(report, dict):
            result["report"] = report
            if report.get("schema") == "pm.minecraft-project-report.v1" and report.get("verified") is True:
                from workflows.minecraft_project.acceptance import verified_project
                verification = verified_project(root)
                result["minecraft_verification"] = verification
                result["task_verified"] = receipt.get("state") == "completed" and verification["verified"]
            if report.get("schema") == "pm.note-verification.v1" and report.get("verified") is True:
                path = (root / report["note_spec"]["path"]).resolve()
                if not path.is_relative_to(root / "outbox"):
                    raise ValueError("The note evidence escaped its task outbox.")
                actual = path.read_bytes() if path.is_file() else None
                expected = report["note_spec"]["text"].encode("utf-8")
                result["task_verified"] = actual == expected and report.get("verified") is True
                result["actual_hex"] = actual.hex() if actual is not None else None
            elif (report.get("schema") == "pm.minecraft-actor-report.v1" and report.get("verified") is True
                  and report.get("acceptance", {}).get("kind") == "wood_return"):
                from workflows.minecraft.specialist_session import verify_wood_return
                journal = (root / report["journal"]).resolve()
                if not journal.is_relative_to(root / "scratch"):
                    raise ValueError("The Minecraft journal escaped its task scratch directory.")
                rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
                calls = [row for row in rows if row["event"] == "result"]
                if (calls and calls[0]["tool"] == calls[-1]["tool"] == "minecraft_observe"
                    and calls[0]["ok"] and calls[-1]["ok"]):
                    verification = verify_wood_return(calls, calls[0]["result"], calls[-1]["result"], report["acceptance"])
                    result["minecraft_verification"] = verification
                    result["task_verified"] = receipt.get("state") == "completed" and verification["verified"]
    # The control worker reports only its own artifacts and instruction.
    # Its completed state never establishes arbitrary note execution.
    result.setdefault("task_verified", False)
    return result


class ReplyRecovery(BusModule):
    """Dispatch a terminal reply only when the prior action selected public speech."""

    DATA_TOPICS = frozenset({"workflow.event", "workflow.question", "action.receipt", "public.resume"})

    def __init__(self, *args, ghost, supervisor, **kwargs):
        super().__init__(*args, **kwargs)
        self.ghost = ghost
        self.supervisor = supervisor

    def handle_batch(self, batch):
        for envelope in batch:
            if envelope.topic in {"workflow.event", "workflow.question"} and envelope.payload.get("event_id"):
                self.supervisor.acknowledge_event(envelope.payload["run_id"], envelope.payload["event_id"])
            if envelope.topic == "public.resume":
                self.resume()
            elif envelope.topic == "workflow.event" and envelope.payload.get("event") in TERMINAL:
                self.reply_to_task(self.supervisor.status(envelope.payload["run_id"]))
            elif (envelope.topic == "action.receipt" and envelope.payload.get("workflow_name") == "self_management"
                  and envelope.payload.get("public_reply_requested")):
                receipt = envelope.payload.get("receipt", {})
                if envelope.payload.get("operation") not in {None, "start"}:
                    operation = {**receipt, "run_id": receipt.get("run_id", "operation:" + envelope.id),
                                 "revision": receipt.get("revision", 0), "parent_correlation": envelope.correlation_id,
                                 "state": receipt.get("state", envelope.payload["status"]),
                                 "workflow_name": receipt.get("workflow_name", "self_management"),
                                 "artifact_path": receipt.get("artifact_path", ""),
                                 "operation_result": envelope.payload,
                                 "arguments": {"_public_reply_requested": True,
                                               "_public_action_schema_id": envelope.payload.get("action_schema_id")}}
                    self.reply_to_task(operation)
                elif receipt.get("state") in TERMINAL:
                    self.reply_to_task(receipt)
                elif envelope.payload.get("status") == "rejected":
                    self.fail_exchange(envelope.correlation_id, "The task request was rejected. "
                                       + envelope.payload.get("summary", "No task execution was verified."))

    def fail_exchange(self, correlation: str, text: str):
        """Close a failed task exchange without asserting unverified effects."""
        reply_id = commit_public_reply(self.ghost, text, correlation,
                                      commit_id=f"task-recovery:{correlation}", recovery="task_evidence_failed")
        if reply_id is not None:
            publish_reply(self.bus, self.module_id, correlation, reply_id, recovery_outcome="task_evidence_failed")

    def reply_to_task(self, receipt):
        """Retain the original selected action as the cause of a receipt-based reply."""
        correlation = receipt.get("parent_correlation", "")
        arguments = receipt.get("arguments", {})
        source_id = arguments.get("_public_action_schema_id")
        source = self.ghost.get_knoxel_by_id(source_id)
        exchange = exchange_record(self.ghost, correlation)
        if (exchange is None or exchange.metadata.get("exchange_status") != "pending"
                or not arguments.get("_public_reply_requested")):
            return
        if not isinstance(source, ActionSchemaRecord) or source.action_type != ActionType.ToolCallAndReply:
            self.fail_exchange(correlation, f"The {receipt['workflow_name']} worker is {receipt['state']}. "
                               "Its selected reply intent is unavailable; no further task effect was verified.")
            return
        try:
            evidence = task_evidence(receipt)
        except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
            self.fail_exchange(correlation, f"The {receipt['workflow_name']} worker is {receipt['state']}. "
                               f"Its task evidence could not be verified ({type(error).__name__}).")
            return
        identity = f"task-reply:{receipt['run_id']}:{receipt['revision']}:{source.id}"
        with self.ghost.lock:
            prior = next((item for item in self.ghost.all_knoxels.values()
                          if isinstance(item, ActionSchemaRecord) and item.metadata.get("action_id") == identity), None)
            if prior is None:
                changes = self.ghost.begin_batch(identity)
                action = changes.create(ActionSchemaRecord(
                    action_type=ActionType.Reply,
                    content="Answer the original request using this terminal task receipt. Explain verified work, failure, or uncertainty accurately.",
                    attention_frame_id=source.attention_frame_id, intention_ids=list(source.intention_ids),
                    metadata={**source.metadata, "action_id": identity, "workflow_name": "dialogue_assistant",
                              "source_action_schema_id": source.id, "task_run_id": receipt["run_id"]},
                ))
                accepted = self.ghost.commit(changes)
                action_id = accepted.added_ids[0]
            else:
                action_id = prior.id
        self.bus.put(Envelope(
            id=new_id(), source=self.module_id, kind="data", topic="action.request", correlation_id=correlation,
            target="dialogue_assistant", payload={"knoxel_id": action_id, "action_id": identity,
                "task_receipt": evidence, "stimuli": [{"knoxel_id": exchange.id, "text": exchange.content}],
                "workspace": "The selected task reached a terminal boundary. Its receipt does not imply unverified instruction execution."},
        ))

    def resume(self):
        """Replay pending publication and task replies after a checkpoint restart."""
        from runtime.intents import replay_intents
        handled = replay_intents(self.ghost, self.bus, self.supervisor)
        self.supervisor.replay_events()
        for item in self.ghost.all_features:
            if item.metadata.get("publication_status") == "pending":
                publish_reply(self.bus, self.module_id, item.metadata.get("reply_correlation", ""), item.id,
                              recovery_outcome="replayed")
        receipts = self.supervisor.receipts()
        for receipt in receipts.values():
            if receipt["state"] in TERMINAL:
                self.reply_to_task(receipt)
        waiting = {receipt.get("parent_correlation") for receipt in receipts.values()
                   if receipt.get("arguments", {}).get("_public_reply_requested")}
        waiting.update(handled)
        for item in self.ghost.all_features:
            correlation = item.metadata.get("exchange_correlation")
            if (correlation and item.metadata.get("exchange_status") == "pending" and correlation not in waiting):
                self.bus.put(Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                                     correlation_id=correlation, payload={"knoxel_id": item.id, "recovery_outcome": "redispatched"}))


__all__ = ["ReplyRecovery", "task_evidence"]
