"""Resume selected task replies from durable run and exchange evidence.

A managed task outlives the cycle that started it, so its public answer
cannot come from that cycle. This module rebuilds the missing request from
durable evidence: the accepted user exchange, the selected reply action, and
the terminal run receipt. It never invents speech and never asks for a reply
that the cycle did not already choose.

Inputs are bus envelopes (``workflow.event``, ``workflow.question``,
``action.receipt``, ``public.resume``) and the supervisor's receipts. Outputs
are ``action.request`` envelopes for the assistant workflow, plus retained
reply actions that make the recovery idempotent. ``task_evidence`` is the
only place that decides whether a receipt proves the work it claims.

Invariants:

- A reply is dispatched only when the original action selected public speech
  (``_public_reply_requested``) and the reply action still resolves.
- One identity per run revision yields one reply action, so a replayed resume
  cannot speak twice.
- Evidence that escapes its task folder raises ValueError instead of being
  accepted or silently dropped.
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import TYPE_CHECKING, Any

from pm.utils import yaml_util

from pm.bus.envelopes import Envelope, new_id
from pm.records.base import RecordBase
from pm.records.memory_records import ActionSchemaRecord
from pm.records.enums import ActionType
from workflows.public_speech.publication import commit_public_reply, exchange_record, publish_reply
from pm.host.application import BusModule

if TYPE_CHECKING:
    from pm.memory.store import MemoryStore
    from pm.runs.supervisor import ManagedRunSupervisor
    from workflows.main_cognition.schemas import ActionRequest

TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})


def user_reply_request(ghost: MemoryStore, *, knoxel_id: int, action_id: str,
                       exchange: RecordBase, task_receipt: dict[str, Any],
                       workspace: str = "") -> ActionRequest:
    """Return the typed request of a task reply to the user, outside a cycle.

    The reply answers the user exchange ``exchange``. It carries the self
    shard and the user's shard of the newest cycle state, so the reply reads
    the same conscious content as a reply that a cycle chose. The caller
    holds no store lock.
    """
    from pm.records.workspace import GlobalWorkspace, SELF_SHARD
    from workflows.main_cognition.action_request import partner_key, request_shards
    from workflows.main_cognition.schemas import ActionRequest, Stimulus
    with ghost.lock:
        workspace_state = ghost.states[-1].workspace if ghost.states else GlobalWorkspace()
        user_entity_id = ghost.get_entity_id(ghost.get_user_name())
        user_name = ghost.get_user_name()
    partner = partner_key(SELF_SHARD, user_entity_id)
    return ActionRequest(
        knoxel_id=knoxel_id, action_id=action_id, workflow_name="dialogue_assistant",
        partner_entity_id=user_entity_id, partner_name=user_name,
        stimuli=[Stimulus(knoxel_id=exchange.id, text=str(exchange.content), source="user_input",
                          entity_id=user_entity_id)],
        shards=request_shards(workspace_state, SELF_SHARD, partner,
                              lambda key: user_name if key.kind == "entity" else "self"),
        task_receipt=task_receipt, workspace=workspace)


def task_evidence(receipt: dict[str, Any]) -> dict[str, Any]:
    """Verify note bytes or retained Minecraft receipts before public dispatch.

    Returns a copy of the receipt with ``task_verified`` decided here and the
    evidence that decided it (``report``, ``minecraft_verification``, or the
    actual note bytes). Only a completed run with matching retained evidence
    is verified; every other case stays unverified, never failed silently.
    """
    result = dict(receipt)
    if not receipt.get("artifact_path"):
        result["task_verified"] = False
        return result
    root = Path(receipt["artifact_path"]).resolve()
    report_path = root / "outbox" / "report.yaml"
    if report_path.is_file():
        report = yaml_util.safe_load(report_path.read_text(encoding="utf-8"))
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

    def __init__(self, *args: Any, ghost: MemoryStore, supervisor: ManagedRunSupervisor,
                 **kwargs: Any) -> None:
        """Forward the module configuration and retain the store and supervisor."""
        super().__init__(*args, **kwargs)
        self.ghost = ghost
        self.supervisor = supervisor

    def handle_batch(self, batch: list[Envelope]) -> None:
        """Acknowledge run events and dispatch the replies they owe."""
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

    def fail_exchange(self, correlation: str, text: str) -> None:
        """Request an agent-written explanation of an expected task rejection.

        The supplied text is evidence, not companion speech. Broken internal
        evidence still raises in reply_to_task before any request reaches here.
        """
        exchange = exchange_record(self.ghost, correlation)
        if exchange is None:
            raise ValueError(f"Task rejection has no user source: {correlation}")
        identity = f"task-rejection:{correlation}"
        changes = self.ghost.begin_batch(identity)
        action = changes.create(ActionSchemaRecord(action_type=ActionType.Reply,
            content="Explain the rejected task using its actual receipt, in your own voice.",
            metadata={"action_id": identity, "workflow_name": "dialogue_assistant"}))
        receipt = self.ghost.commit(changes)
        request = user_reply_request(self.ghost, knoxel_id=receipt.added_ids[0], action_id=identity,
            exchange=exchange, task_receipt={"workflow_name": "self_management", "state": "rejected",
                                             "task_verified": False, "summary": text})
        self.bus.put(Envelope(id=new_id(), source=self.module_id, kind="data", topic="action.request",
            correlation_id=correlation, target="dialogue_assistant", payload=request.payload()))

    def reply_to_task(self, receipt: dict[str, Any]) -> None:
        """Retain the original selected action as the cause of a receipt-based reply.

        A receipt that never asked for a public reply, or whose exchange is
        gone, produces nothing. A receipt whose selected action is missing or
        of the wrong type raises, because that is broken internal state.
        """
        correlation = receipt.get("parent_correlation", "")
        arguments = receipt.get("arguments", {})
        source_id = arguments.get("_public_action_schema_id")
        source = self.ghost.get_knoxel_by_id(source_id)
        exchange = exchange_record(self.ghost, correlation)
        if exchange is None or not arguments.get("_public_reply_requested"):
            return
        if not isinstance(source, ActionSchemaRecord) or source.action_type != ActionType.ToolCallAndReply:
            raise ValueError(f"Task reply lost its selected action: {source_id}")
        evidence = task_evidence(receipt)
        # A workflow can re-read its own evidence (for example local artifact
        # bytes) before the reply. This library imports no task workflow, so
        # the check comes from the workflow's managed-run specification.
        workflows = self.supervisor.workflows
        if receipt["workflow_name"] in workflows and workflows[receipt["workflow_name"]].reply_evidence is not None:
            evidence = {**evidence, **workflows[receipt["workflow_name"]].reply_evidence(receipt)}
        identity = f"task-reply:{receipt['run_id']}:{receipt['revision']}:{source.id}"
        with self.ghost.lock:
            prior = next((item for item in self.ghost.all_knoxels.values()
                          if isinstance(item, ActionSchemaRecord) and item.metadata.get("action_id") == identity), None)
            if prior is None:
                changes = self.ghost.begin_batch(identity)
                action = changes.create(ActionSchemaRecord(
                    action_type=ActionType.Reply,
                    content="Answer the original request using this terminal task receipt. Explain verified work, failure, or uncertainty accurately.",
                    intention_ids=list(source.intention_ids),
                    metadata={**source.metadata, "action_id": identity, "workflow_name": "dialogue_assistant",
                              "source_action_schema_id": source.id, "task_run_id": receipt["run_id"]},
                ))
                accepted = self.ghost.commit(changes)
                action_id = accepted.added_ids[0]
            else:
                action_id = prior.id
        request = user_reply_request(self.ghost, knoxel_id=action_id, action_id=identity, exchange=exchange,
            task_receipt=evidence,
            workspace="The selected task reached a terminal boundary. Its receipt does not imply unverified instruction execution.")
        self.bus.put(Envelope(
            id=new_id(), source=self.module_id, kind="data", topic="action.request", correlation_id=correlation,
            target="dialogue_assistant", payload=request.payload(),
        ))

    def resume(self) -> None:
        """Replay pending publication and task replies after a checkpoint restart.

        Replay covers three sources: retained intents, publications left
        pending, and terminal receipts that still owe a reply. A user exchange
        that nothing else answers is redispatched as a fresh stimulus.
        """
        from workflows.main_cognition.dispatch_recovery import replay_intents
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
        paused = getattr(self, "paused_source_ids", lambda: set())()
        for item in self.ghost.all_features:
            correlation = item.metadata.get("exchange_correlation")
            if (correlation and item.metadata.get("exchange_status") == "pending"
                    and correlation not in waiting and item.id not in paused):
                self.bus.put(Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                                     correlation_id=correlation, payload={"knoxel_id": item.id, "recovery_outcome": "redispatched"}))


__all__ = ["ReplyRecovery", "task_evidence", "user_reply_request"]
