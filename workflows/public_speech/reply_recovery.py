"""Resume selected task replies from durable run and exchange evidence.

A managed task outlives the cycle that started it, so its public answer
cannot come from that cycle. This module rebuilds the missing request from
durable evidence: the accepted user exchange, the selected action, and the
admission or terminal run receipt. The action identifies whether its worker
owes a public reply.

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
from dataclasses import dataclass
from pm.host.application import Application
import json
from typing import TYPE_CHECKING
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from workflows.public_speech.task_receipts import AdmissionObservation, TaskReceipt, TaskReportEvidence, TaskVerificationEvidence, recent_body_effects

from pm.utils import yaml_util

from pm.bus.envelopes import Envelope, new_id
from pm.records.base import RecordBase
from pm.records.memory_records import ActionSchemaRecord, Feature
from pm.records.enums import ActionType
from workflows.public_speech.publication import exchange_record, publish_reply
from pm.host.application import BusModule

if TYPE_CHECKING:
    from pm.memory.store import MemoryStore
    from pm.runs.supervisor import ManagedRunSupervisor
    from workflows.main_cognition.schemas import ActionRequest
    from workflows.main_cognition.host import MainCogLoop
    from workflows.cognitive_control.host import CognitiveControl

TERMINAL = frozenset({"completed", "failed", "cancelled", "interrupted"})


class TaskReceiptCausality(BaseModel):
    """The existing supervisor receipt's optional historical commit reference."""

    model_config = ConfigDict(extra="ignore")
    causal_commit_id: str | None = None
    input_record_ids: list[int] = Field(default_factory=list[int])
    input_commit_ids: list[str] = Field(default_factory=list[str])


def user_reply_request(ghost: MemoryStore, *, knoxel_id: int, action_id: str,
                       exchange: RecordBase, task_receipt: TaskReceipt,
                       workspace: str = "") -> ActionRequest:
    """Return the typed request of a task reply to the user, outside a cycle.

    The reply answers the speaker of ``exchange``. It carries the self
    shard and the speaker's shard of the newest cycle state, so the reply reads
    the same conscious content as a reply that a cycle chose. The caller
    holds no store lock.
    """
    from pm.records.workspace import GlobalWorkspace, SELF_SHARD
    from workflows.main_cognition.action_request import partner_key, request_shards
    from workflows.main_cognition.schemas import ActionRequest, Stimulus
    with ghost.lock:
        workspace_state = ghost.states[-1].workspace if ghost.states else GlobalWorkspace()
        if not isinstance(exchange, Feature):
            raise TypeError(f"Task reply requires a dialogue source, got {exchange.id}.")
        partner_name = exchange.source or ghost.get_user_name()
        partner_entity_id = exchange.entity_id or ghost.get_entity_id(partner_name)
    partner = partner_key(SELF_SHARD, partner_entity_id)
    return ActionRequest(
        knoxel_id=knoxel_id, action_id=action_id, workflow_name="dialogue_assistant",
        partner_entity_id=partner_entity_id, partner_name=partner_name,
        stimuli=[Stimulus(knoxel_id=exchange.id, text=str(exchange.content), source="user_input",
                          entity_id=partner_entity_id, correlation_id=str(exchange.metadata["exchange_correlation"]))],
        action_origin="task_reply",
        shards=request_shards(workspace_state, SELF_SHARD, partner,
                              {key: partner_name if key.kind == "entity" else "self" for key in (SELF_SHARD, partner) if key is not None}),
        task_receipt=task_receipt, workspace=workspace)


def task_evidence(receipt: TaskReceipt) -> TaskReceipt:
    result = receipt.model_copy(deep=True)
    result.task_verified = False
    if not receipt.artifact_path:
        return result
    root = Path(receipt.artifact_path).resolve()
    if receipt.workflow_name == "minecraft_project" and receipt.state in TERMINAL:
        result.evidence = (*recent_body_effects(root), *result.evidence)
    report_path = root / "outbox" / "report.yaml"
    if not report_path.is_file():
        return result
    header = TaskReportEvidence.model_validate(yaml_util.safe_load(report_path.read_text(encoding="utf-8")))
    if header.schema_name == "pm.tool-call-report.v1":
        from workflows.tool_call.report import read_reply_evidence
        tool_evidence = read_reply_evidence(root)
        result.report = tool_evidence.report
        result.file_checks = tool_evidence.file_checks
        return result
    result.report = header
    if not header.verified:
        return result
    match header.schema_name:
        case "pm.minecraft-project-report.v1":
            from workflows.minecraft_project.acceptance import verified_project
            verification = TaskVerificationEvidence.model_validate(verified_project(root))
            result.minecraft_verification = verification
            result.task_verified = receipt.state == "completed" and verification.verified
        case "pm.note-verification.v1":
            specification = header.note_spec
            if specification is None:
                raise ValueError("A verified note report requires its exact note specification.")
            path = (root / specification.path).resolve()
            if not path.is_relative_to(root / "outbox"):
                raise ValueError("The note evidence escaped its task outbox.")
            actual = path.read_bytes() if path.is_file() else None
            result.task_verified = receipt.state == "completed" and actual == specification.text.encode("utf-8")
            result.actual_hex = actual.hex() if actual is not None else None
        case "pm.minecraft-actor-report.v1":
            acceptance = header.acceptance
            if acceptance is None:
                raise ValueError("A verified actor report requires its original acceptance contract.")
            if acceptance.kind != "wood_return":
                return result
            if not header.journal:
                raise ValueError("A verified actor report requires its original journal path.")
            from workflows.minecraft.specialist_session import verify_wood_return
            journal = (root / header.journal).resolve()
            if not journal.is_relative_to(root / "scratch"):
                raise ValueError("The Minecraft journal escaped its task scratch directory.")
            rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
            calls = [row for row in rows if row["event"] == "result"]
            if (calls and calls[0]["tool"] == calls[-1]["tool"] == "minecraft_observe" and calls[0]["ok"] and calls[-1]["ok"]):
                verification = TaskVerificationEvidence.model_validate(verify_wood_return(calls,
                    calls[0]["result"], calls[-1]["result"], acceptance))
                result.minecraft_verification = verification
                result.task_verified = receipt.state == "completed" and verification.verified
        case _:
            return result
    return result


@dataclass(frozen=True, slots=True)
class ReplyRecoverySelection:
    ghost: MemoryStore
    control: CognitiveControl
    main_cognition: MainCogLoop

    def selected_source_ids(self) -> set[int]:
        from workflows.interaction.records import interaction_receipts
        return set(interaction_receipts(self.ghost.recovery_snapshot(capture=False)))

    def paused_source_ids(self) -> set[int]:
        return self.control.paused_source_ids() | self.main_cognition.entity_reconsider_source_ids()


class ReplyRecovery(BusModule):
    """Dispatch a terminal reply only when the prior action selected public speech."""

    DATA_TOPICS = frozenset({"workflow.event", "workflow.question", "action.receipt", "public.resume"})

    def __init__(self, module_id: str, application: Application, tick_seconds: float | None, *, ghost: MemoryStore,
                 supervisor: ManagedRunSupervisor) -> None:
        super().__init__(module_id, application, tick_seconds)
        self.ghost = ghost
        self.supervisor = supervisor
        self.selection: ReplyRecoverySelection | None = None

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
                                 "operation_result": envelope.payload, "summary": envelope.payload["summary"],
                                 "arguments": {"_public_reply_requested": True,
                                               "_public_action_schema_id": envelope.payload.get("action_schema_id")}}
                    self.reply_to_task(operation)
                elif receipt.get("state") in TERMINAL:
                    self.reply_to_task(receipt)
                elif receipt.get("state") in {"accepted", "running", "waiting_for_input"}:
                    current = self.supervisor.status(receipt["run_id"])
                    if current["state"] not in TERMINAL:
                        self.reply_to_task(current, admission=True)
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
            exchange=exchange, task_receipt=TaskReceipt(workflow_name="self_management", state="rejected", task_verified=False, summary=text))
        self.bus.put(Envelope(id=new_id(), source=self.module_id, kind="data", topic="action.request",
            correlation_id=correlation, target="dialogue_assistant", payload=request.payload()))

    def reply_to_task(self, receipt: dict[str, JsonValue], *, admission: bool = False) -> None:
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
        evidence = TaskReceipt.model_validate(receipt)
        if admission:
            evidence.admission_only = True
            identity = f"task-admission:{evidence.run_id}:{source.id}"
            instruction = ("Briefly acknowledge the accepted background task in your own voice. If asked what you "
                           "see, answer from the dated body observation if available, or say you will look.")
            workspace = ("This receipt confirms admission. Describe its actual state in one brief acknowledgment. "
                         "Any observation, work done, or result requires separate evidence. "
                         "The terminal outcome will be reported separately.")
        elif "operation_result" in receipt:
            operation = receipt["operation_result"]["operation"]
            if operation in {"terminate", "stop_observed"}:
                evidence.stop_source_id = exchange.id
            identity = f"management-reply:{operation}:{evidence.run_id}:{source.id}"
            instruction = "Answer the current question using the dated management report."
            if operation == "resume" and receipt["operation_result"]["status"] == "accepted":
                instruction = "Acknowledge that the same worker task has resumed from the retained world. Jesse has already authorized it."
            if evidence.awaiting_stop_observation:
                instruction = ("Briefly say you received the stop request and are checking where you stopped. "
                               "Coordinates and confirmation will follow from the worker.")
            workspace = evidence.summary
        else:
            evidence = task_evidence(evidence)
            identity = f"task-reply:{evidence.run_id}:{receipt['revision']}:{source.id}"
            instruction = "Answer the original request using this terminal task receipt. Explain verified work, failure, or uncertainty accurately."
            workspace = "The selected task reached a terminal boundary. Its receipt does not imply unverified instruction execution."
        with self.ghost.lock:
            prior = next((item for item in self.ghost.all_knoxels.values()
                          if isinstance(item, ActionSchemaRecord) and item.metadata.get("action_id") == identity), None)
            if prior is None:
                changes = self.ghost.begin_batch(identity)
                changes.depend_on([source.id, exchange.id], authority=source.id)
                if admission and evidence.workflow_name in {"minecraft", "minecraft_project"}:
                    observed = next((item for item in reversed(self.ghost.all_features)
                                     if item.scope == source.scope and item.metadata.get("origin") == "minecraft.world"), None)
                    if observed is not None:
                        evidence.admission_observation = AdmissionObservation(source_id=observed.id,
                            observed_at=observed.timestamp_world_begin, text=observed.content)
                        changes.depend_on([observed.id])
                causality = TaskReceiptCausality.model_validate(receipt)
                changes.depend_on(causality.input_record_ids)
                changes.depend_on_commits(causality.input_commit_ids)
                if causality.causal_commit_id:
                    changes.depend_on_commits([causality.causal_commit_id])
                action = changes.create(ActionSchemaRecord(
                    action_type=ActionType.Reply,
                    content=instruction,
                    intention_ids=list(source.intention_ids), scope=source.scope,
                    metadata={**source.metadata, "action_id": identity, "workflow_name": "dialogue_assistant",
                              "source_action_schema_id": source.id, "task_run_id": receipt["run_id"]},
                ))
                request = user_reply_request(self.ghost, knoxel_id=action.id, action_id=identity,
                    exchange=exchange, task_receipt=evidence,
                    workspace=workspace)
                action.metadata = {**action.metadata, "dispatch_status": "pending", "dispatch_target": "dialogue_assistant",
                                   "dispatch_correlation": correlation, "dispatch_payload": request.payload()}
                self.ghost.commit(changes)
            else:
                from workflows.main_cognition.schemas import PendingDispatch
                request = PendingDispatch.model_validate(prior.metadata).dispatch_payload
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
        selected_sources = self.selection.selected_source_ids() if self.selection is not None else set()
        paused = self.selection.paused_source_ids() if self.selection is not None else set()
        for item in self.ghost.all_features:
            correlation = item.metadata.get("exchange_correlation")
            if (correlation and item.metadata.get("exchange_status") == "pending"
                    and correlation not in waiting and item.id not in paused and item.id not in selected_sources):
                self.bus.put(Envelope(id=new_id(), source="user_input", kind="data", topic="user.input",
                                     correlation_id=correlation, payload={"knoxel_id": item.id, "recovery_outcome": "redispatched"}))


__all__ = ["ReplyRecovery", "task_evidence", "user_reply_request"]
