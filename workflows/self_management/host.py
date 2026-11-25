"""Module-level tool factory for the SelfManagement workflow.

The application supplies one managed-run supervisor and one MemoryStore. The read
tools resolve the supervisor's durable table; the ghost tools use a scoped
view. The same function set is registered on the Assistant MCP server, so
Python steps and MCP clients share one implementation.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from pydantic import JsonValue
from pathlib import Path
from workflows.steward.gate import Steward
from pm.runs.run_receipt import ManagedRunError

from pm.runs.management_api import build_management_tools
from pm.records.memory_records import ActionSchemaRecord
from pm.records.enums import ActionType
from pm.bus.envelopes import Envelope
from workflows.self_management.schemas import ManagementAction, SelfManagementRequest, SelfManagementReport
from workflows.public_speech.publication import exchange_record
from pm.bus.envelopes import new_envelope
from pm.runs.workflow_module import CognitiveWorkflow

if TYPE_CHECKING:
    from pm.runs.supervisor import ManagedRunSupervisor


def build_self_management_tools(
    supervisor: ManagedRunSupervisor,
) -> Callable[[object, object], dict[str, Callable[..., Any]]]:
    """Return ``tools_factory(environment, batch)`` for SelfManagement."""

    def factory(environment: object, batch: object) -> dict[str, Callable[..., Any]]:
        """Return the managed-run tools of one SelfManagement invocation."""
        return build_management_tools(supervisor)

    return factory


__all__ = ["build_self_management_tools"]


class SelfManagement(CognitiveWorkflow):
    """Deterministic managed-run operations as a workflow module.

    It accepts typed actions targeting ``self_management`` and maps the action
    arguments into one typed request. The workflow's tools write the durable
    run receipt; this module publishes an explicit action receipt and never
    invents a completed run.
    """

    DATA_TOPICS = frozenset({"action.request"})
    WORKFLOW_NAME = "self_management"
    managed_workflows: frozenset[str] = frozenset()

    steward: Steward | None = None

    def handle_batch(self, batch: list[Envelope]) -> None:
        for envelope in batch:
            selected = ManagementAction.model_validate(envelope.payload)
            operation = selected.arguments.operation
            if self.steward is None or envelope.target != self.module_id or operation not in {"status", "terminate"}:
                super().handle_batch([envelope])
                continue
            supervisor = self.steward.supervisor
            run_id = selected.arguments.run_id
            try:
                receipt = supervisor.status(run_id)
            except ManagedRunError as error:
                rejected = SelfManagementReport(operation=operation, accepted=False, summary=str(error))
                self.publish_output(rejected.model_dump(mode="json"), [envelope])
                continue
            report = SelfManagementReport(operation=operation, accepted=True, summary=self.steward.describe_progress(receipt),
                                          receipt=dict(receipt))
            body = self.steward.body
            if operation == "terminate":
                token = body.leases.token(body.endpoint, run_id) if body and body.leases.owner(body.endpoint) == run_id else None
                expected = selected.arguments.expected_epoch
                artifact = Path(receipt["artifact_path"])
                epoch = token.epoch if token else None
                if expected is not None and expected != receipt["execution_epoch"]:
                    report.accepted = False
                    report.summary = f"Run {run_id} changed execution epoch after this stop was requested."
                    self.publish_output(report.model_dump(mode="json"), [envelope])
                    continue
                if selected.arguments.yield_body and body:
                    body.relinquish(run_id)
                elif body and expected is not None and (artifact / f"body-stop-request-{expected}.json").is_file():
                    from workflows.minecraft_body.ownership import PausedTask
                    epoch = PausedTask.model_validate_json((artifact / f"body-stop-request-{expected}.json").read_bytes()).epoch
                elif body and token and expected is not None:
                    body.retain_for_resume(run_id, token.epoch, artifact, expected)
                elif body and token is None and operation == "terminate" and selected.arguments.expected_epoch is None:
                    body.relinquish(run_id)
                try:
                    receipt = supervisor.terminate(run_id, request_id=selected.action_id,
                                                  expected_epoch=selected.arguments.expected_epoch)
                except ManagedRunError as error:
                    report.accepted = False
                    report.summary = str(error)
                    self.publish_output(report.model_dump(mode="json"), [envelope])
                    continue
                report.receipt = dict(receipt)
                report.summary = ("Stop requested. Waiting for the worker to confirm the stop and observe its position."
                                  if epoch is not None else f"Run {run_id} is {receipt['state']}.")
                if body and epoch is not None:
                    report.receipt["awaiting_stop_observation"] = True
                    self.publish_output(report.model_dump(mode="json"), [envelope])
                    stopped = body.stop(run_id, epoch, artifact)
                    report.receipt["awaiting_stop_observation"] = False
                    report.operation = "stop_observed"
                    report.summary = "Stopped. " + stopped.reading.summary()
                    report.receipt["stop_observation"] = stopped.model_dump(mode="json")
            if operation == "status" or report.operation == "stop_observed":
                self.bus.put(new_envelope(self.module_id, "telemetry", "workflow.event", envelope.correlation_id,
                    {"run_id": run_id, "workflow_name": receipt["workflow_name"], "event": report.operation,
                     "detail": report.summary, "question_action_id": selected.knoxel_id}, causation_id=envelope.id))
            self.publish_output(report.model_dump(mode="json"), [envelope])

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "action.request" and envelope.kind == "data":
            return envelope.target in {None, self.module_id, self.WORKFLOW_NAME} | self.managed_workflows
        return super().accepts(envelope)

    def prepare_input(self, batch: list[Envelope]) -> dict[str, JsonValue]:
        envelope = batch[0]
        selected = ManagementAction.model_validate(envelope.payload)
        args = selected.arguments
        if self.ghost is None:
            raise ValueError("A self-management action requires companion memory.")
        action = self.ghost.get_knoxel_by_id(selected.knoxel_id) if selected.knoxel_id is not None else None
        original = exchange_record(self.ghost, envelope.correlation_id)
        instructions = args.instructions.strip() or args.instruction.strip() or selected.content
        if not instructions and action is not None:
            instructions = action.content
        request = SelfManagementRequest(operation=args.operation or "status",
            action_id=selected.action_id or args.action_id, run_id=args.run_id,
            workflow_name=args.workflow_name, instructions=instructions,
            parent_correlation=args.parent_correlation or envelope.correlation_id,
            request_id=args.request_id, question_id=args.question_id, answer=args.answer,
            arguments=args.arguments)
        if envelope.target is not None and envelope.target in self.managed_workflows:
            request.workflow_name = envelope.target
            request.operation = args.operation.strip() or ("update" if args.run_id else "start")
            if request.operation == "start":
                request.arguments = args.payload()
        if request.operation == "update" and args.change_source_id is not None:
            request.request_id = f"change:{request.run_id}:{args.change_source_id}"
        request.arguments["_public_reply_requested"] = isinstance(action, ActionSchemaRecord) and action.action_type == ActionType.ToolCallAndReply
        request.arguments["_public_action_schema_id"] = action.id if action else None
        request.arguments["_public_original_request"] = original.content if original else ""
        request.arguments["_public_original_request_id"] = original.id if original else None
        return request.model_dump(mode="json")

    def publish_output(self, report: dict[str, JsonValue], batch: list[Envelope]) -> None:
        if self.ghost is None:
            raise ValueError("A self-management receipt requires companion memory.")
        result = SelfManagementReport.model_validate(report)
        correlation = batch[0].correlation_id if batch else ""
        cause = batch[0].id if batch else None
        selection = ManagementAction.model_validate(batch[0].payload)
        source = self.ghost.get_knoxel_by_id(selection.knoxel_id) if selection.knoxel_id is not None else None
        if source is not None:
            from workflows.main_cognition.dispatch_recovery import acknowledge_intent
            acknowledge_intent(self.ghost, source.id)
        self.bus.put(new_envelope(
            self.module_id, "telemetry", "action.receipt", correlation,
            {
                "workflow_name": self.WORKFLOW_NAME,
                "status": "accepted" if result.accepted else "rejected",
                "operation": result.operation,
                "summary": result.summary,
                "receipt": result.receipt,
                "action_schema_id": source.id if source is not None else None,
                "public_reply_requested": isinstance(source, ActionSchemaRecord)
                and source.action_type == ActionType.ToolCallAndReply,
            },
            causation_id=cause,
        ))
