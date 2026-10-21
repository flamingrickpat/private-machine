"""Module-level tool factory for the SelfManagement workflow.

The application supplies one managed-run supervisor and one MemoryStore. The read
tools resolve the supervisor's durable table; the ghost tools use a scoped
view. The same function set is registered on the Assistant MCP server, so
Python steps and MCP clients share one implementation.
"""
from __future__ import annotations

from typing import Callable

from pm.runs.management_api import build_management_tools
from pm.records.memory_records import ActionSchemaRecord
from pm.records.enums import ActionType
from pm.bus.envelopes import Envelope
from workflows.self_management.schemas import SelfManagementRequest
from workflows.public_speech.publication import exchange_record
from pm.bus.envelopes import new_envelope
from pm.runs.workflow_module import CognitiveWorkflow


def build_self_management_tools(supervisor) -> Callable[[object, object], dict]:
    """Return ``tools_factory(environment, batch)`` for SelfManagement."""

    def factory(environment, batch) -> dict:
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
    managed_workflows = frozenset()

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "action.request" and envelope.kind == "data":
            return envelope.target in {None, self.module_id, self.WORKFLOW_NAME} | self.managed_workflows
        return super().accepts(envelope)

    def prepare_input(self, batch: list[Envelope]) -> dict:
        payload = batch[0].payload
        action = self.ghost.get_knoxel_by_id(payload["knoxel_id"]) if "knoxel_id" in payload else None
        arguments = dict(payload.get("arguments") or {})
        original = exchange_record(self.ghost, batch[0].correlation_id)
        if batch[0].target in self.managed_workflows:
            # A direct alias can address an existing run. Keep its identity at
            # the control boundary, rather than nesting it in new-worker input.
            operation = arguments.get("operation", "update" if arguments.get("run_id") else "start")
            if operation == "start":
                arguments = {
                    "operation": "start", "workflow_name": batch[0].target,
                    "instructions": str(arguments.get("instructions", arguments.get("instruction", payload.get("content", "")))),
                    "arguments": arguments,
                }
            else:
                arguments = {**arguments, "operation": operation, "workflow_name": batch[0].target,
                    "instructions": str(arguments.get("instructions", arguments.get("instruction", payload.get("content", ""))))}
        request = SelfManagementRequest(
            operation=str(arguments.get("operation", "status")),
            action_id=str(payload.get("action_id", arguments.get("action_id", ""))),
            run_id=str(arguments.get("run_id", "")),
            workflow_name=str(arguments.get("workflow_name", "")),
            instructions=str(arguments.get("instructions", getattr(action, "content", ""))),
            parent_correlation=str(arguments.get("parent_correlation", batch[0].correlation_id)),
            request_id=str(arguments.get("request_id", "")),
            question_id=str(arguments.get("question_id", "")),
            answer=str(arguments.get("answer", "")),
            arguments={**dict(arguments.get("arguments", {})),
                       "_public_reply_requested": bool(action is not None and action.action_type == ActionType.ToolCallAndReply),
                       "_public_action_schema_id": getattr(action, "id", None),
                       "_public_original_request": original.content if original is not None else "",
                       "_public_original_request_id": original.id if original is not None else None},
        )
        return request.model_dump(mode="json")

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        correlation = batch[0].correlation_id if batch else ""
        cause = batch[0].id if batch else None
        source = self.ghost.get_knoxel_by_id(batch[0].payload.get("knoxel_id"))
        if source is not None:
            from workflows.main_cognition.dispatch_recovery import acknowledge_intent
            acknowledge_intent(self.ghost, source.id)
        self.bus.put(new_envelope(
            self.module_id, "telemetry", "action.receipt", correlation,
            {
                "workflow_name": self.WORKFLOW_NAME,
                "status": "accepted" if report.get("accepted") else "rejected",
                "operation": report.get("operation", ""),
                "summary": report.get("summary", ""),
                "receipt": report.get("receipt", {}),
                "action_schema_id": getattr(source, "id", None),
                "public_reply_requested": isinstance(source, ActionSchemaRecord)
                and source.action_type == ActionType.ToolCallAndReply,
            },
            causation_id=cause,
        ))
