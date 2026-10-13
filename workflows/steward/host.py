"""Module-level tool factory for the steward workflow."""

from __future__ import annotations

from typing import Callable

from workflows.steward.gate import build_steward_tools
from pm.bus.envelopes import Envelope
from workflows.steward.schemas import StewardRequest
from pm.bus.envelopes import new_envelope
from pm.runs.workflow_module import CognitiveWorkflow


def build_steward_module_tools(steward) -> Callable[[object, object], dict]:
    """Return ``tools_factory(environment, batch)`` for the steward module."""

    def factory(environment, batch) -> dict:
        return build_steward_tools(steward)

    return factory


__all__ = ["build_steward_module_tools"]


class StewardModule(CognitiveWorkflow):
    """Route typed actions to the dormant steward.

    The module never activates the steward. While dormant, every operation is
    refused and no run state changes. A caller activates ``app.steward`` when a
    representative is wanted.
    """

    DATA_TOPICS = frozenset({"action.request"})
    WORKFLOW_NAME = "steward"

    def accepts(self, envelope: Envelope) -> bool:
        if envelope.topic == "action.request" and envelope.kind == "data":
            return envelope.target in {None, self.module_id, self.WORKFLOW_NAME}
        return super().accepts(envelope)

    def prepare_input(self, batch: list[Envelope]) -> dict:
        payload = batch[0].payload
        arguments = dict(payload.get("arguments") or {})
        request = StewardRequest(
            operation=str(arguments.get("operation", "status")),
            run_id=str(arguments.get("run_id", "")),
            question_id=str(arguments.get("question_id", "")),
            answer=str(arguments.get("answer", "")),
            instructions=str(arguments.get("instructions", payload.get("content", ""))),
            request_id=str(arguments.get("request_id", "")),
        )
        return request.model_dump(mode="json")

    def publish_output(self, report: dict, batch: list[Envelope]) -> None:
        correlation = batch[0].correlation_id if batch else ""
        cause = batch[0].id if batch else None
        self.bus.put(new_envelope(
            self.module_id, "telemetry", "action.receipt", correlation,
            {
                "workflow_name": self.WORKFLOW_NAME,
                "status": "accepted" if report.get("accepted") else "rejected",
                "operation": report.get("operation", ""),
                "summary": report.get("summary", ""),
                "receipt": report.get("result", {}),
            },
            causation_id=cause,
        ))
