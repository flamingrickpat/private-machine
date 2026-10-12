"""Run states, run errors, and the durable receipt of one managed run."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping




STATES = frozenset({
    "accepted", "running", "waiting_for_input", "completed", "failed",
    "cancelled", "interrupted",
})


ACTIVE_STATES = frozenset({"accepted", "running", "waiting_for_input"})


class ManagedRunError(RuntimeError):
    """A management operation cannot complete as requested."""


class UnavailableWorkflow(ManagedRunError):
    """The run named a workflow the application does not manage."""


@dataclass
class ManagedRunReceipt:
    """One row of the managed-run table. Serialized into MemoryStore verbatim."""

    run_id: str
    action_id: str
    workflow_name: str
    parent_correlation: str
    state: str
    latest_event: str
    artifact_path: str
    instructions: str = ""
    revision: int = 0
    applied_revision: int = -1
    updates_open: bool = True
    question_id: str = ""
    answer: str = ""
    request_ids: list[str] = field(default_factory=list)
    arguments: dict = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    execution_epoch: int = 1
    control_version: int = 0
    interrupted_from: str = ""
    events: list[dict] = field(default_factory=list)
    resume_requests: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError(f"Unknown managed-run state: {self.state!r}")

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "action_id": self.action_id,
            "workflow_name": self.workflow_name,
            "parent_correlation": self.parent_correlation,
            "state": self.state,
            "latest_event": self.latest_event,
            "artifact_path": self.artifact_path,
            "instructions": self.instructions,
            "revision": self.revision,
            "applied_revision": self.applied_revision,
            "updates_open": self.updates_open,
            "question_id": self.question_id,
            "answer": self.answer,
            "request_ids": list(self.request_ids),
            "arguments": dict(self.arguments),
            "evidence": list(self.evidence),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "execution_epoch": self.execution_epoch,
            "control_version": self.control_version,
            "interrupted_from": self.interrupted_from,
            "events": list(self.events),
            "resume_requests": list(self.resume_requests),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ManagedRunReceipt":
        known = {name: deepcopy(data[name]) for name in cls.__dataclass_fields__ if name in data}
        return cls(**known)

