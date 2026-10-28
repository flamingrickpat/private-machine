"""Run states, run errors, and the durable receipt of one managed run.

One managed run is one accepted row in the supervisor's table. The receipt is
the durable identity of that work: who requested it, which workflow runs, what
instructions the role currently holds, how far the control surface advanced,
and which evidence and events already reached cognition.

:meth:`ManagedRunReceipt.to_dict` is the stored and transported form. It crosses
``MemoryStore``, a small per-run control file read by other processes, and the
JSON tool boundary, so the mapping keeps the same keys as the dataclass. That is
why the row travels as a plain mapping while this dataclass stays the typed
form that code constructs and reads.

A row is accepted before its worker starts, and a terminal row is never
rewritten. ``state`` is one of :data:`STATES`; an unknown value fails in
``__post_init__`` instead of becoming a half-known run.
"""
from __future__ import annotations

from pm.records.copying import pickle_copy

from dataclasses import dataclass, field
from typing import Any, Mapping, TypedDict

from pydantic import JsonValue




STATES = frozenset({
    "accepted", "running", "waiting_for_input", "completed", "failed",
    "cancelled", "interrupted",
})


ACTIVE_STATES = frozenset({"accepted", "running", "waiting_for_input"})


class RunEvent(TypedDict):
    """One published-once event of a run, with its acknowledgment state.

    ``message`` is the event that cognition receives; ``event_id`` inside it is
    the stable identity a consumer acknowledges. ``published`` stays false until
    that acknowledgment, so a restart can publish the same event again.
    """

    message: dict[str, JsonValue]
    published: bool


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
    arguments: dict[str, JsonValue] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    execution_epoch: int = 1
    control_version: int = 0
    interrupted_from: str = ""
    events: list[RunEvent] = field(default_factory=list)
    resume_requests: list[str] = field(default_factory=list)
    singleton_key: str = ""
    blocked_by_run_id: str = ""
    authority_record_id: int | None = None
    input_record_ids: list[int] = field(default_factory=list)
    input_commit_ids: list[str] = field(default_factory=list)
    resume_phase: str = ""
    causal_commit_id: str = ""
    parked_state: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject an unknown run state before the row can be stored."""
        if self.state not in STATES:
            raise ValueError(f"Unknown managed-run state: {self.state!r}")

    def to_dict(self) -> dict:
        """Return the row that is stored in the table and mirrored to control files.

        The shape is this dataclass: the same keys, in the same order, with the
        parked state copied so a later disk write cannot mutate the live receipt.
        """
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
            "singleton_key": self.singleton_key,
            "blocked_by_run_id": self.blocked_by_run_id,
            "authority_record_id": self.authority_record_id,
            "input_record_ids": list(self.input_record_ids),
            "input_commit_ids": list(self.input_commit_ids),
            "resume_phase": self.resume_phase,
            "causal_commit_id": self.causal_commit_id,
            "parked_state": pickle_copy(self.parked_state),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ManagedRunReceipt":
        """Rebuild a receipt from a stored or transported row.

        Only declared fields are read, so a row written by an older build keeps
        working and its absent fields take the dataclass defaults.
        """
        known = {name: pickle_copy(data[name]) for name in cls.__dataclass_fields__ if name in data}
        return cls(**known)

