"""Runtime lifecycle controls shared by local chat and simulation clients.

``RuntimeControl`` is the single read/write surface for lifecycle state: it
freezes and thaws a runtime, advances a virtual clock, sends stimuli, captures
snapshots, and answers trace questions. The web chat and the simulation
browser call it; the assembly supplies the application object it wraps.

Each call returns a JSON response that the caller writes as-is. Invalid state
or an unknown control path raises ``ValueError``, so a controller never
receives a plausible-looking success. Any clock change or snapshot requires an
idle runtime, because a snapshot taken during work would not match its cursor.
"""
from __future__ import annotations

from datetime import datetime
from collections.abc import Sequence
from pathlib import Path
import re
import threading
from typing import Protocol, TypedDict

from pydantic import BaseModel, JsonValue
from pm.host.application import Application

from pm import vclock
from pm.app.transport import event_cursor
from pm.memory.checkpoint import write_checkpoint
from pm.memory.store import MemoryStore
from pm.records.media import RecordImage
from pm.runs.workflow_module import WorkflowModule


class IntakeClosed(RuntimeError):
    """A stimulus arrived after the operator closed intake."""


class OperatorIntake(Protocol):
    """The public input operations exposed to terminal and simulation controllers."""

    def close_intake(self) -> None: ...
    def open_intake(self) -> None: ...
    def submit(self, text: str, *, session: str = "", speaker: str | None = None,
               images: Sequence[RecordImage] = (), delivery_id: str = "", entity_ref: str = "") -> str: ...


class OperatorStimulus(BaseModel):
    """One HTTP control stimulus, optionally delivered at a requested world time."""

    text: str
    speaker: str | None = None
    delivery_id: str = ""
    ts: str | None = None


class ModuleActivity(TypedDict):
    """Runnable state of one bus module at the moment of the read."""

    id: str
    working: bool
    queued: int
    parked: bool
    next_tick: float | None


class RuntimeState(TypedDict):
    """Lifecycle and activity snapshot of one runtime.

    ``status`` is ``running``, ``draining``, ``frozen``, ``stopped``, or
    ``failed``. ``clock`` is an aware ISO timestamp; ``virtual`` says whether
    a virtual clock drives it. ``next_due`` is the next semantic timer
    deadline, or None when no timer is due. ``error`` is the representation of
    the first application error.
    """

    status: str
    idle: bool
    tick_id: int
    clock: str
    virtual: bool
    output_cursor: str
    next_due: str | None
    error: str | None
    modules: list[ModuleActivity]


class DeliveryState(TypedDict):
    """Whether a stimulus exists and its retained history confirms acceptance."""

    present: bool
    accepted: bool
    record_id: int | None


class StimulusReceipt(TypedDict):
    """The correlation identity of one submitted stimulus."""

    status: str
    stimulus_id: str


class SnapshotReceipt(TypedDict):
    """The written checkpoint of one manual snapshot."""

    name: str
    path: str
    clock: str


class TraceReceipt(TypedDict):
    """One output record with the commit transaction that produced it."""

    record: dict[str, JsonValue]
    transaction: dict[str, JsonValue] | None
    calls: list[JsonValue]
    call_linkage: str


class RuntimeControl:
    """Expose one running application without importing companion behavior.

    Snapshot names stay inside the persona runtime directory. Clock changes
    require quiescence. Restoring a process belongs to its external controller.
    """

    def __init__(self, app: Application, ghost: MemoryStore, intake: OperatorIntake, runtime_root: Path) -> None:
        self.app = app
        self.ghost = ghost
        self.intake = intake
        self.runtime_root = runtime_root
        self._lock = threading.RLock()

    def state(self) -> RuntimeState:
        """Return runnable activity, lifecycle state, and the latest event cursor."""
        app = self.app
        cursor = event_cursor(self.ghost)
        return {"status": "failed" if app.error else "stopped" if app.shutdown_event.is_set()
                else "draining" if app.draining.is_set() else "frozen" if app.frozen.is_set() else "running",
                "idle": app.idle(), "tick_id": self.ghost.current_tick_id,
                "clock": vclock.now_aware().isoformat(), "virtual": vclock.active() is not None,
                "output_cursor": cursor,
                "next_due": self.next_due(),
                "error": repr(app.error) if app.error else None,
                "modules": [{"id": item.module_id, "working": item.working,
                             "queued": item.mailbox.pending(), "parked": isinstance(item, WorkflowModule) and item.parked,
                             "next_tick": item.next_tick_seconds} for item in app.modules]}

    def next_due(self) -> str | None:
        """Return the next semantic timer deadline, excluding empty polling.

        A cognitive wait suppresses private inference. Its wake deadline
        remains active. The scheduler's idle poll is not a world event.
        """
        from datetime import timedelta
        now = vclock.now_aware()
        with self.ghost.lock:
            wait = next((item for item in reversed(self.ghost.all_features)
                         if item.metadata.get("control_kind") == "inference_wait"), None)
            if wait is not None and wait.metadata["status"] == "waiting":
                if wait.metadata.get("memory_action_id") is not None:
                    return None
                return datetime.fromisoformat(wait.metadata["wake_at"]).replace(tzinfo=now.tzinfo).isoformat()
        due = [now + timedelta(seconds=max(0, item.next_tick_seconds - self.app.clock.monotonic()))
               for item in self.app.modules if item.next_tick_seconds is not None and not item.draining
               and item.module_id not in {"scheduler", "cognitive_control"}]
        return min(due).isoformat() if due else None

    def advance_clock(self, timestamp: str) -> None:
        """Advance the simulation clock and let each module notice an overdue timer."""
        clock = vclock.active()
        if clock is None:
            raise ValueError("This process has no virtual clock.")
        clock.set(datetime.fromisoformat(timestamp))
        for module in self.app.modules:
            module.mailbox.wake()

    def __call__(
        self, path: str, body: dict[str, JsonValue],  # pm-lint: allow PM004 -- HTTP control request body
    ) -> RuntimeState | DeliveryState | StimulusReceipt | SnapshotReceipt | TraceReceipt:
        """Apply one control and return its JSON response.

        ``body`` is a decoded request body, so an absent or mistyped field
        raises here rather than becoming a default. A control that a stopping
        or busy runtime cannot honour raises ``ValueError``.
        """
        app = self.app
        with self._lock:
            if path in {"/control/state", "/control/status", "/control/start"}:
                return self.state()
            if path in {"/control/shutdown", "/control/stop"}:
                app.request_shutdown()
                return self.state()
            if path == "/control/freeze":
                with app.shutdown_request_lock:
                    if app.draining.is_set() or app.shutdown_event.is_set():
                        raise ValueError("A stopping application cannot enter a checkpoint barrier.")
                    self.intake.close_intake()
                    with app.admission_lock:
                        app.frozen.set()
                        for module in app.modules:
                            module.draining = True
                            module.mailbox.wake()
                return self.state()
            if path == "/control/thaw":
                with app.shutdown_request_lock:
                    if app.draining.is_set() or app.shutdown_event.is_set():
                        raise ValueError("A stopping application cannot reopen intake.")
                    if app.frozen.is_set():
                        with app.admission_lock:
                            for module in app.modules:
                                module.draining = False
                                module.mailbox.wake()
                            app.frozen.clear()
                        self.intake.open_intake()
                return self.state()
            if path == "/control/delivery":
                with self.ghost.lock:
                    record = next((item for item in self.ghost.all_features
                                   if item.metadata.get("delivery_id") == body["delivery_id"]), None)
                    history = self.ghost.causal_history
                    accepted = False
                    if record is not None and history is not None:
                        producer = history.producers().get(record.id)
                        accepted = record.id in history.baseline_ids or producer in history.accepted()
                return DeliveryState(present=record is not None, accepted=accepted,
                                     record_id=record.id if record is not None else None)
            if path in {"/control/advance", "/control/advance-step", "/control/send", "/control/snapshot"}:
                if app.draining.is_set() or app.shutdown_event.is_set():
                    raise ValueError("PM is shutting down.")

            if path in {"/control/advance", "/control/advance-step"}:
                timestamp = body["to"]
                if not isinstance(timestamp, str):
                    raise ValueError("Clock advancement requires an ISO timestamp.")
                self.advance_clock(timestamp)
                return self.state()
            if path == "/control/send":
                stimulus = OperatorStimulus.model_validate(body)
                if stimulus.ts:
                    self.advance_clock(stimulus.ts)
                stimulus_id = self.intake.submit(stimulus.text, speaker=stimulus.speaker, delivery_id=stimulus.delivery_id)
                return StimulusReceipt(status="accepted", stimulus_id=stimulus_id)
            if path == "/control/snapshot":
                name = body["name"]
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
                    raise ValueError("Use a simple snapshot name.")
                if not app.idle():
                    raise ValueError("The runtime must be idle before a snapshot.")
                root = self.runtime_root / "snapshots"
                target = root / f"{name}.pickle"
                if target.exists():
                    raise ValueError("This snapshot name already exists. Choose a new name.")
                with self.ghost.lock:
                    write_checkpoint(target, self.ghost.checkpoint_data())
                return SnapshotReceipt(name=name, path=str(target), clock=vclock.now_aware().isoformat())
            if path.startswith("/control/trace/"):
                record_id = int(path.rsplit("/", 1)[1])
                with self.ghost.lock:
                    record = self.ghost.get_knoxel_by_id(record_id)
                    if record is None:
                        raise ValueError("Unknown output ID.")
                    history = self.ghost.causal_history
                    transaction = None
                    if history is not None:
                        producer = history.producers().get(record_id)
                        if producer is not None:
                            transaction = history.transactions[producer].model_dump(mode="json")  # pm-lint: allow PM007 -- HTTP trace response
                    return TraceReceipt(record=record.as_payload(include_embedding=False), transaction=transaction,
                        calls=[], call_linkage="unavailable: contributor gateway IDs are not supplied")
            raise ValueError(f"Unsupported runtime control: {path}")


class OperatorApplication(Application):
    """An assembled runtime has one operator for its store, input channel and snapshots."""

    def close_shutdown_intake(self) -> None:
        self.runtime_control.intake.close_intake()

    def bind_operator(self, ghost: MemoryStore, intake: OperatorIntake, runtime_root: Path) -> None:
        self.ghost = ghost
        self.runtime_root = runtime_root
        self.runtime_control = RuntimeControl(self, ghost, intake, runtime_root)
