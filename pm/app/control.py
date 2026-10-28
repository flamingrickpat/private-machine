"""Runtime lifecycle controls shared by local chat and simulation clients."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import threading

from pm import vclock
from pm.app.transport import event_cursor
from pm.memory.checkpoint import write_checkpoint


class RuntimeControl:
    """Expose one running application without importing companion behavior.

    Snapshot names stay inside the persona runtime directory. Clock changes
    require quiescence. Restoring a process belongs to its external controller.
    """

    def __init__(self, app):
        self.app = app
        self._lock = threading.RLock()

    def state(self) -> dict:
        """Return runnable activity, lifecycle state, and the latest event cursor."""
        app = self.app
        cursor = event_cursor(app.ghost)
        return {"status": "failed" if app.error else "stopped" if app.shutdown_event.is_set()
                else "draining" if app.draining.is_set() else "frozen" if app.frozen.is_set() else "running",
                "idle": app.idle(), "tick_id": app.ghost.current_tick_id,
                "clock": vclock.now_aware().isoformat(), "virtual": vclock.active() is not None,
                "output_cursor": cursor,
                "next_due": self.next_due(),
                "error": repr(app.error) if app.error else None,
                "modules": [{"id": item.module_id, "working": item.working,
                             "queued": item.mailbox.pending(), "parked": getattr(item, "_pending", None) is not None,
                             "next_tick": item._next_tick} for item in app.modules]}

    def next_due(self) -> str | None:
        """Return the next semantic timer deadline, excluding empty polling.

        A cognitive wait suppresses private inference. Its wake deadline
        remains active. The scheduler's idle poll is not a world event.
        """
        from datetime import timedelta
        now = vclock.now_aware()
        control = getattr(self.app, "cognitive_control", None)
        if control is not None and control.waiting():
            waits = [item for item in self.app.ghost.all_features
                     if item.metadata.get("control_kind") == "inference_wait"
                     and item.metadata.get("status") == "waiting"]
            if waits:
                return datetime.fromisoformat(waits[-1].metadata["wake_at"]).replace(tzinfo=now.tzinfo).isoformat()
            return None
        due = [now + timedelta(seconds=max(0, item._next_tick - item.clock()))
               for item in self.app.modules if item._next_tick is not None and not item.draining
               and item.module_id not in {"scheduler", "cognitive_control"}]
        return min(due).isoformat() if due else None

    def __call__(self, path: str, body: dict) -> dict:
        """Apply one control. Invalid state or unknown controls raise ValueError."""
        app = self.app
        with self._lock:
            if path in {"/control/state", "/control/status", "/control/start"}:
                return self.state()
            if path in {"/control/shutdown", "/control/stop"}:
                app.request_shutdown()
                return self.state()
            if path == "/control/freeze":
                if app.draining.is_set() or app.shutdown_event.is_set():
                    raise ValueError("A stopping application cannot enter a checkpoint barrier.")
                app.user_input.close_intake()
                app.frozen.set()
                for module in app.modules:
                    module.draining = True
                    module.mailbox.wake()
                return self.state()
            if path == "/control/thaw":
                if app.draining.is_set() or app.shutdown_event.is_set():
                    raise ValueError("A stopping application cannot reopen intake.")
                if app.frozen.is_set():
                    for module in app.modules:
                        module.draining = False
                        module.mailbox.wake()
                    app.frozen.clear()
                    app.user_input.open_intake()
                return self.state()
            if path == "/control/delivery":
                record = next((item for item in app.ghost.all_features
                               if item.metadata.get("delivery_id") == body["delivery_id"]), None)
                history = app.ghost.causal_history
                producer = history.producers().get(record.id) if record is not None else None
                accepted = record is not None and (record.id in history.baseline_ids or producer in history.accepted())
                return {"present": record is not None, "accepted": accepted,
                        "record_id": record.id if record is not None else None}
            if path in {"/control/advance", "/control/advance-step"}:
                clock = vclock.active()
                if clock is None:
                    raise ValueError("This process has no virtual clock.")
                if not app.idle():
                    raise ValueError("The runtime must be idle before time advances.")
                clock.set(datetime.fromisoformat(body["to"]))
                for module in app.modules:
                    module.mailbox.wake()
                return self.state()
            if path == "/control/send":
                if body.get("ts"):
                    self("/control/advance", {"to": body["ts"]})
                stimulus_id = app.user_input.submit(body["text"], speaker=body.get("speaker"),
                                                    delivery_id=body.get("delivery_id", ""))
                return {"status": "accepted", "stimulus_id": stimulus_id}
            if path == "/control/snapshot":
                name = body["name"]
                if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
                    raise ValueError("Use a simple snapshot name.")
                if not app.idle():
                    raise ValueError("The runtime must be idle before a snapshot.")
                root = Path(app.runtime_root) / "snapshots"
                target = root / f"{name}.pickle"
                if target.exists():
                    raise ValueError("This snapshot name already exists. Choose a new name.")
                with app.ghost.lock:
                    write_checkpoint(target, app.ghost.checkpoint_data(), lambda: None)
                return {"name": name, "path": str(target), "clock": vclock.now_aware().isoformat()}
            if path.startswith("/control/trace/"):
                record_id = int(path.rsplit("/", 1)[1])
                record = app.ghost.get_knoxel_by_id(record_id)
                if record is None:
                    raise ValueError("Unknown output ID.")
                history = app.ghost.causal_history
                producer = history.producers().get(record_id)
                return {"record": record.as_payload(include_embedding=False),
                        "transaction": history.transactions[producer].model_dump(mode="json") if producer else None,
                        "calls": [], "call_linkage": "unavailable: contributor gateway IDs are not supplied"}
            raise ValueError(f"Unsupported runtime control: {path}")
