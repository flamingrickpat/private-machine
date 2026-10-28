"""Host one real PM process behind the shared public and control APIs.

The simulation owns process lifecycle. Companion behavior runs through the
ordinary assembly. This module supplies no simulated cognition or fixed replies.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time

from pm import vclock
from pm.app.character_home import CharacterHome
from pm.app.control import RuntimeControl
from pm.app.web_chat import WebChatServer
from pm.host.diagnostics import configure_logging, close_logging
from pm.host.durable_files import ProcessLock, atomic_json
from pm.llm.prompt_log import configure_root


class Participant:
    """Own one persona process, its clock, and serialized lifecycle controls."""

    def __init__(self, home: Path, config: str, port: int):
        self.home = home.resolve()
        self.config = config
        self.app = None
        self.error = None
        self.starting = False
        self.closed = threading.Event()
        self._lock = threading.RLock()
        self._telemetry_lock = threading.Lock()
        self._home_lock = ProcessLock(self.home / ".simulation-process.lock")
        self._home_lock.acquire()
        self.server = WebChatServer(None, self.submit, port=port, on_error=self.fail, control=self.control)
        self.clock_start = None
        self.ready = threading.Event()
        self.env_pending: list[dict] = []
        self.output_cursor = "0"
        self.carry: list[dict] = []

    def clock_receipt(self) -> dict:
        """Capture elapsed time and timer deadlines beside a drained ghost."""
        return {"clock": vclock.active().state(), "environment": self.env_pending,
                "output_cursor": self.output_cursor, "carry": self.carry,
                "timers": {item.module_id: item._next_tick for item in self.app.modules}}

    def save_clock(self) -> None:
        """Keep timer deadlines for restart without resetting elapsed time."""
        if self.app is not None and vclock.active() is not None:
            atomic_json(self.home / "runtime" / "clock.json", self.clock_receipt())

    def submit(self, text: str, **options) -> str:
        """Deliver ordinary entity input to this process's real companion."""
        if self.app is None or self.starting:
            raise RuntimeError("The participant is not ready.")
        return self.app.user_input.submit(text, **options)

    def fail(self, error: BaseException) -> None:
        """Retain a failure and stop runtime work. Keep control evidence readable."""
        self.error = error
        atomic_json(self.home / "runtime" / "participant-failure.json", {"error": repr(error)})
        if self.app is not None:
            self.app.on_error(error)

    def observe(self, envelope, recipient: str) -> None:
        """Append full bus delivery evidence under the owning persona home."""
        path = self.home / "runtime" / "telemetry.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"real_time": datetime.now(timezone.utc).isoformat(), "world_time": vclock.now_aware().isoformat(),
               "recipient": recipient, "envelope": json.loads(envelope.to_json())}
        with self._telemetry_lock, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _start(self, body: dict) -> None:
        try:
            from workflows.companion import build_app
            self.clock_start = body["clock_start"]
            saved_clock = self.home / "runtime" / "clock.json"
            saved = json.loads(saved_clock.read_text(encoding="utf-8")) if saved_clock.exists() else None
            start = datetime.fromisoformat(self.clock_start)
            if saved and datetime.fromisoformat(saved["clock"]["now"]) == start:
                clock = vclock.VirtualClock.from_state(saved["clock"])
            else:
                saved = None
                clock = vclock.VirtualClock(start, body.get("utc_offset_hours"))
            vclock.install(clock)
            configure_root(self.home / "logs" / "prompts")
            configure_logging(self.home)
            store = CharacterHome(self.home).open_store()
            app = build_app(None, self.home / "runtime", ghost=store, cognition="real",
                            config_path=self.config, enable_minecraft=False, world_poll_seconds=None,
                            emit_reply=lambda text: None, bus_observer=self.observe, record_lifecycle=False)
            self.app = app
            self.server.store = store
            from pm.app.transport import event_cursor
            self.output_cursor = event_cursor(store)
            if saved:
                self.env_pending = saved.get("environment", [])
                self.output_cursor = saved.get("output_cursor", self.output_cursor)
                self.carry = saved.get("carry", [])
                for module in app.modules:
                    if module.module_id in saved["timers"]:
                        module._next_tick = saved["timers"][module.module_id]
            app.run()
        except BaseException as error:
            self.fail(error)
        finally:
            self.starting = False
            self.ready.set()

    def control(self, path: str, body: dict) -> dict:
        """Handle contributor-compatible process controls and runtime controls.

        Start returns during loading. State reports readiness and failures.
        Dependent controls wait for loading without holding the control lock.
        Restore stops the prior runtime before opening its named checkpoint.
        """
        if self.starting and path not in {"/control/start", "/control/state", "/control/status",
                                          "/control/shutdown", "/control/stop"}:
            self.ready.wait()
        with self._lock:
            if path in {"/control/shutdown", "/control/stop"} and self.app is None:
                self.closed.set()
                return {"status": "stopped"}
            if path == "/control/start":
                if body.get("cognition", "real") != "real":
                    raise ValueError("Simulation participants require real cognition.")
                if self.app is not None or self.starting:
                    return self.control("/control/state", {})
                if body.get("base_url"):
                    from pm.utils import yaml_util
                    config = yaml_util.safe_load(Path(self.config).read_text(encoding="utf-8"))
                    config["chat"]["Default"]["base_url"] = body["base_url"].rstrip("/")
                    selected = self.home / "runtime" / "model-config.yaml"
                    selected.parent.mkdir(parents=True, exist_ok=True)
                    selected.write_text(yaml_util.safe_dump(config), encoding="utf-8")
                    self.config = str(selected)
                self.starting = True
                threading.Thread(target=self._start, args=(body,), name="participant-start", daemon=False).start()
                return {"status": "starting", "home": str(self.home)}
            if path in {"/control/state", "/control/status"} and (self.app is None or self.starting or self.error):
                return {"status": "failed" if self.error else "starting" if self.starting else "ready",
                        "idle": False, "error": repr(self.error) if self.error else None}
            if self.app is None:
                raise ValueError("Start this participant first.")
            if path == "/control/env":
                for item in body["events"]:
                    item = {"text": item} if isinstance(item, str) else dict(item)
                    timestamp = datetime.fromisoformat(item.get("ts", vclock.now_aware().isoformat()))
                    item["ts"] = vclock.active()._utc(timestamp).isoformat()
                    self.env_pending.append(item)
                self.save_clock()
                return {"ok": True, "pending": len(self.env_pending)}
            if path == "/control/inject":
                raise ValueError("Experimental injection controls are deferred to a separate work item.")
            if path == "/control/advance":
                self.advance(body["to"])
                from pm.app.transport import public_events
                events, self.output_cursor = public_events(self.app.ghost, self.output_cursor)
                outputs = [*self.carry, *[{"output_id": event["output_id"], "virtual_ts": event["timestamp"],
                    "text": event["content"], "kind": "reply" if event["correlation_id"] else "unprompted",
                    "call_ids": [], "call_linkage": "unavailable: contributor gateway IDs are not supplied"}
                    for event in events]]
                self.carry = []
                self.save_clock()
                return {"outputs": outputs}
            if path == "/control/send":
                if body.get("ts"):
                    prior = self.control("/control/advance", {"to": body["ts"]})["outputs"]
                    self.carry.extend(prior)
                stimulus = self.submit(body["text"], speaker=body.get("speaker"),
                                       delivery_id=body.get("delivery_id", ""))
                return {"ok": True, "stimulus_id": stimulus}
            if path == "/control/snapshot":
                controller = RuntimeControl(self.app)
                frozen = self.app.frozen.is_set()
                if not frozen:
                    controller("/control/freeze", {})
                try:
                    self.quiet()
                    result = controller(path, body)
                    atomic_json(Path(result["path"]).with_suffix(".json"), self.clock_receipt())
                    return {"ok": True, **result}
                finally:
                    if not frozen:
                        controller("/control/thaw", {})
            if path == "/control/restore":
                name = body["name"]
                if Path(name).name != name or name in {".", ".."}:
                    raise ValueError("Use a simple snapshot name.")
                snapshot = self.home / "runtime" / "snapshots" / f"{name}.pickle"
                metadata = snapshot.with_suffix(".json")
                if not metadata.exists():
                    raise ValueError("The named snapshot has no clock receipt.")
                self.starting = True
                self.ready.clear()
                self.app.request_shutdown()
                self.app.shutdown_event.wait()
                self.app.join()
                self.app.raise_if_error()
                saved = json.loads(metadata.read_text(encoding="utf-8"))
                shutil.copyfile(snapshot, self.home / "ghost.pickle")
                atomic_json(self.home / "runtime" / "clock.json", saved)
                self.app = None
                self.server.store = None
                self.starting = True
                threading.Thread(target=self._start, args=({"clock_start": saved["clock"]["now"]},),
                                 name="participant-restore", daemon=False).start()
                return {"status": "starting", "name": name}
            controller = RuntimeControl(self.app)
            result = controller(path, body)
            if path == "/control/advance-step":
                self.deliver_environment()
                self.save_clock()
            if path in {"/control/state", "/control/status"}:
                deadlines = [datetime.fromisoformat(item["ts"]) for item in self.env_pending]
                if result["next_due"]:
                    deadlines.append(datetime.fromisoformat(result["next_due"]))
                result["next_due"] = min(deadlines).isoformat() if deadlines else None
                result.update(virtual_ts=result["clock"], cycles=len(self.app.ghost.states),
                    waiting=self.app.cognitive_control.waiting(), injects={},
                    mental_state=self.app.ghost.states[-1].mental_state.model_dump(mode="json") if self.app.ghost.states else {},
                    module_errors_total=int(self.app.error is not None))
            return result

    def quiet(self) -> None:
        """Wait for runnable work. Parked workflows do not block the controller."""
        stable = None
        while True:
            self.app.raise_if_error()
            if self.app.idle():
                stable = stable or time.monotonic()
                if time.monotonic() - stable >= .15:
                    return
            else:
                stable = None
            time.sleep(.02)

    def deliver_environment(self) -> None:
        """Deliver queued observations only when their world time arrives."""
        due = sorted((item for item in self.env_pending
                      if datetime.fromisoformat(item["ts"]) <= vclock.now_aware()), key=lambda item: item["ts"])
        if due:
            self.environment({"events": due})
            self.env_pending = [item for item in self.env_pending if item not in due]

    def advance(self, target: str) -> None:
        """Visit semantic deadlines before returning legacy buffered outputs."""
        stop = vclock.active()._utc(datetime.fromisoformat(target))
        if stop < vclock.now_aware():
            raise ValueError("Virtual time cannot move backward. Restore a checkpoint.")
        self.deliver_environment()
        self.quiet()
        while vclock.now_aware() < stop:
            state = self.control("/control/state", {})
            due = datetime.fromisoformat(state["next_due"]) if state["next_due"] else stop
            following = min(stop, due) if due > vclock.now_aware() else stop
            self.control("/control/advance-step", {"to": following.isoformat()})
            self.quiet()

    def environment(self, body: dict) -> dict:
        """Deliver observed environment events through ordinary cognition input."""
        from pm.records.memory_records import new_feature
        from pm.records.enums import FeatureType, SourceKind
        from pm.bus.envelopes import new_envelope
        ids = []
        for item in body["events"]:
            text = item if isinstance(item, str) else item["text"]
            record = new_feature(self.app.ghost, text, FeatureType.WorldEvent, source="environment",
                                 source_kind=SourceKind.System, metadata={"origin": "environment.event"})
            ids.append(record.id)
            self.app.bus.put(new_envelope("environment", "data", "environment.event", "", {"knoxel_id": record.id}))
        return {"status": "accepted", "record_ids": ids}

    def close(self) -> None:
        """Drain runtime work, close public sockets, and release the home lock."""
        try:
            if self.starting:
                self.ready.wait()
            if self.app is not None:
                self.app.request_shutdown()
                self.app.shutdown_event.wait()
                self.app.join()
                self.save_clock()
                if self.app.error is not None:
                    self.error = self.app.error
                    atomic_json(self.home / "runtime" / "participant-failure.json", {"error": repr(self.error)})
            self.server.stop()
        finally:
            close_logging()
            self._home_lock.close()


def main() -> int:
    """Run one controlled participant. Process errors remain visible to its owner."""
    from pm.host.processes import ensure_process_cleanup
    ensure_process_cleanup()
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()
    participant = Participant(args.home, args.config, args.port)
    def stop(signum, frame):
        participant.closed.set()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    participant.server.start()
    atomic_json(args.ready_file, {"url": participant.server.url, "home": str(participant.home), "pid": os.getpid()})
    try:
        while not participant.closed.wait(.1):
            if participant.app is not None and participant.app.shutdown_event.is_set() and not participant.starting:
                break
    finally:
        participant.close()
    return 1 if participant.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
