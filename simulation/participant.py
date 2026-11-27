"""Host one real PM process behind the shared public and control APIs.

A participant owns one persona home: its process lock, the assembled application,
the virtual clock it resumes, the queued environment observations, and the bus
telemetry it appends. ``/control/*`` paths mirror the contributor control API and
add the simulation's clock, environment, snapshot, and restore steps; ``/v1/*``
paths are the ordinary public API that the harness and a browser use.

Durability: a snapshot is written together with its clock receipt, so a restore
has both the ghost and the timer deadlines of one instant. An environment event
waits in this process until its world time arrives and is then delivered as
ordinary cognition input. This module supplies no simulated cognition and no
fixed replies; the package overview is in ``simulation/__init__.py``.

A participant may hold one Minecraft body. ``/control/start`` names its MCP
endpoint, and this process writes that endpoint into its own model configuration
before the application starts, so the body section is typed and validated like
any other operator configuration. The body works on demand: the simulation
advances a virtual clock between scenes while a Minecraft server keeps real
time, so a world poll on the virtual clock would fire hundreds of times inside
one scene advance. Body workflows therefore read the world when they run.
"""
from __future__ import annotations

from pm.utils.operating_system import OperatingSystem
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import threading
import time
from collections.abc import Sequence
from types import FrameType
from typing import TYPE_CHECKING

from pydantic import BaseModel, JsonValue, TypeAdapter

from pm import vclock
from pm.app.character_home import CharacterHome
from pm.app.control import OperatorStimulus
from pm.app.web_chat import WebChatServer
from pm.bus.envelopes import Envelope
from pm.host.diagnostics import configure_logging, close_logging
from pm.host.durable_files import ProcessLock, atomic_json
from pm.llm.prompt_log import configure_root
from pm.memory.reconsolidation.coordination import MemoryWorkProgress
from pm.memory.reconsolidation.publication import MemoryUnitReceipt
from pm.memory.store import MemoryStore
from pm.records.media import RecordImage
from simulation.client import ParticipantFailure
from workflows.interaction.records import MainSelectionMetadata, interaction_receipts
from workflows.interaction.schemas import InteractionOutcome

if TYPE_CHECKING:
    from workflows.companion import CompanionApplication


class PreparedMemoryStatus(BaseModel):
    commit_id: str
    receipt: MemoryUnitReceipt


class MemoryWorkStatus(BaseModel):
    sleep_action_id: int | None
    active: MemoryWorkProgress | None
    ready: list[PreparedMemoryStatus]

    @property
    def busy(self) -> bool:
        return self.sleep_action_id is not None or self.active is not None or bool(self.ready)


class ParticipantClockStart(BaseModel):
    clock_start: str
    utc_offset_hours: float | None = None


class SourceCycleStatus(BaseModel):
    interaction_outcome: InteractionOutcome | None = None
    interaction_selection_id: str | None = None
    main_tick_id: int | None = None

    @property
    def complete(self) -> bool:
        return self.interaction_selection_id is not None or self.main_tick_id is not None


def source_cycle_status(store: MemoryStore, source_id: int) -> SourceCycleStatus:
    """Read the first accepted decision that owned this public input."""
    view = store.snapshot(capture=False)
    receipt = interaction_receipts(view).get(source_id)
    if receipt is not None:
        return SourceCycleStatus(interaction_outcome=receipt.outcome, interaction_selection_id=receipt.selection_id)
    for turn in view.gwt_turns.values():
        if turn.slot_id != "main" or source_id not in turn.stimulus_ids or turn.decision_id is None:
            continue
        selected = MainSelectionMetadata.model_validate(view.get(turn.decision_id).metadata)
        if source_id not in selected.interaction_source_ids:
            return SourceCycleStatus(main_tick_id=turn.tick_id)
    return SourceCycleStatus()


class Participant:
    """Own one persona process, its clock, and serialized lifecycle controls."""

    def __init__(self, home: Path, config: str, port: int, failure_file: Path) -> None:
        self.home: Path = home.resolve()
        self.failure_file: Path = failure_file
        self.config: str = config
        self.app: CompanionApplication | None = None
        self.error: BaseException | None = None
        self.starting: bool = False
        self.closed: threading.Event = threading.Event()
        self._lock = threading.RLock()
        self._telemetry_lock = threading.Lock()
        self._home_lock: ProcessLock = ProcessLock(self.home / ".simulation-process.lock")
        self._home_lock.acquire()
        self.server: WebChatServer = WebChatServer(None, self.submit, port=port, on_error=self.fail, control=self.control)
        self.record_lifecycle: bool = False
        # The one Minecraft body this participant holds, or None for no body.
        self.minecraft_mcp_url: str | None = None
        self.ready: threading.Event = threading.Event()
        # Observations whose virtual world time has not arrived yet.
        self.env_pending: list[dict[str, JsonValue]] = []
        self.output_cursor: str = "0"
        # Outputs already produced by an advance that a later request still needs.
        self.carry: list[dict[str, JsonValue]] = []

    def clock_receipt(self) -> dict[str, JsonValue]:
        """Capture elapsed time, timer deadlines, and pending input beside a ghost.

        The receipt is stored next to a snapshot, so a restore can resume the same
        virtual instant instead of restarting the clock. ``timers`` holds each
        module's next due tick in virtual seconds.
        """
        return {"clock": vclock.active().state(), "environment": self.env_pending,
                "output_cursor": self.output_cursor, "carry": self.carry,
                "timers": {item.module_id: item.next_tick_seconds for item in self.app.modules}}

    def save_clock(self) -> None:
        """Keep timer deadlines for restart without resetting elapsed time."""
        if self.app is not None and vclock.active() is not None:
            atomic_json(self.home / "runtime" / "clock.json", self.clock_receipt())

    def submit(self, text: str, *, session: str = "", speaker: str | None = None,
               images: Sequence[RecordImage] = (), delivery_id: str = "", entity_ref: str = "") -> str:
        """Accept partner input and return its correlation ID. Repeat deliveries reuse the exchange."""
        if self.app is None or self.starting:
            raise RuntimeError("The participant is not ready.")
        return self.app.runtime_control.intake.submit(text, session=session, speaker=speaker,
            images=images, delivery_id=delivery_id, entity_ref=entity_ref)

    def fail(self, error: BaseException) -> None:
        """Retain a failure and stop runtime work. Keep control evidence readable."""
        self.error = error
        self.save_failure(error)
        if self.app is not None:
            self.app.on_error(error)

    def save_failure(self, error: BaseException) -> None:
        """Retain the cause for both the attempt supervisor and the persona debugger."""
        receipt = ParticipantFailure(error=repr(error))
        receipt.save(self.failure_file)
        receipt.save(self.home / "runtime" / "participant-failure.json")

    def observe(self, envelope: Envelope, recipient: str) -> None:
        """Append full bus delivery evidence under the owning persona home.

        Every delivered envelope is recorded with both wall-clock and world time,
        so the owner can reconstruct routing after the run. The append holds its
        own lock, so concurrent deliveries cannot interleave one JSON line.
        """
        path = self.home / "runtime" / "telemetry.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"real_time": datetime.now(timezone.utc).isoformat(), "world_time": vclock.now_aware().isoformat(),
               "recipient": recipient, "envelope": json.loads(envelope.to_json())}
        with self._telemetry_lock, path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _start(self, request: ParticipantClockStart) -> None:
        try:
            from workflows.companion import build_app
            saved_clock = self.home / "runtime" / "clock.json"
            saved = json.loads(saved_clock.read_text(encoding="utf-8")) if saved_clock.exists() else None
            start = datetime.fromisoformat(request.clock_start)
            if saved and datetime.fromisoformat(saved["clock"]["now"]) == start:
                clock = vclock.VirtualClock.from_state(saved["clock"])
            else:
                saved = None
                clock = vclock.VirtualClock(start, request.utc_offset_hours)
            vclock.install(clock)
            configure_root(self.home / "logs" / "prompts")
            configure_logging(self.home)
            store = CharacterHome(self.home).open_store()
            # A body starts only when this participant declared its endpoint, and
            # then the configuration names it and the application probes it
            # before the first cycle.
            app = build_app(None, self.home / "runtime", ghost=store, cognition="real",
                            config_path=self.config, enable_minecraft=self.minecraft_mcp_url is not None,
                            emit_reply=lambda text: None,
                            bus_observer=self.observe, record_lifecycle=self.record_lifecycle)
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
                        module.next_tick_seconds = saved["timers"][module.module_id]
            app.run()
        except BaseException as error:
            self.fail(error)
        finally:
            self.starting = False
            self.ready.set()

    def control(self, path: str, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
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
                record_lifecycle = body.get("record_lifecycle", False)
                if not isinstance(record_lifecycle, bool):
                    raise ValueError("record_lifecycle must be a boolean.")
                clock_start = ParticipantClockStart.model_validate(body)
                self.record_lifecycle = record_lifecycle
                base_url = self._text_option(body, "base_url")
                minecraft_mcp_url = self._text_option(body, "minecraft_mcp_url")
                if base_url is not None or minecraft_mcp_url is not None:
                    from simulation.source import SimulationConfigOverrides, copy_simulation_config
                    selected = self.home / "runtime" / "model-config.yaml"
                    overrides = SimulationConfigOverrides(model_url=base_url, body_url=minecraft_mcp_url)
                    copy_simulation_config(Path(self.config), selected, overrides)
                    self.config = str(selected)
                self.minecraft_mcp_url = minecraft_mcp_url
                self.starting = True
                threading.Thread(target=self._start, args=(clock_start,), name="participant-start", daemon=False).start()
                return {"status": "starting", "home": str(self.home)}
            if path in {"/control/state", "/control/status"} and (self.app is None or self.starting or self.error):
                return {"status": "failed" if self.error else "starting" if self.starting else "ready",
                        "idle": False, "error": repr(self.error) if self.error else None}
            if self.app is None:
                raise ValueError("Start this participant first.")
            if path == "/control/source-cycle":
                source_id = TypeAdapter(int).validate_python(body["source_id"])
                status = source_cycle_status(self.app.runtime_control.ghost, source_id)
                return TypeAdapter(dict[str, JsonValue]).validate_python(  # pm-lint: allow PM004 -- HTTP response shape
                    status.model_dump(mode="json"))  # pm-lint: allow PM007 -- HTTP response serialization
            if path == "/control/memory-work":
                memory = self.app.memory_work
                with memory.store.lock:
                    sleep = memory.sleep
                    worker = memory.worker
                    ready = [PreparedMemoryStatus(commit_id=item.unit.batch.commit_id, receipt=item.unit.receipt)
                             for item in memory.ready]
                    status = MemoryWorkStatus(sleep_action_id=sleep.action_record_id if sleep else None,
                                              active=worker.progress if worker else None, ready=ready)
                    return TypeAdapter(dict[str, JsonValue]).validate_python(  # pm-lint: allow PM004 -- HTTP response shape
                        status.model_dump(mode="json"))  # pm-lint: allow PM007 -- HTTP response serialization
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
                events, self.output_cursor = public_events(self.app.runtime_control.ghost, self.output_cursor)
                outputs = [*self.carry, *[{"output_id": event["output_id"], "virtual_ts": event["timestamp"],
                    "text": event["content"], "kind": "reply" if event["correlation_id"] else "unprompted",
                    "call_ids": [], "call_linkage": "unavailable: contributor gateway IDs are not supplied"}
                    for event in events]]
                self.carry = []
                self.save_clock()
                return {"outputs": outputs}
            if path == "/control/send":
                request = OperatorStimulus.model_validate(body)
                if request.ts:
                    prior = self.control("/control/advance", {"to": request.ts})["outputs"]  # pm-lint: allow PM005 -- simulator control request
                    self.carry.extend(prior)
                stimulus = self.submit(request.text, speaker=request.speaker, delivery_id=request.delivery_id)
                return {"ok": True, "stimulus_id": stimulus}
            if path == "/control/snapshot":
                controller = self.app.runtime_control
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
                clock_start = ParticipantClockStart(clock_start=saved["clock"]["now"])
                threading.Thread(target=self._start, args=(clock_start,),
                                 name="participant-restore", daemon=False).start()
                return {"status": "starting", "name": name}
            controller = self.app.runtime_control
            if path in {"/control/state", "/control/status"}:
                state = controller.state()
                ghost = controller.ghost
                cognition = self.app.cognitive_control
                if cognition is None:
                    raise RuntimeError("A running simulation participant requires real cognitive control.")
                deadlines = [datetime.fromisoformat(item["ts"]) for item in self.env_pending]
                deadline = state["next_due"]
                if deadline is not None:
                    deadlines.append(datetime.fromisoformat(deadline))
                response = TypeAdapter(dict[str, JsonValue]).validate_python(state)
                response["next_due"] = min(deadlines).isoformat() if deadlines else None
                response["user_entity_id"] = ghost.get_entity_id(ghost.get_user_name())
                response["virtual_ts"] = state["clock"]
                response["cycles"] = len(ghost.states)
                response["waiting"] = cognition.waiting()
                response["injects"] = {}
                response["mental_state"] = ghost.states[-1].mental_state.model_dump(mode="json") if ghost.states else {}  # pm-lint: allow PM007 -- simulator state response
                response["module_errors_total"] = int(self.app.error is not None)
                return response
            receipt = controller(path, body)
            if path == "/control/advance-step":
                self.deliver_environment()
                self.save_clock()
            return TypeAdapter(dict[str, JsonValue]).validate_python(receipt)

    @staticmethod
    def _text_option(body: dict[str, JsonValue], key: str) -> str | None:
        """Read one optional start option that must be text when it is present.

        The harness sends these values, and a wrong type would otherwise reach
        the model configuration builder as an unnamed object.
        """
        value = body.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"The start option {key} must be text or null.")
        return value

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

    def environment(self, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Deliver observed environment events through ordinary cognition input.

        Each event becomes an ordinary ``WorldEvent`` feature and a bus message,
        so cognition treats it exactly like any other observed fact. Returns the
        accepted record IDs in delivery order.
        """
        from pm.records.memory_records import new_feature
        from pm.records.enums import FeatureType, SourceKind
        from pm.bus.envelopes import new_envelope
        if self.app is None:
            raise RuntimeError("A participant must be running to receive environment events.")
        ids = []
        for item in body["events"]:
            text = item if isinstance(item, str) else item["text"]
            record = new_feature(self.app.runtime_control.ghost, text, FeatureType.WorldEvent, source="environment",
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
                    self.save_failure(self.error)
            self.server.stop()
        finally:
            close_logging()
            self._home_lock.close()


def main() -> int:
    """Run one controlled participant. Process errors remain visible to its owner."""
    OperatingSystem.install_child_cleanup()
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()
    participant = Participant(args.home, args.config, args.port, args.ready_file.with_suffix(".failure.json"))
    def stop(signum: int, frame: FrameType | None) -> None:
        participant.closed.set()
    OperatingSystem.register_stop_signals(stop)
    participant.server.start()
    atomic_json(args.ready_file, {"url": participant.server.url, "home": str(participant.home), "pid": OperatingSystem.process_id()})
    try:
        while not participant.closed.wait(.1):
            if participant.app is not None and participant.app.shutdown_event.is_set() and not participant.starting:
                break
    finally:
        participant.close()
    return 1 if participant.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
