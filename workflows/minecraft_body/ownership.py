"""One endpoint reservation from admission through the final confirmed body read."""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from threading import Event, RLock

from pydantic import BaseModel, JsonValue
from workflows.minecraft_body.journal import read_actor_journal
from workflows.minecraft_body.spatial import WorldPosition

from pm.host.durable_files import BodyBusy, atomic_json
from pm.runs.errors import TaskServiceError
from workflows.minecraft_body.lease import BODY_LEASES, BodyLeaseEvidence
from workflows.minecraft_body.mcp_client import MinecraftToolReply, MinecraftTransport
from workflows.minecraft_body.world_text import BodyReading, WorldObservation, StopObservation


@dataclass(slots=True)
class ReservedBody:
    lease: BodyLeaseEvidence
    scope: ExitStack
    claimed: bool = False
    observed: bool = False


class BodyCommandStatus(BaseModel):
    active_command: str | None
    cancellation_requested: bool


class JournalOperands(BaseModel):
    position: WorldPosition | None = None
    target: WorldPosition | None = None
    base: WorldPosition | None = None
    chest: WorldPosition | None = None


class PendingCommand(BaseModel):
    seq: int
    tool: str
    arguments: JournalOperands


class PausedTask(BaseModel):
    run_id: str
    epoch: int


class BodyOwnership:
    def __init__(self, transport: MinecraftTransport, shutdown: Event | None = None) -> None:
        self.transport = transport
        self.shutdown = shutdown if shutdown is not None else Event()
        self.endpoint = transport.settings.endpoint
        self.lock = RLock()
        self.reservation: ReservedBody | None = None
        self.resume_authorization: str | None = None
        self.recovery_path = BODY_LEASES.path(self.endpoint, ".recovery.json")
        self.pause_path = BODY_LEASES.path(self.endpoint, ".paused.json")

    def reconcile_at_startup(self, task_root: Path) -> None:
        path = BODY_LEASES.path(self.endpoint, ".json")
        if BODY_LEASES.owner(self.endpoint) is not None:
            return
        record = self.recovery_path if self.recovery_path.is_file() else path
        if not record.is_file():
            return
        previous = BodyLeaseEvidence.model_validate_json(record.read_bytes())
        if not previous.active:
            return
        with BODY_LEASES.acquire(self.endpoint, previous.owner) as replacement:
            # A second restart must still reconcile the original command if this recovery fails.
            atomic_json(self.recovery_path, previous.model_dump())
            status = self.transport.call("minecraft_status", {})
            if not status["ok"]:
                raise TaskServiceError("Cannot establish the state of the interrupted Minecraft command.")
            if BodyCommandStatus.model_validate(status["result"]).active_command:
                # The server waits for idle before it admits a replacement epoch.
                self.interrupt_command(previous.owner, previous)
            if not self.transport.claim(replacement)["ok"]:
                raise TaskServiceError("Cannot fence the interrupted Minecraft epoch.")
            try:
                final = self.read_for_reconciliation("minecraft_observe", {"include_image": False}, replacement)
                if not final["ok"]:
                    raise TaskServiceError("Cannot observe the interrupted Minecraft body.")
                facts = self.inspect_uncertain_effects(task_root / previous.owner, replacement)
                atomic_json(task_root / previous.owner / "body-restart.json", {
                    "previous_epoch": previous.epoch, "reconciled_epoch": replacement.epoch,
                    "observed_at": final["observed_at"], "observation": final["result"], "uncertain_effects": list[JsonValue](facts)})
            finally:
                if not self.transport.release(replacement)["ok"]:
                    raise TaskServiceError("Cannot confirm release of the recovered Minecraft epoch.")

            self.recovery_path.unlink()

    def owner(self) -> str | None:
        active = BODY_LEASES.owner(self.endpoint)
        if active is not None:
            return active
        paused = self.paused_task()
        return paused.run_id if paused else None

    def paused_task(self) -> PausedTask | None:
        try:
            return PausedTask.model_validate_json(self.pause_path.read_bytes())
        except FileNotFoundError:
            return None

    def retain_for_resume(self, run_id: str, epoch: int, artifact: Path, execution_epoch: int) -> None:
        with self.lock:
            token = BODY_LEASES.token(self.endpoint, run_id)
            if token.epoch != epoch:
                raise BodyBusy("The body epoch changed before its stop could be admitted.")
            paused = PausedTask(run_id=run_id, epoch=epoch)
            atomic_json(artifact / f"body-stop-request-{execution_epoch}.json", paused.model_dump())
            atomic_json(self.pause_path, paused.model_dump())

    def relinquish(self, run_id: str) -> None:
        with self.lock:
            try:
                pause = PausedTask.model_validate_json(self.pause_path.read_bytes())
            except FileNotFoundError:
                return
            if pause.run_id == run_id:
                self.pause_path.unlink()


    def authorize_resume(self, run_id: str) -> None:
        with self.lock:
            owner = self.owner()
            if owner and owner != run_id:
                raise BodyBusy(f"Minecraft body is busy with {owner}.")
            paused = self.paused_task()
            if paused is not None and paused.run_id == run_id:
                self.resume_authorization = run_id

    def reserve(self, run_id: str) -> None:
        if self.recovery_path.is_file():
            raise BodyBusy("Minecraft restart reconciliation is pending.")
        current = self.owner()
        if current and current != run_id:
            raise BodyBusy(f"Minecraft body is busy with {current}. Check status, wait or terminate that run.")
        with self.lock:
            if self.reservation is not None and self.reservation.lease.owner == run_id:
                return
            scope = ExitStack()
            try:
                lease = scope.enter_context(BODY_LEASES.acquire(self.endpoint, run_id))
            except BodyBusy as error:
                raise BodyBusy(f"Minecraft body is busy with {self.owner()}. Check status, wait or terminate that run.") from error
            try:
                if self.pause_path.is_file():
                    pause = PausedTask.model_validate_json(self.pause_path.read_bytes())
                    if pause.run_id != run_id or self.resume_authorization != run_id:
                        raise BodyBusy(f"Minecraft is paused for {pause.run_id}; resume or terminate the retained task.")
            except BaseException:
                scope.close()
                raise
            self.reservation = ReservedBody(lease, scope)

    def activate(self, run_id: str) -> None:
        with self.lock:
            slot = self.reservation
            if slot is None or slot.lease.owner != run_id:
                raise BodyBusy(f"Minecraft body belongs to {self.owner()}.")
            if slot.observed:
                return
            if not slot.claimed:
                if not self.transport.claim(slot.lease)["ok"]:
                    raise TaskServiceError("Minecraft refused the replacement epoch.")
                slot.claimed = True
            observed = self.transport.fenced_call("minecraft_observe", {"include_image": False}, slot.lease)
            if not observed["ok"]:
                raise TaskServiceError("Minecraft body could not be observed after the epoch changed.")
            slot.observed = True
            if self.pause_path.is_file():
                paused = PausedTask.model_validate_json(self.pause_path.read_bytes())
                if paused.run_id == run_id and paused.epoch < slot.lease.epoch:
                    self.pause_path.unlink()
                    self.resume_authorization = None


    def finish(self, run_id: str, interrupted: bool, artifact: Path) -> None:
        with self.lock:
            slot = self.reservation
            if slot is None or slot.lease.owner != run_id:
                return
            try:
                if interrupted:
                    self.stop(run_id, slot.lease.epoch, artifact)
                if slot.claimed and not self.transport.release(slot.lease)["ok"]:
                    raise TaskServiceError(f"Minecraft release unconfirmed for {run_id}.")
            except Exception:
                # The next startup must reconcile before another run can enter.
                atomic_json(self.recovery_path, slot.lease.model_dump())
                raise
            finally:
                slot.scope.close()
                self.reservation = None

    def inspect_uncertain_effects(self, artifact: Path, lease: BodyLeaseEvidence) -> list[str]:
        from workflows.minecraft_body.gateway import READ_TOOLS
        from math import floor
        facts: list[str] = []
        for journal in artifact.rglob("actor-journal.jsonl"):
            rows = read_actor_journal(journal)
            finished = {row["seq"] for row in rows if row["event"] == "result"}
            for row in rows:
                if row["event"] != "dispatch" or row["seq"] in finished or row["tool"] in READ_TOOLS:
                    continue
                command = PendingCommand.model_validate(row)
                facts.append(f"{journal.relative_to(artifact)} #{command.seq} {command.tool}: original outcome uncertain")
                for point in (command.arguments.position, command.arguments.target, command.arguments.base, command.arguments.chest):
                    if point is None:
                        continue
                    cell: dict[str, JsonValue] = {"x": floor(point.x), "y": floor(point.y), "z": floor(point.z)}
                    read = self.read_for_reconciliation("minecraft_inspect_block", {"position": cell}, lease)
                    block = read["result"]
                    if not read["ok"] and not (isinstance(block, dict) and block.get("block_name") == "air"
                                               and block.get("reason") == "not_found"):
                        raise TaskServiceError(f"Cannot reconcile {command.tool} at {cell}: {block}")
                    facts.append(f"Cell {cell}: {block}")
        return facts

    def read_for_reconciliation(self, name: str, arguments: dict[str, JsonValue], lease: BodyLeaseEvidence) -> MinecraftToolReply:
        if name not in {"minecraft_status", "minecraft_inspect_block", "minecraft_observe"}:
            raise ValueError(f"Reconciliation requires a read-only body operation: {name}")
        attempt = 0
        while True:
            attempt += 1
            try:
                reading = self.transport.fenced_call(name, arguments, lease)
                result = reading["result"]
                if reading["ok"] or not isinstance(result, dict) or result.get("reason") not in {"not_connected", "not_spawned"}:
                    return reading
                reason = str(result["reason"])
            except TaskServiceError as error:
                if self.shutdown.is_set():
                    raise
                reason = str(error)
            logging.getLogger(__name__).warning("Minecraft %s read %d for %s: %s", name, attempt, lease.owner, reason)
            if self.shutdown.wait(min(8.0, float(attempt))):
                raise TaskServiceError(f"Shutdown interrupted the {name} read for {lease.owner}: {reason}")

    def interrupt_command(self, run_id: str, lease: BodyLeaseEvidence) -> None:
        try:
            stopped = self.transport.fenced_call("minecraft_stop", {"scope": "both"}, lease)
            if stopped["ok"]:
                return
            result = stopped["result"]
        except TaskServiceError:
            logging.getLogger(__name__).warning("Stop delivery uncertain for %s. Checking the worker.", run_id)
        else:
            if not isinstance(result, dict) or result.get("reason") != "stop_pending":
                raise TaskServiceError(f"Minecraft stop unconfirmed for {run_id}: {result}")
        while True:
            status = self.read_for_reconciliation("minecraft_status", {}, lease)
            if not status["ok"]:
                raise TaskServiceError(f"Cannot reconcile the pending stop for {run_id}: {status['result']}")
            command = BodyCommandStatus.model_validate(status["result"])
            if command.active_command is None:
                return
            if not command.cancellation_requested:
                # The server confirmed an active command with no pending interrupt.
                try:
                    self.transport.fenced_call("minecraft_stop", {"scope": "both"}, lease)
                except TaskServiceError:
                    logging.getLogger(__name__).warning("Stop delivery uncertain for %s. Checking again.", run_id)
            logging.getLogger(__name__).info("Waiting for %s to stop for run %s", command.active_command, run_id)
            if self.shutdown.wait(0.25):
                raise TaskServiceError(f"Shutdown interrupted confirmation of {command.active_command} for {run_id}.")

    def stop(self, run_id: str, epoch: int, artifact: Path) -> StopObservation:
        path = artifact / f"body-stop-{epoch}.json"
        with self.lock:
            if path.is_file():
                return StopObservation.model_validate_json(path.read_bytes())
            slot = self.reservation
            if slot is None or slot.lease.owner != run_id or slot.lease.epoch != epoch:
                raise TaskServiceError(f"Body epoch {epoch} ended without confirmed stop evidence.")
            self.activate(run_id)
            self.interrupt_command(run_id, slot.lease)
            with BODY_LEASES.command(self.endpoint):
                acknowledged = datetime.now(timezone.utc)
                reconciliation = self.inspect_uncertain_effects(artifact, slot.lease)
                final = self.read_for_reconciliation("minecraft_observe", {"include_image": False}, slot.lease)
                if not final["ok"]:
                    raise TaskServiceError(f"Final Minecraft observation unavailable for {run_id}.")
                reading = BodyReading(observed_at=datetime.fromisoformat(final["observed_at"]),
                                      observation=WorldObservation.model_validate(final["result"]))
            receipt = StopObservation(run_id=run_id, epoch=epoch, acknowledged_at=acknowledged, reading=reading,
                                      uncertain_effects=tuple(reconciliation))
            atomic_json(path, receipt.model_dump(mode="json"))
            return receipt
