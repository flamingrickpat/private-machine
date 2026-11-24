"""Fence each body with an operating-system lease and a durable epoch."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from pm.utils.operating_system import OperatingSystem
from contextlib import contextmanager
import hashlib
from pathlib import Path
import threading
import time
from typing import Iterator, Generator
from urllib.parse import urlsplit, urlunsplit
import uuid
from pm.host.durable_files import BodyBusy, ProcessLock, atomic_json, read_json


class BodyEpoch(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    owner: str
    epoch: int
    session: str


class BodyLeaseEvidence(BodyEpoch):
    endpoint: str
    pid: int
    active: bool
    acquired_at: float


class BodyEpochFloor(BaseModel):
    """Last epoch already issued by the body at a previous address."""

    epoch: NonNegativeInt


class StaleEpoch(PermissionError):
    """The command belongs to an obsolete execution session."""


def body_key(endpoint: str) -> str:
    """Normalize the local host alias and remove trailing URL separators."""
    parts = urlsplit(endpoint.rstrip("/"))
    if not parts.hostname:
        return endpoint.rstrip("/")
    host = parts.hostname.lower()
    if host in {"localhost", "::1"}:
        host = "127.0.0.1"
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return urlunsplit((parts.scheme.lower(), f"{host}:{port}", parts.path.rstrip("/"), "", ""))


class BodyLeases:
    """Share leases across threads, application instances, and local processes.

    All clients must use the same registry. PM_BODY_LEASE_ROOT selects it.
    The default registry survives workspace and worker replacement.
    The MCP server separately fences an already dispatched remote call.
    """

    def __init__(self, root: Path | None = None) -> None:
        base = Path(OperatingSystem.environment.get("LOCALAPPDATA", str(Path.home() / ".local/share")))
        self.root = Path(root or OperatingSystem.environment.get("PM_BODY_LEASE_ROOT", base / "pm-next-v2/body-leases"))
        self._lock = threading.RLock()
        self._tokens: dict[str, BodyLeaseEvidence] = {}

    def path(self, endpoint: str, suffix: str) -> Path:
        """Return a stable registry path without exposing endpoint credentials."""
        digest = hashlib.sha256(body_key(endpoint).encode()).hexdigest()
        return self.root / (digest + suffix)

    def preserve_epoch(self, endpoint: str, prior_epoch: BodyEpochFloor) -> None:
        """Carry a saved body's epoch forward before assigning its new endpoint."""
        lock = ProcessLock(self.path(endpoint, ".owner.lock")).acquire()
        try:
            path = self.path(endpoint, ".epoch-floor.json")
            previous = BodyEpochFloor.model_validate(read_json(path)).epoch if path.exists() else 0
            atomic_json(path, BodyEpochFloor(epoch=max(previous, prior_epoch.epoch)).model_dump())
        finally:
            lock.close()

    @contextmanager
    def command(self, endpoint: str, cancel: object | None = None) -> Iterator[None]:
        """Serialize local reads and actions. Cancellation stops a queued call."""
        lock = ProcessLock(self.path(endpoint, ".command.lock")).acquire(wait=True, cancel=cancel)
        try:
            yield
        finally:
            lock.close()

    @contextmanager
    def acquire(self, endpoint: str, owner: str) -> Generator[BodyLeaseEvidence, None, None]:
        """Issue one durable epoch after exclusive acquisition. Never expire it."""
        key = body_key(endpoint)
        lock = ProcessLock(self.path(endpoint, ".owner.lock")).acquire()
        path = self.path(endpoint, ".json")
        try:
            previous = read_json(path) if path.exists() else {}
            floor_path = self.path(endpoint, ".epoch-floor.json")
            floor = BodyEpochFloor.model_validate(read_json(floor_path)).epoch if floor_path.exists() else 0
            token = BodyLeaseEvidence(
                endpoint=key, owner=owner, epoch=max(previous.get("epoch", 0), floor) + 1,
                session=uuid.uuid4().hex, pid=OperatingSystem.process_id(), active=True, acquired_at=time.time(),
            )
            atomic_json(path, token.model_dump())
            with self._lock:
                self._tokens[key] = token
            yield token
        finally:
            with self._lock:
                active_epoch = self._tokens.pop(key, None)
            try:
                if active_epoch is not None:
                    atomic_json(path, {**active_epoch.model_dump(), "active": False, "released_at": time.time()})
            finally:
                lock.close()

    def token(self, endpoint: str, owner: str) -> BodyLeaseEvidence:
        """Return this process's active token, or reject a stale owner."""
        with self._lock:
            token = self._tokens.get(body_key(endpoint))
            if token is None or token.owner != owner:
                raise StaleEpoch("This session does not own the Minecraft body.")
            return token

    def validate(self, endpoint: str, token: BodyEpoch) -> None:
        """Reject an earlier session even when it reused the same owner name."""
        current = self.token(endpoint, token.owner)
        if current.epoch != token.epoch or current.session != token.session:
            raise StaleEpoch("The Minecraft execution epoch changed.")

    def owner(self, endpoint: str) -> str | None:
        """Read durable ownership. A dead process has no active lease."""
        with self._lock:
            token = self._tokens.get(body_key(endpoint))
            if token is not None:
                return token.owner
        probe = ProcessLock(self.path(endpoint, ".owner.lock"))
        try:
            probe.acquire()
        except BodyBusy:
            path = self.path(endpoint, ".json")
            return read_json(path)["owner"] if path.exists() else "pending owner"
        else:
            probe.close()
            return None


BODY_LEASES = BodyLeases()

