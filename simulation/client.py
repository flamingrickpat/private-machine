"""Public-protocol client and owned PM subprocess lifecycle.

``PMProcess`` owns one participant process: it launches it against the frozen
source and a virtual clock, waits for its ready file and runtime state, sends
control and chat requests over HTTP, and mirrors the public event stream into a
local queue. The engine never reads a participant's private files; public speech
arrives only through this surface.

A readiness wait and an inference wait have no harness deadline, because the real
runtime decides when it is done. A dead process or a reported runtime failure
raises with the evidence folder, so an attempt records a real failure instead of
an empty turn. The package overview is in ``simulation/__init__.py``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import IO

import httpx
from pydantic import JsonValue
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

from pm.paths import REPO_ROOT


class PMProcess:
    """Own one PM process. No companion messages bypass its public API.

    Readiness and inference have no harness time limit. A dead process or a
    reported runtime failure raises with its retained evidence path.
    """

    def __init__(self, participant_id: str, home: Path, config: str, endpoint_file: Path,
                 *, source_root: Path = REPO_ROOT) -> None:
        self.id: str = participant_id
        self.home: Path = home
        self.config: str = config
        self.endpoint_file: Path = endpoint_file
        self.source_root: Path = source_root
        self.process: subprocess.Popen[bytes] | None = None
        self.url: str = ""
        # Public speech, already decoded. The engine drains this queue and owns
        # the durable timeline; a queue element carries no ownership of state.
        self.events: queue.Queue[dict[str, JsonValue]] = queue.Queue()
        self.reader_stop: threading.Event = threading.Event()
        self.reader: threading.Thread | None = None
        self.reader_error: BaseException | None = None
        self.http: httpx.Client = httpx.Client(timeout=None, trust_env=False)
        self._log: IO[bytes] | None = None

    def launch(self, clock: str) -> None:
        """Start a hidden participant process and request real runtime startup.

        The child writes its ready file with the bound URL. This method waits for
        that file and for the runtime to leave ``starting``; both waits have no
        deadline beyond the process itself, because model loading time is real
        work. ``clock`` is the coordinated ISO start time the child must adopt.
        """
        self.endpoint_file.parent.mkdir(parents=True, exist_ok=True)
        self.endpoint_file.unlink(missing_ok=True)
        self._log = (self.endpoint_file.parent / f"{self.id}-process.log").open("ab")
        from pm.llm.env_file import env_file_path
        environment = dict(os.environ)
        environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        environment.setdefault("PM_ENV_FILE", str(env_file_path().resolve()))
        self.process = subprocess.Popen(
            [sys.executable, "-m", "simulation.participant", "--home", str(self.home), "--config", self.config,
             "--ready-file", str(self.endpoint_file.resolve())], cwd=self.source_root, stdin=subprocess.DEVNULL,
            stdout=self._log, stderr=subprocess.STDOUT, env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        while not self.endpoint_file.exists():
            self.check_process()
            time.sleep(.05)
        self.url = json.loads(self.endpoint_file.read_text(encoding="utf-8"))["url"].rstrip("/")
        self.post("/control/start", {"clock_start": clock, "cognition": "real", "character": str(self.home)})
        while self.state()["status"] == "starting":
            self.check_process()
            time.sleep(.1)

    def check_process(self) -> None:
        """Raise for process or stream failure before another dependent operation."""
        if self.process is not None and self.process.poll() is not None:
            raise RuntimeError(f"Participant {self.id} exited ({self.process.returncode}). Evidence: {self.endpoint_file.parent}")
        if self.reader_error is not None:
            raise RuntimeError(f"Participant {self.id} event stream failed.") from self.reader_error

    def post(self, path: str, body: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Send one JSON request. HTTP errors remain visible to the run."""
        response = self.http.post(self.url + path, json=body)
        response.raise_for_status()
        return response.json()

    def state(self) -> dict[str, JsonValue]:
        """Read runtime status and reject a reported engineering failure.

        The receipt carries the participant's virtual clock, ``idle``, and
        optional ``next_due``. A ``failed`` status raises, because a run that
        continued past it would record evidence the runtime never produced.
        """
        self.check_process()
        response = self.http.get(self.url + "/control/state")
        response.raise_for_status()
        result = response.json()
        if result["status"] == "failed":
            raise RuntimeError(f"Participant {self.id} failed: {result['error']}. Evidence: {self.home}")
        return result

    def listen(self, cursor: str) -> None:
        """Replay public speech after a durable cursor into a local queue.

        ``cursor`` is the receiver's last committed output position, so a resumed
        attempt replays exactly the speech it has not recorded yet. The reader is
        a non-daemon thread: ``close`` stops it before the process is reaped. A
        receive timeout is the idle poll, not a failure; a genuine close or
        exception is retained in ``reader_error`` for ``check_process``.
        """
        def read() -> None:
            try:
                with connect(self.url.replace("http:", "ws:") + f"/v1/events?after={cursor}",
                             proxy=None, max_size=None) as websocket:
                    while not self.reader_stop.is_set():
                        try:
                            value = websocket.recv(timeout=.2)
                        except TimeoutError:
                            continue
                        self.events.put(json.loads(value))
            except ConnectionClosed as error:
                if not self.reader_stop.is_set():
                    self.reader_error = error
            except BaseException as error:
                self.reader_error = error
        self.reader = threading.Thread(target=read, name=f"speech-{self.id}", daemon=False)
        self.reader.start()

    def close(self) -> None:
        """Drain approved work, then wait for normal process exit."""
        self.reader_stop.set()
        try:
            if self.process is not None and self.process.poll() is None and self.url:
                self.post("/control/shutdown", {})
                self.process.wait()
            if self.process is not None and self.process.poll() not in {None, 0}:
                raise RuntimeError(f"Participant {self.id} exited with code {self.process.returncode} during shutdown.")
        finally:
            if self.reader is not None:
                self.reader.join()
            self.http.close()
            if self._log is not None:
                self._log.close()
