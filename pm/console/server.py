"""Loopback HTTP service for instance lifecycle and recorded runtime evidence.

The operator console is one local process that starts, observes, and stops
companion instances. An :class:`Instance` owns one worker process
(``pm.console.worker``), the JSON commands it acknowledges, and its durable
``events.jsonl`` log. :class:`Console` owns the deployment definitions under a
private runtime root and the set of instances.

The service binds to loopback only and answers JSON. Every browser mutation
must carry the console's own origin, because another page could otherwise post
to loopback. A running worker reports its own state; the console never claims
progress it did not observe. The event log is the evidence trail: it holds the
worker's events, commands, protocol errors, and lifecycle transitions.

Reading map: :class:`Instance` is the process and log owner, :class:`Console`
is the deployment table, and :func:`serve` is the HTTP boundary.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import NotRequired, TypedDict
from urllib.parse import urlparse, parse_qs
from uuid import uuid4

from pydantic import JsonValue

from pm.paths import REPO_ROOT

ROOT = REPO_ROOT


class InstanceSettings(TypedDict):
    """The deployment definition that the console writes to ``instance.json``.

    ``seed`` is the persona YAML file, ``config`` the model config file (empty
    when the deployment uses the default), and ``cognition`` selects real or
    dummy cognition. ``deploy`` writes and validates those four keys.

    ``assembly`` and ``operator_hooks`` name the modules the worker imports, so
    the console keeps no companion behavior of its own. A deployment written by
    hand may omit them, and the defaults of
    :mod:`pm.console.assembly` apply.
    """

    name: str
    seed: str
    config: str
    cognition: str
    assembly: NotRequired[str]
    operator_hooks: NotRequired[str]


class Instance:
    """Own one worker process, its command acknowledgments, and its event log."""

    def __init__(self, root: Path) -> None:
        """Read the deployment and replay its recorded events; start no process."""
        self.root = root
        self.settings: InstanceSettings = json.loads((root / "instance.json").read_text(encoding="utf-8"))
        self.process: subprocess.Popen[str] | None = None
        # Worker protocol events carry kind-specific fields, so the log keeps a
        # freeform JSON object per line and forwards it unchanged to the page.
        self.events: deque[dict] = deque(maxlen=10000)
        self.lock = threading.RLock()
        self.command_lock = threading.Lock()
        # One pending command per request ID: the set event, then the merged
        # acknowledgment fields of that command.
        self.waiters: dict[str, tuple[threading.Event, dict[str, JsonValue]]] = {}
        self.sequence = 0
        self.snapshot: dict = {}
        # The node list of the last ``ready`` event, forwarded verbatim as JSON.
        self.nodes: list = []
        log = root / "events.jsonl"
        if log.exists():
            for line in log.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A killed process can leave the final line incomplete.
                self.sequence = max(self.sequence, event["seq"])
                self.events.append(event)
                if event["kind"] == "snapshot":
                    self.snapshot = event
                if event["kind"] == "ready":
                    self.nodes = event["nodes"]

    def record(self, event: dict) -> None:
        """Append one event to memory and to the durable log under the same sequence.

        The sequence is assigned here, so a replayed log and a live stream agree
        on message order. A pending command with this request ID is woken with
        the merged event fields.
        """
        with self.lock:
            self.sequence += 1
            event = {**event, "seq": self.sequence}
            self.events.append(event)
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            if event["kind"] == "snapshot":
                self.snapshot = event
            if event["kind"] == "ready":
                self.nodes = event["nodes"]
            request_id = event.get("request_id")
            waiter = self.waiters.get(request_id) if isinstance(request_id, str) else None
            if waiter is not None:
                waiter[1].update(event)
                waiter[0].set()

    def info(self) -> dict[str, object]:
        """Return the deployment settings plus live process and worker state.

        The settings are spread unchanged, so a deployment file keeps any extra
        key it holds. ``worker_state`` comes from the last lifecycle event, not
        from a guess: a live worker counts as ``starting`` until it reports
        ``ready``.
        """
        with self.lock:
            process = self.process
            live = process is not None and process.poll() is None
            lifecycle = next((e["kind"] for e in reversed(self.events)
                              if e["kind"] in {"ready", "instance.start", "instance.stop", "application.error"}), None)
            info: dict[str, object] = {
                **self.settings, "id": self.root.name, "state": "running" if live else "stopped",
                "worker_state": ("failed" if lifecycle == "application.error" else
                                 "stopping" if lifecycle == "instance.stop" else
                                 "ready" if lifecycle == "ready" else "starting") if live else "stopped",
                "pid": process.pid if process is not None and live else None,
                "exit_code": process.poll() if process is not None else None}
            return info

    def start(self) -> None:
        """Launch one worker process and its stdout reader. A running instance raises."""
        with self.command_lock:
            if self.info()["state"] == "running":
                raise ValueError("The instance already runs.")
            (self.root / "logs").mkdir(exist_ok=True)
            error_log = (self.root / "logs" / "worker.log").open("a", encoding="utf-8")
            try:
                self.process = subprocess.Popen(
                    [sys.executable, "-u", "-m", "pm.console.worker", str(self.root / "instance.json")],
                    cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=error_log,
                    text=True, encoding="utf-8", bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            finally:
                error_log.close()
            threading.Thread(target=self._read, args=(self.process,), daemon=True).start()
        self.record({"kind": "instance.start", "time": time.time(), "pid": self.process.pid})

    def _read(self, process: subprocess.Popen[str]) -> None:
        """Record every worker event line until the process exits, then its code."""
        output, intake = process.stdout, process.stdin
        # The worker is launched with both pipes, so neither stream is absent.
        assert output is not None and intake is not None
        for line in output:
            try:
                self.record(json.loads(line))
            except json.JSONDecodeError:
                self.record({"kind": "protocol.error", "time": time.time(), "text": line})
        process.wait()
        self.record({"kind": "instance.exit", "time": time.time(), "exit_code": process.returncode})
        output.close()
        intake.close()

    def command(self, command: dict, timeout: float = 90) -> dict[str, JsonValue]:
        """Send one command to the worker and return its acknowledgment.

        ``command`` needs an ``action`` key. A worker rejection raises
        ``ValueError`` with the worker's own reason, and a worker that does not
        answer within ``timeout`` seconds raises ``TimeoutError``. The request
        ID is registered before the write, because a fast acknowledgment can
        arrive before ``wait`` starts.
        """
        request_id = uuid4().hex
        waiter: tuple[threading.Event, dict[str, JsonValue]] = (threading.Event(), {})
        try:
            with self.command_lock:
                if self.info()["state"] != "running":
                    raise ValueError("Start the instance first.")
                intake = self.process.stdin if self.process is not None else None
                if intake is None:
                    raise ValueError("Start the instance first.")
                with self.lock:
                    self.waiters[request_id] = waiter
                intake.write(json.dumps({**command, "request_id": request_id}) + "\n")
                intake.flush()
            if not waiter[0].wait(timeout):
                raise TimeoutError("The worker did not acknowledge the command. Read the worker log.")
            if waiter[1]["kind"] == "rejected":
                raise ValueError(waiter[1]["error"])
            if command["action"] == "stop":
                self.record({"kind": "instance.stop", "time": time.time()})
            return waiter[1]
        finally:
            with self.lock:
                self.waiters.pop(request_id, None)

    def kill(self) -> None:
        """Kill the worker process tree. A stopped instance is left alone."""
        with self.command_lock:
            if self.info()["state"] != "running":
                return
            process = self.process
            if process is None:
                return
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                process.kill()
            process.wait(timeout=10)
        self.record({"kind": "instance.kill", "time": time.time()})


class Console:
    """Retain deployment definitions and sessions under a private runtime root."""

    def __init__(self, root: Path) -> None:
        """Bind the console to its runtime root and load the stored deployments."""
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.instances: dict[str, Instance] = {
            item.parent.name: Instance(item.parent) for item in self.root.glob("*/instance.json")}
        self.lock = threading.RLock()

    def deploy(self, settings: dict) -> dict[str, object]:
        """Create one deployment from decoded browser settings and return its info.

        ``name`` and ``seed`` (a persona YAML file) are required. The console
        holds no persona of its own. ``settings`` is an unvalidated request
        body, so invalid or missing values raise ValueError here.
        """
        if not str(settings.get("name") or "").strip():
            raise ValueError("Name the deployment.")
        if not str(settings.get("seed") or "").strip():
            raise ValueError("Select a persona YAML file.")
        seed = Path(settings["seed"]).resolve()
        if not seed.is_file():
            raise ValueError("The persona YAML file does not exist.")
        cognition = settings.get("cognition", "real")
        if cognition not in {"real", "dummy"}:
            raise ValueError("Select real or dummy cognition.")
        config = settings.get("config", "")
        if config:
            config = str(Path(config).resolve())
            if not Path(config).is_file():
                raise ValueError("The model configuration file does not exist.")
        settings = {"name": str(settings["name"]).strip(), "seed": str(seed),
                    "config": config, "cognition": cognition}
        with self.lock:
            root = self.root / uuid4().hex
            root.mkdir()
            (root / "instance.json").write_text(json.dumps(settings), encoding="utf-8")
            instance = Instance(root)
            self.instances[root.name] = instance
        return instance.info()


def serve(root: Path, port: int = 8090) -> None:
    """Serve only on loopback until the process ends.

    Browser mutations require the console origin. On the way out, every
    running worker is asked to stop and is killed when it does not.
    """
    console = Console(root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def respond(self, value: object, status: int = 200) -> None:
            """Write one JSON response with no caching."""
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            """Serve the deployment list, one instance's events, or one page file."""
            if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
                return self.respond({"error": "Use the console through its loopback address."}, 403)
            path = urlparse(self.path)
            try:
                if path.path == "/api/instances":
                    with console.lock:
                        return self.respond([i.info() for i in console.instances.values()])
                if path.path.startswith("/api/instances/"):
                    instance = console.instances[path.path.split("/")[3]]
                    after = int(parse_qs(path.query).get("after", [0])[0])
                    with instance.lock:
                        events = [e for e in instance.events if e["seq"] > after]
                        if parse_qs(path.query).get("light") == ["1"]:
                            return self.respond({"instance": instance.info(), "events": events})
                        return self.respond({"instance": instance.info(), "events": events,
                                             "snapshot": instance.snapshot if instance.snapshot.get("seq", 0) > after else None,
                                             "nodes": instance.nodes,
                                             "log": (instance.root / "logs" / "worker.log").read_text(encoding="utf-8", errors="replace")[-20000:]
                                             if (instance.root / "logs" / "worker.log").exists() else ""})
                assets = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
                filename = assets.get(path.path)
                if filename is None:
                    return self.respond({"error": "Unknown path."}, 404)
                data = (ROOT / "pm/console/static" / filename).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", {"html": "text/html", "js": "text/javascript", "css": "text/css"}[filename.split(".")[-1]] + "; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
                self.end_headers()
                self.wfile.write(data)
            except (KeyError, ValueError) as exc:
                self.respond({"error": str(exc)}, 404)

        def do_POST(self) -> None:
            """Apply one browser mutation: open a file, probe, deploy, or command."""
            expected = f"http://127.0.0.1:{self.server.server_port}"
            if self.headers.get("Host") != expected.removeprefix("http://") or self.headers.get("Origin") not in {None, expected}:
                return self.respond({"error": "Use the console through its loopback address."}, 403)
            if self.headers.get("Content-Type") != "application/json":
                return self.respond({"error": "Use application/json."}, 415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("Invalid command size.")
                command = json.loads(self.rfile.read(length))
                if self.path == "/api/open":
                    path = Path(command["path"])
                    path = (path if path.is_absolute() else ROOT / path).resolve()
                    if not path.is_relative_to(ROOT) or path.suffix not in {".md", ".py", ".yaml", ".yml"} or not path.is_file():
                        raise ValueError("Select an existing source file within the project.")
                    if path.suffix == ".md":
                        os.startfile(str(path))
                    else:
                        editor = shutil.which("subl") or next((str(p) for p in
                            [Path("C:/Program Files/Sublime Text/sublime_text.exe"),
                             Path("C:/Program Files/Sublime Text 3/sublime_text.exe")] if p.is_file()), None)
                        if editor is None:
                            raise ValueError("Sublime Text is required to open Python and YAML source safely.")
                        subprocess.Popen([editor, str(path)], creationflags=subprocess.CREATE_NO_WINDOW)
                    return self.respond({"opened": str(path)})
                if self.path == "/api/connections":
                    from pm.app.operator_config import connection_status
                    from pm.console.assembly import DEFAULT_OPERATOR_HOOKS, load
                    instance = console.instances[command["instance"]]
                    hooks = load(instance.settings.get("operator_hooks") or DEFAULT_OPERATOR_HOOKS)
                    return self.respond(asyncio.run(connection_status(instance.settings.get("config") or None,
                                                                      probes=hooks.probes)))
                if self.path == "/api/deploy":
                    return self.respond(console.deploy(command))
                instance = console.instances[command.pop("instance")]
                action = command["action"]
                if action == "start":
                    instance.start()
                elif action == "kill":
                    instance.kill()
                else:
                    return self.respond(instance.command(command))
                self.respond(instance.info())
            except (KeyError, ValueError, TimeoutError, OSError, subprocess.SubprocessError) as exc:
                self.respond({"error": str(exc)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Operator console: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        for instance in console.instances.values():
            if instance.info()["state"] == "running":
                try:
                    instance.command({"action": "stop"}, timeout=2)
                    instance.process.wait(timeout=5)
                except (ValueError, TimeoutError, OSError, subprocess.SubprocessError):
                    instance.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pm operator console")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--root", type=Path, default=ROOT / ".runtime/console")
    args = parser.parse_args()
    serve(args.root, args.port)
