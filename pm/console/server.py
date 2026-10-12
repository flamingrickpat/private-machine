"""Loopback HTTP service for instance lifecycle and recorded runtime evidence."""
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
from urllib.parse import urlparse, parse_qs
from uuid import uuid4
from pm.paths import REPO_ROOT

ROOT = REPO_ROOT


class Instance:
    """Own one worker process, its command acknowledgments, and its event log."""

    def __init__(self, root: Path):
        self.root = root
        self.settings = json.loads((root / "instance.json").read_text(encoding="utf-8"))
        self.process = None
        self.events = deque(maxlen=10000)
        self.lock = threading.RLock()
        self.command_lock = threading.Lock()
        self.waiters = {}
        self.sequence = 0
        self.snapshot = {}
        self.nodes = []
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

    def record(self, event):
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
            waiter = self.waiters.get(event.get("request_id"))
            if waiter is not None:
                waiter[1].update(event)
                waiter[0].set()

    def info(self):
        with self.lock:
            live = self.process is not None and self.process.poll() is None
            lifecycle = next((e["kind"] for e in reversed(self.events)
                              if e["kind"] in {"ready", "instance.start", "instance.stop", "application.error"}), None)
            return {**self.settings, "id": self.root.name, "state": "running" if live else "stopped",
                    "worker_state": ("failed" if lifecycle == "application.error" else
                                     "stopping" if lifecycle == "instance.stop" else
                                     "ready" if lifecycle == "ready" else "starting") if live else "stopped",
                    "pid": self.process.pid if live else None,
                    "exit_code": self.process.poll() if self.process else None}

    def start(self):
        with self.command_lock:
            if self.info()["state"] == "running":
                raise ValueError("The instance already runs.")
            error_log = (self.root / "worker.log").open("a", encoding="utf-8")
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

    def _read(self, process):
        for line in process.stdout:
            try:
                self.record(json.loads(line))
            except json.JSONDecodeError:
                self.record({"kind": "protocol.error", "time": time.time(), "text": line})
        process.wait()
        self.record({"kind": "instance.exit", "time": time.time(), "exit_code": process.returncode})
        process.stdout.close()
        process.stdin.close()

    def command(self, command, timeout=90):
        request_id = uuid4().hex
        waiter = (threading.Event(), {})
        try:
            with self.command_lock:
                if self.info()["state"] != "running":
                    raise ValueError("Start the instance first.")
                with self.lock:
                    self.waiters[request_id] = waiter
                self.process.stdin.write(json.dumps({**command, "request_id": request_id}) + "\n")
                self.process.stdin.flush()
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

    def kill(self):
        with self.command_lock:
            if self.info()["state"] != "running":
                return
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                               capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                self.process.kill()
            self.process.wait(timeout=10)
        self.record({"kind": "instance.kill", "time": time.time()})


class Console:
    """Retain deployment definitions and sessions under a private runtime root."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.instances = {p.parent.name: Instance(p.parent) for p in self.root.glob("*/instance.json")}
        self.lock = threading.RLock()

    def deploy(self, settings):
        seed = Path(settings.get("seed") or ROOT / "test_data/personas/spamuel/canon.yaml").resolve()
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
        settings = {"name": settings.get("name") or "Spamuel", "seed": str(seed),
                    "config": config, "cognition": cognition}
        with self.lock:
            root = self.root / uuid4().hex
            root.mkdir()
            (root / "instance.json").write_text(json.dumps(settings), encoding="utf-8")
            instance = Instance(root)
            self.instances[root.name] = instance
        return instance.info()


def serve(root: Path, port: int = 8090):
    """Serve only on loopback. Browser mutations require the console origin."""
    console = Console(root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, value, status=200):
            data = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
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
                                             "log": (instance.root / "worker.log").read_text(encoding="utf-8", errors="replace")[-20000:]
                                             if (instance.root / "worker.log").exists() else ""})
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

        def do_POST(self):
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
                    instance = console.instances[command["instance"]]
                    return self.respond(asyncio.run(connection_status(instance.settings.get("config") or None)))
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
    parser = argparse.ArgumentParser(description="Spamuel operator console")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--root", type=Path, default=ROOT / ".runtime/console")
    args = parser.parse_args()
    serve(args.root, args.port)
