"""Loopback browser application for persona forks, scenarios, and run evidence.

``Simulation`` is the browser-facing owner of background jobs and run
controllers: it forks persona homes, starts plan generation, holds the one
``RunEngine`` per run, and stores owner feedback. The handler below serves the
static page and a JSON API on ``127.0.0.1`` only.

Browser/API boundary: a request body is a JSON object read from the socket, a
response is JSON or a captured artifact, and a path is resolved only after it is
checked against its owner's root. A POST must be same-origin. A rejected request
answers 400 with its reason; an unexpected exception answers 500 and is not
hidden. Nothing here advances a scenario by itself: run operations are forwarded
to the engine, which owns the durable controller document.

The package overview is in ``simulation/__init__.py``.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

from pydantic import JsonValue

from pm.host.durable_files import atomic_json
from pm.host.processes import ensure_process_cleanup
from pm.memory.store import MemoryStore
from simulation.agents import Agents
from simulation.engine import RunEngine
from simulation.personas import DEFAULT_HOMES, DEFAULT_RUNS, canon_home, fork_home, homes
from simulation.plans import ScenarioPlan
from simulation.review.catalog import captures, identifier, load_capture
from simulation.review.record_format import present_packages, record_links, visible_data


STATIC = Path(__file__).parent / "static"


class Simulation:
    """Own browser jobs and run controllers. PM remains in separate processes."""

    def __init__(self, homes_root: Path, runs_root: Path, config: str) -> None:
        self.homes_root: Path = homes_root.resolve()
        self.runs_root: Path = runs_root.resolve()
        self.config: str = str(Path(config).resolve())
        self.engines: dict[str, RunEngine] = {}
        self._lock = threading.RLock()

    def run_root(self, run_id: str) -> Path:
        """Resolve a registered run without allowing a path outside the run root."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Invalid run ID.")
        return self.runs_root / "runs" / run_id

    def engine(self, run_id: str) -> RunEngine:
        """Reuse the single controller of a local run."""
        with self._lock:
            if run_id not in self.engines:
                self.engines[run_id] = RunEngine(self.run_root(run_id))
            return self.engines[run_id]

    def catalog(self) -> dict[str, JsonValue]:
        """List personas and retained runs without model calls."""
        runs: list[dict[str, JsonValue]] = []
        for path in sorted((self.runs_root / "runs").glob("*/state.json"), reverse=True):
            state = json.loads(path.read_text(encoding="utf-8"))
            runs.append({key: state[key] for key in ("id", "title", "status", "created_at", "step_index", "attempt")})
        return {"homes": self.all_homes(), "runs": runs, "captures": captures(self.runs_root), "config": self.config}

    def all_homes(self) -> list[dict[str, str]]:
        """Include run-owned participant homes so their checkpoints can be forked."""
        result = homes(self.homes_root)
        for run in sorted((self.runs_root / "runs").glob("*/state.json")):
            result.extend({**item, "folder": f"{run.parent.name} / {item['folder']}"}
                          for item in homes(run.parent / "participants"))
        return result

    def home(self, value: str) -> Path:
        """Require an existing home registered under the selected persona root."""
        selected = Path(value).resolve()
        if str(selected) not in {item["id"] for item in self.all_homes()}:
            raise ValueError("Choose a registered character home.")
        return selected

    def checkpoint(self, home: Path, name: str) -> Path:
        """Resolve a home backup or named runtime snapshot without leaving its owner."""
        selected = (home / name).resolve()
        allowed = {home.resolve(), (home / "runtime/snapshots").resolve()}
        if selected.parent not in allowed or selected.suffix != ".pickle":
            raise ValueError("Choose a checkpoint in the selected home.")
        return selected

    def job(self, kind: str, body: dict[str, JsonValue]) -> str:
        """Persist a browser job before background work starts.

        The job folder and its ``running`` receipt are durable before the worker
        thread exists, so the browser can poll a job it has just created. The
        worker replaces the receipt with ``completed`` and its result, or with
        ``failed`` and the error repr. Returns the job ID.
        """
        job_id = uuid4().hex
        root = self.runs_root / "jobs" / job_id
        root.mkdir(parents=True)
        atomic_json(root / "state.json", {"status": "running", "kind": kind})
        def work() -> None:
            try:
                if kind == "plan":
                    plan = Agents(self.config, root).plan(body["brief"], body["start_time"])
                    result = {"plan": plan.model_dump(mode="json")}
                else:
                    name = body["name"]
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
                        raise ValueError("Use letters, numbers, underscores, or hyphens for a home folder.")
                    target = self.homes_root / name
                    if kind == "canon":
                        created = canon_home(target, cache_root=self.runs_root / "seeds")
                    else:
                        source = self.home(body["source"])
                        selected = self.checkpoint(source, body["snapshot"]) if body.get("snapshot") else None
                        created = fork_home(source, target, snapshot=selected)
                    result = {"home": str(created.root)}
                atomic_json(root / "state.json", {"status": "completed", "kind": kind, **result})
            except BaseException as error:
                atomic_json(root / "state.json", {"status": "failed", "kind": kind, "error": repr(error)})
        threading.Thread(target=work, name=f"simulation-job-{job_id}", daemon=False).start()
        return job_id

    def close(self) -> None:
        """Pause active runs and wait for their coordinated checkpoints."""
        for engine in self.engines.values():
            if engine.thread is not None and engine.thread.is_alive() and engine.view()["status"] in {"starting", "running"}:
                engine.pause()
        for engine in self.engines.values():
            if engine.thread is not None:
                engine.thread.join()

    def feedback(self, path: Path, changes: dict[str, JsonValue]) -> None:
        """Merge owner notes without replacing ratings on other captured records."""
        with self._lock:
            previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            review = previous.get("review", {})
            records = {**review.get("records", {}), **changes.get("records", {})}
            review.update(changes)
            review["records"] = records
            atomic_json(path, {"updated_at": datetime.now(timezone.utc).isoformat(), "review": review})


def handler(application: Simulation) -> type[BaseHTTPRequestHandler]:
    """Build same-origin browser routes. Invalid requests retain clear failures."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def respond(self, status: int, body: bytes, mime: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def output(self, value: JsonValue, status: int = 200) -> None:
            self.respond(status, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            try:
                self.get()
            except (ValueError, KeyError, FileNotFoundError) as error:
                self.output({"error": str(error)}, 400)
            except ConnectionError:
                pass
            except Exception as error:
                self.output({"error": repr(error)}, 500)

        def get(self) -> None:
            path = urlparse(self.path)
            if path.path == "/favicon.ico":
                return self.respond(204, b"", "image/x-icon")
            query = {key: values[0] for key, values in parse_qs(path.query).items()}
            if path.path == "/api/catalog":
                return self.output(application.catalog())
            if path.path.startswith("/api/captures/"):
                capture_id = identifier(path.path.split("/")[3])
                if path.path.endswith("/feedback"):
                    source = application.runs_root / "reviews" / capture_id / "feedback.json"
                    return self.output(json.loads(source.read_text(encoding="utf-8")) if source.exists() else {})
                return self.output(visible_data(load_capture(capture_id, root=application.runs_root)))
            if path.path == "/api/history":
                home = application.home(query["home"])
                snapshot = query.get("snapshot", "ghost.pickle")
                selected = application.checkpoint(home, snapshot)
                store = MemoryStore.from_checkpoint(selected)
                return self.output({"messages": store.dialogue_history(),
                    "snapshots": [item.relative_to(home).as_posix() for item in sorted([
                        *home.glob("ghost*.pickle"), *(home / "runtime/snapshots").glob("*.pickle")])]})
            if path.path.startswith("/api/jobs/"):
                job_id = path.path.rsplit("/", 1)[1]
                if not re.fullmatch(r"[a-f0-9]+", job_id):
                    raise ValueError("Invalid job ID.")
                return self.output(json.loads((application.runs_root / "jobs" / job_id / "state.json").read_text(encoding="utf-8")))
            if path.path.startswith("/api/runs/"):
                parts = path.path.strip("/").split("/")
                engine = application.engine(parts[2])
                if len(parts) == 3:
                    return self.output({"state": engine.view(), "plan": engine.plan.model_dump(mode="json")})
                if parts[3] == "artifact":
                    relative = unquote(query["path"])
                    artifact = (engine.root / relative).resolve()
                    if not artifact.is_relative_to(engine.root.resolve()) or not artifact.is_file():
                        raise ValueError("Unknown run artifact.")
                    if artifact.suffix not in {".json", ".jsonl", ".md", ".yaml", ".txt", ".log"}:
                        raise ValueError("Use the captured record packages to inspect this artifact.")
                    mime = "application/json; charset=utf-8" if artifact.suffix == ".json" else "text/plain; charset=utf-8"
                    return self.respond(200, artifact.read_bytes(), mime)
                if parts[3] == "record":
                    checkpoint = (engine.root / query["checkpoint"]).resolve()
                    if not checkpoint.is_relative_to(engine.root.resolve()) or checkpoint.name != "ghost.pickle":
                        raise ValueError("Choose a captured checkpoint in this run.")
                    store = MemoryStore.from_checkpoint(checkpoint, raw=True)
                    record = store.get_knoxel_by_id(int(query["id"]))
                    if record is None:
                        raise ValueError("This record was not captured in the checkpoint.")
                    return self.output(record.as_payload(include_embedding=False))
                if parts[3] == "feedback":
                    source = engine.root / "feedback.json"
                    return self.output(json.loads(source.read_text(encoding="utf-8")) if source.exists() else {})
            names = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/style.css": ("style.css", "text/css")}
            if path.path not in names:
                return self.output({"error": "Unknown path."}, 404)
            filename, mime = names[path.path]
            return self.respond(200, (STATIC / filename).read_bytes(), mime + "; charset=utf-8")

        def do_POST(self) -> None:
            if self.headers.get("Origin") not in {None, f"http://{self.headers.get('Host')}"}:
                return self.output({"error": "Cross-origin requests are rejected."}, 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("The request must be a JSON object.")
                self.post(urlparse(self.path).path, body)
            except (ValueError, KeyError, FileNotFoundError, FileExistsError) as error:
                self.output({"error": str(error)}, 400)
            except ConnectionError:
                pass
            except Exception as error:
                self.output({"error": repr(error)}, 500)

        def post(self, path: str, body: dict[str, JsonValue]) -> None:
            if path == "/api/present":
                return self.output(present_packages({"packages": [body["package"]]}))
            if path == "/api/record-links":
                return self.output(record_links(body["record"]))
            if path.startswith("/api/captures/") and path.endswith("/feedback"):
                capture_id = identifier(path.split("/")[3])
                load_capture(capture_id, root=application.runs_root)
                application.feedback(application.runs_root / "reviews" / capture_id / "feedback.json", body)
                return self.output({"status": "saved"})
            if path == "/api/shutdown":
                threading.Thread(target=self.server.shutdown, name="simulation-server-stop", daemon=True).start()
                return self.output({"status": "draining"}, 202)
            if path in {"/api/plans", "/api/homes/canon", "/api/homes/fork"}:
                kind = "plan" if path == "/api/plans" else path.rsplit("/", 1)[1]
                return self.output({"job_id": application.job(kind, body)}, 202)
            if path == "/api/runs":
                plan = ScenarioPlan.model_validate(body["plan"])
                participants = [{**item, "home": str(application.home(item["home"]))} for item in body["participants"]]
                run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:8]
                engine = RunEngine.create(application.run_root(run_id), plan, participants, body.get("edges", []), application.config)
                application.engines[run_id] = engine
                return self.output({"run_id": run_id}, 201)
            if path.startswith("/api/runs/"):
                parts = path.strip("/").split("/")
                engine = application.engine(parts[2])
                operation = parts[3]
                if operation in {"start", "resume"}:
                    engine.start()
                elif operation in {"pause", "stop"}:
                    engine.pause(stop=operation == "stop")
                elif operation == "restore":
                    engine.restore(int(body["attempt"]))
                elif operation == "feedback":
                    application.feedback(engine.root / "feedback.json", body)
                else:
                    raise ValueError("Unknown run operation.")
                return self.output(engine.view(), 202)
            self.output({"error": "Unknown path."}, 404)
    return Handler


def main(argv: Sequence[str] | None = None) -> int:
    """Serve the local simulation browser and drain active runs on exit."""
    parser = argparse.ArgumentParser(description="PM simulation browser")
    parser.add_argument("--homes", type=Path, default=DEFAULT_HOMES)
    parser.add_argument("--data", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--config", default="config.alpha.example.yaml")
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args(argv)
    ensure_process_cleanup()
    application = Simulation(args.homes, args.data, args.config)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(application))
    print(f"Simulation: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        application.close()
        server.server_close()
    return 0
