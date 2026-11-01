"""Start the read-only K003 execution timeline.

The launcher chooses an evidence root, binds the loopback server, and prints the
direct review URL. It does not load a checkpoint or run a workflow, so it is
safe to open while the owner is reading a finished attempt.

Owns: command-line options and server lifecycle.
Does not own: evidence parsing or page presentation.
See the package overview in ``pm.tools.timeline``.
"""
from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path

from pm import constants
from pm.tools.timeline.api import TimelineApi
from pm.tools.timeline.evidence import DEFAULT_RUNS_ROOT, EvidenceStore
from pm.tools.timeline.server import server_factory


def serve(store: EvidenceStore, host: str, port: int) -> ThreadingHTTPServer:
    """Bind the timeline API to ``host`` and ``port`` and return the server."""
    return server_factory(TimelineApi(store))((host, port))


def run_timeline(*, runs_root: Path = DEFAULT_RUNS_ROOT, host: str = constants.TIMELINE_HOST,
                 port: int = constants.TIMELINE_PORT, run_id: str = "", attempt: int = 0) -> int:
    """Serve the execution timeline and block until Ctrl-C.

    ``run_id`` and ``attempt`` only select the initial page state. The page can
    change both from its own controls. Returns ``1`` when the evidence root or
    the port cannot be used.
    """
    root = runs_root.expanduser().resolve()
    if not root.is_dir():
        print(f"No simulation evidence root at {root}.", flush=True)
        print("Pass --runs-root with the folder that contains run folders.", flush=True)
        return 1
    store = EvidenceStore(root)
    if run_id:
        try:
            selected = store.run(run_id)
            if attempt:
                store.attempt(run_id, attempt)
            print(f"Selected {selected.summary.id}: {selected.summary.title}", flush=True)
        except (FileNotFoundError, ValueError) as error:
            print(str(error), flush=True)
            return 1
    try:
        server = serve(store, host, port)
    except OSError as error:
        print(f"Cannot open {host}:{port}: {error}. Pass --port with a free port.", flush=True)
        return 1
    route = ""
    if run_id:
        route = f"#/run/{run_id}/{attempt or 1}/overview"
    print(f"Execution timeline: http://{host}:{server.server_port}/{route}", flush=True)
    print(f"  evidence root: {root}", flush=True)
    print("  read-only: no checkpoint load, no workflow, no model call. Ctrl-C stops.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nExecution timeline stopped.", flush=True)
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the execution timeline from a command line."""
    parser = argparse.ArgumentParser(prog="python -m pm.tools.timeline",
                                     description="Inspect one simulation attempt's execution evidence. Reads only.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
                        help=f"folder with simulation run folders (default: {DEFAULT_RUNS_ROOT})")
    parser.add_argument("--run", default="", help="select a run id when the page opens")
    parser.add_argument("--attempt", type=int, default=0, help="select an attempt when the page opens")
    parser.add_argument("--host", default=constants.TIMELINE_HOST, help="host to bind (default: loopback)")
    parser.add_argument("--port", type=int, default=constants.TIMELINE_PORT,
                        help=f"port to serve (default: {constants.TIMELINE_PORT})")
    args = parser.parse_args(argv)
    return run_timeline(runs_root=args.runs_root, host=args.host, port=args.port,
                        run_id=args.run, attempt=args.attempt)


__all__ = ["main", "run_timeline", "serve"]
