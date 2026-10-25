"""Start the inspector: load one checkpoint, serve the page, change nothing.

The launcher is the only place in the inspector that touches the disk. It
reads a checkpoint, builds the index and the renderer, and starts a loopback
HTTP server. It never saves the store, never starts an embedder, never opens a
model endpoint, and never runs a cycle, so it is safe to point at a
character's own ``ghost.pickle`` while that character is asleep.

The same ``run_inspector`` serves the ``--inspector`` flag of ``main.py`` and
the standalone ``python -m pm.tools.inspector`` entry point.
"""
from __future__ import annotations

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path

from pm import constants
from pm.memory.store import MemoryStore
from pm.tools.inspector.api import InspectorApi
from pm.tools.inspector.model import MemoryIndex
from pm.tools.inspector.rendering import StoryRenderer
from pm.tools.inspector.server import server_factory


def open_index(checkpoint: Path) -> MemoryIndex:
    """Load one checkpoint read-only and build the lookup tables over it.

    Reading a 90 MB checkpoint takes about a second, and the index scan over
    its records, shards, and references takes a second more. Loading builds no
    embeddings: a plain checkpoint read does not carry them, and the inspector
    shows no similarity results rather than pretending that it has them.

    Raises ``FileNotFoundError`` when the checkpoint does not exist.
    """
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint}. Pass --checkpoint, or --home with a ghost.pickle.")
    return MemoryIndex(MemoryStore.from_checkpoint(checkpoint))


def serve(index: MemoryIndex, host: str, port: int) -> ThreadingHTTPServer:
    """Bind the inspector API of ``index`` to ``host`` and ``port`` and return the server.

    The caller runs ``serve_forever``. Pass a loopback host unless you know
    why the memory of a character should reach another machine.
    """
    return server_factory(InspectorApi(index))((host, port))


def run_inspector(*, home: Path | None = None, checkpoint: Path | None = None,
                  host: str = constants.INSPECTOR_HOST, port: int = constants.INSPECTOR_PORT) -> int:
    """Serve one character checkpoint in a browser and block until Ctrl-C.

    Give either ``checkpoint`` or ``home``. With ``home`` the launcher reads
    ``home/ghost.pickle``, which is the file that the running character uses.
    Returns ``0`` on Ctrl-C and ``1`` when the checkpoint or the port cannot be
    used. Nothing here writes to the character's home.
    """
    path = checkpoint if checkpoint is not None else (home / "ghost.pickle" if home is not None else None)
    if path is None:
        print("Give --home or --checkpoint, so the inspector knows which memory to read.", flush=True)
        return 1
    path = Path(path).expanduser()
    # Every line flushes, because a maintainer runs this in a terminal and
    # wants the URL before the store finishes loading.
    print(f"Loading {path} ...", flush=True)
    try:
        index = open_index(path)
    except FileNotFoundError as error:
        print(str(error), flush=True)
        return 1
    try:
        server = serve(index, host, port)
    except OSError as error:
        print(f"Cannot open {host}:{port}: {error}. Pass --inspector-port with a free port.", flush=True)
        return 1
    print(f"Inspector for {index.companion_name}: http://{host}:{server.server_port}", flush=True)
    print(f"  checkpoint: {path}", flush=True)
    print(f"  records: {len(index.records)}   cycles: {len(index.states)}   gwt turns: {len(index.turns)}",
          flush=True)
    print("  read-only: no cycle runs, no model call, nothing is saved. Ctrl-C stops.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nInspector stopped.", flush=True)
    finally:
        server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the inspector straight from a command line, without the app launcher."""
    parser = argparse.ArgumentParser(prog="python -m pm.tools.inspector",
                                     description="Inspect one character checkpoint in a browser. Reads only.")
    parser.add_argument("--home", help="character home folder; reads <home>/ghost.pickle")
    parser.add_argument("--checkpoint", help="path to a ghost.pickle to read")
    parser.add_argument("--inspector-host", default=constants.INSPECTOR_HOST, help="host to bind (default: loopback)")
    parser.add_argument("--inspector-port", type=int, default=constants.INSPECTOR_PORT,
                        help=f"port to serve (default: {constants.INSPECTOR_PORT})")
    args = parser.parse_args(argv)
    return run_inspector(home=Path(args.home) if args.home else None,
                         checkpoint=Path(args.checkpoint) if args.checkpoint else None,
                         host=args.inspector_host, port=args.inspector_port)


__all__ = ["main", "open_index", "run_inspector", "serve"]
