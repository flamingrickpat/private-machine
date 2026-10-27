"""Serve a local development board with Python's standard library.

The JSON file owns the cards. One HTTP server serializes browser saves.
Invalid input receives a plain error. Logic errors retain their tracebacks.
Windows replacement locks retry until the same save succeeds.
Each save replaces the complete file before the server reports success.
Ctrl+C stops the server. A completed replacement survives an interrupted reply.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import tempfile
import time
from typing import TypedDict, cast
from urllib.parse import urlsplit
import webbrowser


HERE = Path(__file__).resolve().parent
COLUMNS = {
    "backlog": "Backlog",
    "todo": "To do",
    "doing": "In progress",
    "done": "Done",
}


class Card(TypedDict):
    """One development card. Its positive integer id remains stable."""

    id: int
    title: str
    description: str
    status: str


class Board(TypedDict):
    """The complete file. next_id exceeds every existing card id."""

    next_id: int
    cards: list[Card]


def validate_board(value: object) -> Board:
    """Validate external JSON once. Raise ValueError without changing the file."""
    if not isinstance(value, dict) or set(value) != {"next_id", "cards"}:
        raise ValueError("The board needs next_id and cards fields.")
    if type(value["next_id"]) is not int or value["next_id"] < 1:
        raise ValueError("next_id must be a positive integer.")
    if not isinstance(value["cards"], list):
        raise ValueError("cards must be a list.")
    seen: set[int] = set()
    for card in value["cards"]:
        if not isinstance(card, dict) or set(card) != {"id", "title", "description", "status"}:
            raise ValueError("Each card needs id, title, description, and status fields.")
        if type(card["id"]) is not int or not 0 < card["id"] < value["next_id"]:
            raise ValueError("Each card id must be positive and less than next_id.")
        if card["id"] in seen:
            raise ValueError("Card ids must be unique.")
        seen.add(card["id"])
        if not isinstance(card["title"], str) or not card["title"].strip():
            raise ValueError("Each card needs a title.")
        if not isinstance(card["description"], str):
            raise ValueError("Each description must be text.")
        if not isinstance(card["status"], str) or card["status"] not in COLUMNS:
            raise ValueError("Use backlog, todo, doing, or done for status.")
    return cast(Board, value)


def read_board(path: Path) -> tuple[Board, str]:
    """Read validated cards and a revision of the exact file bytes.

    Invalid JSON raises ValueError. File errors raise OSError.
    A direct file edit changes the revision that browser saves must match.
    """
    content = path.read_bytes()
    return validate_board(json.loads(content)), hashlib.sha256(content).hexdigest()


def write_board(path: Path, board: Board) -> None:
    """Replace a validated board atomically after flushing it to disk.

    The caller owns revision checks. Run only one server per board file.
    A Windows file lock delays replacement. Other file errors raise OSError.
    An interruption before replacement leaves the previous board intact.
    """
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(board, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        while True:
            try:
                temporary.replace(path)
                return
            except PermissionError as error:
                # Windows scanners can briefly retain a handle to the board file.
                print(f"{error}. Waiting for the board file lock.", flush=True)
                time.sleep(0.2)
    finally:
        temporary.unlink(missing_ok=True)


def board_handler(path: Path) -> type[BaseHTTPRequestHandler]:
    """Bind one board file to a handler that serves only the board assets.

    GET reads current disk state. PUT validates input and rejects stale revisions.
    Cross-origin browser writes receive 403. Only board data and browser assets are served.
    """
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, content: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def _json(self, status: int, value: object) -> None:
            self._reply(status, json.dumps(value, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8")

        def _snapshot(self) -> dict:
            board, revision = read_board(path)
            return {"board": board, "revision": revision, "columns": COLUMNS}

        def do_GET(self) -> None:
            """Return the current board or an explicit static asset."""
            route = urlsplit(self.path).path
            if route == "/api/board":
                try:
                    snapshot = self._snapshot()
                except (ValueError, OSError) as error:
                    self._json(500, {"error": f"Cannot read the board: {error}"})
                    return
                self._json(200, snapshot)
                return
            assets = {"/": ("index.html", "text/html"),
                      "/board.js": ("board.js", "text/javascript"),
                      "/board.css": ("board.css", "text/css")}
            if route not in assets:
                self._json(404, {"error": "Page not found."})
                return
            filename, content_type = assets[route]
            self._reply(200, (HERE / filename).read_bytes(), f"{content_type}; charset=utf-8")

        def do_PUT(self) -> None:
            """Save a complete board only when the loaded revision still matches."""
            if urlsplit(self.path).path != "/api/board":
                self._json(404, {"error": "Page not found."})
                return
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://127.0.0.1:{self.server.server_port}":
                self._json(403, {"error": "Open the board at its printed local address."})
                return
            if self.headers.get_content_type() != "application/json":
                self._json(415, {"error": "Send the board as application/json."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0:
                    raise ValueError("Content-Length must not be negative.")
                board = validate_board(json.loads(self.rfile.read(length)))
                # Read disk after the request body to detect edits during upload.
                _, revision = read_board(path)
                if self.headers.get("If-Match") != revision:
                    self._json(409, {"error": "The board changed on disk. Copy your edits, close this card, and reload the board."})
                    return
            except ValueError as error:
                self._json(400, {"error": str(error)})
                return
            except OSError as error:
                self._json(500, {"error": f"Cannot read the board: {error}"})
                return
            try:
                write_board(path, board)
                snapshot = self._snapshot()
            except OSError as error:
                self._json(500, {"error": f"Cannot save the board: {error}. Reload to inspect the file."})
                return
            self._json(200, snapshot)

    return Handler


def main() -> None:
    """Serve the selected JSON file on loopback until Ctrl+C.

    --port selects the local port. --data selects an existing board file.
    --no-browser leaves the browser closed. Startup errors exit with plain text.
    """
    parser = argparse.ArgumentParser(description="Open the local private-machine Kanban board.")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--data", type=Path, default=HERE / "board.json")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        read_board(args.data)
        server = HTTPServer(("127.0.0.1", args.port), board_handler(args.data))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Cannot start the board: {error}\n")
    address = f"http://127.0.0.1:{server.server_port}"
    print(f"Kanban: {address}\nCards: {args.data.resolve()}", flush=True)
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
