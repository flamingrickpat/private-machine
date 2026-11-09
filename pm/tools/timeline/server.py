"""The HTTP server for the K003 review timeline.

The server defaults to ``0.0.0.0``. The page shows private prompts, captured
records, and companion speech, and it has no login. Every endpoint is a read.
The launcher builds the evidence store and this module serves the API and the
three static files.

Owns: HTTP request routing and static-file delivery.
Does not own: evidence parsing or timeline construction.
See the package overview in ``pm.tools.timeline``.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pm.tools.timeline.api import TimelineApi

STATIC_ROOT = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css",
                "/workspace-theme.css": "theme.css"}
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8"}
CSP = "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:;"


def handler(api: TimelineApi) -> type[BaseHTTPRequestHandler]:
    """Return a request handler bound to ``api``."""

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def handle(self) -> None:
            """End a read connection when its browser closes the socket."""
            try:
                super().handle()
            except (ConnectionResetError, BrokenPipeError):
                self.close_connection = True

        def do_GET(self) -> None:  # noqa: N802 - the name comes from BaseHTTPRequestHandler
            """Answer one API call, or serve one static file."""
            url = urlparse(self.path)
            if url.path.startswith("/api/"):
                status, payload = api.get(url.path[len("/api"):], parse_qs(url.query))
                self._send(status, "application/json; charset=utf-8",
                           json.dumps(payload, default=str, indent=1).encode("utf-8"))
                return
            name = STATIC_FILES.get(url.path)
            if name is None:
                self._send(404, "text/plain; charset=utf-8", b"not found\n")
                return
            path = STATIC_ROOT / name
            if not path.exists():
                self._send(404, "text/plain; charset=utf-8",
                           f"static file is missing: {path}\n".encode("utf-8"))
                return
            self._send(200, CONTENT_TYPES[path.suffix], path.read_bytes())

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Keep the terminal clean. One page reload is not a runtime event."""

    return _Handler


def server_factory(api: TimelineApi) -> Callable[[tuple[str, int]], ThreadingHTTPServer]:
    """Return a factory that binds the timeline API to a host and port."""

    def build(address: tuple[str, int]) -> ThreadingHTTPServer:
        server = ThreadingHTTPServer(address, handler(api))
        server.daemon_threads = True
        return server

    return build


__all__ = ["CONTENT_TYPES", "CSP", "STATIC_FILES", "STATIC_ROOT", "handler", "server_factory"]
