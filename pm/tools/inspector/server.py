"""The local HTTP server that shows one loaded checkpoint.

Only ``127.0.0.1`` should be passed as the host, because the page shows the
contents of a private memory store and has no login. The server has no
endpoints that change anything: every request is a read.

This module holds no memory logic. It turns a path into a call on
:class:`pm.tools.inspector.api.InspectorApi`, and it serves the three static
files of ``static/``. The launcher builds the index and starts the server.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pm.tools.inspector.api import InspectorApi

#: Where the page, the script, and the style live.
STATIC_ROOT = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".json": "application/json; charset=utf-8"}
CSP = "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:;"


def handler(api: InspectorApi) -> type[BaseHTTPRequestHandler]:
    """Return a request handler bound to ``api``.

    The caller passes this to ``ThreadingHTTPServer``. Each request reads the
    prepared index, so a reload of the page costs only the work of one answer.
    """

    class _Handler(BaseHTTPRequestHandler):
        # HTTP/1.1 needs an explicit length on every reply, so one connection
        # can carry the page and its API calls without hanging a client.
        protocol_version = "HTTP/1.1"

        def do_GET(self):  # noqa: N802 - the name comes from BaseHTTPRequestHandler
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

        def log_message(self, *args):
            """Keep the terminal clean. A page reload is not a runtime event."""

    return _Handler


def server_factory(api: InspectorApi) -> Callable[[tuple[str, int]], ThreadingHTTPServer]:
    """Return a factory that binds the inspector API to a host and port.

    The threads are daemons, so Ctrl-C in the terminal ends the inspector
    without waiting for a request that a browser left open.
    """

    def build(address: tuple[str, int]) -> ThreadingHTTPServer:
        # ``HTTPServer`` already sets ``allow_reuse_address``, so a restarted
        # inspector can take its port back at once.
        server = ThreadingHTTPServer(address, handler(api))
        server.daemon_threads = True
        return server

    return build


__all__ = ["CONTENT_TYPES", "CSP", "STATIC_FILES", "STATIC_ROOT", "handler", "server_factory"]
