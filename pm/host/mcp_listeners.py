"""Temporary loopback MCP listeners for one workflow run.

Each named scoped service becomes a FastMCP HTTP server on an
operating-system-assigned loopback port. The bound socket is held for the
listener's whole lifetime, so the port cannot be taken between selection and
startup. Connection descriptors stay private to the run; servers close after
success, error, or cancellation.
"""
from __future__ import annotations

import socket
import threading
import time
import logging
from contextlib import contextmanager
from typing import Iterator, Mapping

from fastmcp import FastMCP


class _Listener:
    def __init__(self, name: str, server: FastMCP, bound: socket.socket):
        self.name = name
        self.server = server
        self.socket = bound
        self.port = bound.getsockname()[1]
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.http_server = None

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run, name=f"mcp-{self.name}", daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        import asyncio
        logger = logging.getLogger(__name__)
        logger.info("MCP listener start name=%s port=%s", self.name, self.port)

        async def serve() -> None:
            import uvicorn
            asyncio.get_running_loop().set_exception_handler(_ignore_aborted_socket)
            # Scoped tools return one complete result per request. Stateless
            # JSON avoids background SSE sessions that outlive this listener.
            app = self.server.http_app(transport="http", stateless_http=True, json_response=True)
            config = uvicorn.Config(app, host="127.0.0.1", port=self.port,
                                    lifespan="on", log_level="error", timeout_graceful_shutdown=2)
            self.http_server = uvicorn.Server(config)
            async with self.server._lifespan_manager():
                await self.http_server.serve(sockets=[self.socket])

        try:
            asyncio.run(serve())
        except BaseException as exc:  # the listener thread must record its fault
            self.error = exc
            logger.exception("MCP listener failed name=%s", self.name)
        finally:
            logger.info("MCP listener stopped name=%s", self.name)

    def wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.error is not None:
                raise RuntimeError(
                    f"MCP service '{self.name}' failed to start"
                ) from self.error
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.02)
        raise TimeoutError(f"MCP service '{self.name}' did not become ready")

    def close(self) -> None:
        logging.getLogger(__name__).info("MCP listener close requested name=%s", self.name)
        # Stop the accept loop before closing the held socket. Closing only
        # the socket leaves the server thread and its MCP sessions alive.
        if self.http_server is not None:
            self.http_server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                logging.getLogger(__name__).warning("MCP listener remains active name=%s", self.name)
        try:
            self.socket.close()
        except OSError:
            pass
        if self.thread is not None:
            self.thread.join(timeout=5)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"


class ScopedMcpRegistry:
    """Owns the loopback listeners started for one run and tears them down."""

    def __init__(self) -> None:
        self._listeners: dict[str, _Listener] = {}

    def serve(
        self, servers: Mapping[str, FastMCP], *, timeout: float = 10.0
    ) -> dict[str, dict]:
        """Start each named server on a held port and return HTTP descriptors."""
        if not servers:
            return {}
        started: list[_Listener] = []
        try:
            for name, server in servers.items():
                if name in self._listeners:
                    raise ValueError(f"MCP service '{name}' is already running.")
                bound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                bound.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                bound.bind(("127.0.0.1", 0))
                bound.listen(128)
                listener = _Listener(name, server, bound)
                self._listeners[name] = listener
                listener.start()
                started.append(listener)
                listener.wait_ready(timeout)
            return {name: {"url": svc.url, "type": "http"} for name, svc in self._listeners.items()}
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for name in list(reversed(list(self._listeners))):
            self._listeners.pop(name).close()

    @property
    def active_services(self) -> list[str]:
        return sorted(self._listeners)

    def __enter__(self) -> "ScopedMcpRegistry":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _ignore_aborted_socket(loop: object, context: dict) -> None:
    """Closing a held listener aborts its pending accept on Windows."""
    exc = context.get("exception")
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 995:
        return
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in {9, 22, 10038}:
        return
    loop.default_exception_handler(context)  # type: ignore[attr-defined]


@contextmanager
def serve_services(
    servers: Mapping[str, FastMCP], *, timeout: float = 10.0
) -> Iterator[dict[str, dict]]:
    """Start scoped services for a run and guarantee their cleanup."""
    with ScopedMcpRegistry() as registry:
        yield registry.serve(servers, timeout=timeout)


__all__ = ["ScopedMcpRegistry", "serve_services"]
