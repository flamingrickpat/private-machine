"""Start a fresh Minecraft server and the pm-minecraft MCP for one live gate.

The live Minecraft gate must not depend on a server that a human started and
left running. This launcher copies the configured server template into a fresh
run directory, starts the server, starts the pm-minecraft MCP transport, and
starts the body player. ``stop`` reverses each step and deletes the run
directory.

The launcher is only used when ``MINECRAFT_MCP_URL`` points at the default
local endpoint and that endpoint is not already reachable. An explicitly
configured external endpoint is never started or stopped.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_REPO = Path("C:/source/pm/pm-minecraft")
DEFAULT_ENV_FILE = ".env_tdd"
IGNORED = (".git", ".fabric", "logs", "world", "real-server")


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def _port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_port(process: subprocess.Popen, host: str, port: int, timeout: float, label: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{label} stopped during startup.")
        if _port_open(host, port):
            return
        time.sleep(0.25)
    raise TimeoutError(f"{label} did not open {host}:{port} in {timeout:g}s.")


def _wait_log(path: Path, needle: str, timeout: float, label: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and needle in path.read_text(encoding="utf-8", errors="replace"):
            return
        time.sleep(0.25)
    raise TimeoutError(f"{label} did not reach {needle!r} in {timeout:g}s.")


class MinecraftStack:
    """Own one fresh server plus the MCP transport and body for one test."""

    def __init__(
        self,
        *,
        repo: Path = DEFAULT_REPO,
        env_file: Path | str = DEFAULT_ENV_FILE,
        run_root: Path | None = None,
    ) -> None:
        self.repo = Path(repo)
        self.env_path = (self.repo / env_file) if not Path(env_file).is_absolute() else Path(env_file)
        self.values = load_env_file(self.env_path)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base = Path(run_root) if run_root is not None else Path(self.values["TEST_RUN_ROOT"])
        self.run_dir = base / f"pm_next_v2_{stamp}"
        self.server: subprocess.Popen | None = None
        self.mcp: subprocess.Popen | None = None

    @property
    def host(self) -> str:
        return self.values.get("MCP_HOST", "127.0.0.1")

    @property
    def port(self) -> int:
        return int(self.values.get("MCP_PORT", "6767"))

    @property
    def mcp_url(self) -> str:
        return f"http://{self.host}:{self.port}/mcp"

    # -------------------------------------------------------------- lifecycle

    def start(self, *, timeout: float | None = None, body_timeout: float = 180.0, reset_character: bool = True) -> None:
        self.run_dir.parent.mkdir(parents=True, exist_ok=True)
        self._start_server(timeout or float(self.values.get("SERVER_START_TIMEOUT_SECONDS", "180")))
        self._start_mcp()
        _wait_port(self.mcp, self.host, self.port, 60, "The MCP")
        if reset_character:
            self._control("/test/reset")
        self._control("/test/body/start")
        self._wait_body(body_timeout)

    def stop(self) -> None:
        if self.mcp is not None and self.mcp.poll() is None:
            try:
                self._control("/test/body/stop", timeout=30)
            except Exception:
                pass
            self.mcp.terminate()
            self._reap(self.mcp, 15)
        if self.server is not None and self.server.poll() is None:
            try:
                self.server.stdin.write("stop\n")
                self.server.stdin.flush()
            except Exception:
                pass
            self._reap(self.server, 60)
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def _start_server(self, timeout: float) -> None:
        template = Path(self.values["MINECRAFT_SERVER_TEMPLATE"])
        run_dir = self.run_dir
        shutil.copytree(template, run_dir, ignore=shutil.ignore_patterns(*IGNORED))
        properties = run_dir / "server.properties"
        port = int(self.values["MINECRAFT_PORT"])
        lines = properties.read_text(encoding="utf-8").splitlines()
        lines = [
            f"server-port={port}" if line.startswith("server-port=") else line
            for line in lines
        ]
        properties.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log_path = run_dir / "server.log"
        log = log_path.open("w", encoding="utf-8")
        memory = self.values.get("SERVER_MEMORY", "2G")
        jar = Path(self.values["MINECRAFT_SERVER_JAR"])
        self.server = subprocess.Popen(
            [self.values["JAVA_EXE"], f"-Xms{memory}", f"-Xmx{memory}",
             "-jar", str(jar), "nogui"],
            cwd=run_dir, stdin=subprocess.PIPE, stdout=log,
            stderr=subprocess.STDOUT, text=True,
        )
        _wait_log(log_path, "Done (", timeout, "The Minecraft server")

    def _start_mcp(self) -> None:
        log = (self.run_dir / "mcp.log").open("w", encoding="utf-8")
        self.mcp = subprocess.Popen(
            [self.values["PYTHON_EXE"], "main.py", str(self.env_path)],
            cwd=self.repo, stdin=subprocess.PIPE, stdout=log,
            stderr=subprocess.STDOUT, text=True,
        )

    def _control(self, path: str, *, timeout: float = 15) -> None:
        import httpx
        response = httpx.post(f"http://{self.host}:{self.port}{path}", timeout=timeout)
        response.raise_for_status()

    def _wait_body(self, timeout: float) -> None:
        from workflows.minecraft_body.mcp_client import build_minecraft_tools
        tools = build_minecraft_tools(self.mcp_url)
        deadline = time.monotonic() + timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            try:
                observation = tools["minecraft_call"]("minecraft_observe", {"include_image": False})
                if observation.get("ok"):
                    return
                last = RuntimeError(str(observation.get("result")))
            except Exception as error:  # the body is not connected yet
                last = error
            time.sleep(1.0)
        raise TimeoutError(f"The Minecraft body did not become ready in {timeout:g}s: {last!r}")

    @staticmethod
    def _reap(process: subprocess.Popen, timeout: float) -> None:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def autostart_enabled(mcp_url: str) -> bool:
    """Autostart only for the default local endpoint and when it is offline."""
    from workflows.minecraft_body.mcp_client import DEFAULT_MCP_URL
    override = os.environ.get("PM_MINECRAFT_AUTOSTART", "1") != "0"
    if not override or mcp_url != DEFAULT_MCP_URL:
        return False
    from urllib.parse import urlsplit
    parts = urlsplit(mcp_url)
    return not _port_open(parts.hostname or "127.0.0.1", parts.port or 80)


__all__ = ["MinecraftStack", "autostart_enabled", "load_env_file"]
