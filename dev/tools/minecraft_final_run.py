"""Start one disposable Minecraft world and one pm-minecraft body per Spamuel.

This is the local preparation tool for the third beta final run. It owns only
the real services that run sits beside: a fresh Fabric server and one
pm-minecraft MCP process per companion. The simulation harness never starts a
body; it names an endpoint and fails loudly when nothing answers there.

The tool is deliberately concrete. It hard-codes the owner's paths, ports, and
player names, because it exists for one machine and one run. What it does:

1. ``up`` copies the served world template into the ignored runtime folder, so a
   run never mutates the owner's real server directory, patches the copy's
   ports and game rules, and starts it.
2. It writes one ``.env`` per body with different MCP, body, and viewer ports,
   and different character folders, so two bodies cannot collide in one lease
   registry entry or one agent home.
3. It starts both MCP processes and waits until each body reports a live
   observation. A body that never connects raises; it is never replaced.
4. It stays resident as the stack's supervisor: the server's console pipe must
   stay open, and the supervisor stops the MCP processes when the server ends.
   An exit receipt is written on every exit.

``stop`` and ``status`` run in a second process and read the receipt. ``stop``
sends the server ``stop`` over RCON, so the world saves, then terminates the two
MCP process trees. ``up`` refuses to start when its ports are already in use.

Addresses stay on ``127.0.0.1`` even though PM servers bind ``0.0.0.0``. The MCP
URL is not only a listening address here: it is the address the PM client dials
and the key of the body lease. It must be the same string on both sides.

Usage::

    python dev/tools/minecraft_final_run.py up
    python dev/tools/minecraft_final_run.py status
    python dev/tools/minecraft_final_run.py stop
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import socket
import struct
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
# The PM modules below live in the repository, and this file lives in dev/tools.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pydantic import AwareDatetime, BaseModel  # noqa: E402  (path must be set first)
from pm.host.durable_files import atomic_json  # noqa: E402  (path must be set first)
from pm.utils.operating_system import OperatingSystem  # noqa: E402  (path must be set first)

WORKSPACE = REPO / ".runtime" / "beta-final" / "minecraft"
SERVER_TEMPLATE = Path("C:/source/minecraft-server-real")
SERVER_DIR = WORKSPACE / "server"
SERVER_LOG = WORKSPACE / "server.log"
RECEIPT = WORKSPACE / "stack.json"
MINECRAFT_REPO = Path("C:/source/pm/pm-minecraft")
MINECRAFT_PYTHON = MINECRAFT_REPO / ".venv" / "Scripts" / "python.exe"
NODE_EXE = Path("C:/Program Files/nodejs/node.exe")
JAVA_EXE = Path("C:/Users/rick/AppData/Roaming/PrismLauncher/java/java-runtime-gamma/bin/java.exe")
BROWSER_EXE = Path("C:/Program Files/BraveSoftware/Brave-Browser/Application/brave.exe")
SERVER_MEMORY = "2G"
SERVER_PORT = 12345
RCON_PORT = 25575
RCON_PASSWORD = "babymode-test"
# A disposable copy: the source world, its logs, and its crash reports never move.
IGNORED = (".git", ".fabric", "logs", "world", "real-server", "crash-reports")


@dataclass(frozen=True, slots=True)
class BodyProfile:
    """One Minecraft body: player name, ports, and where its files live."""

    id: str
    player: str
    mcp_port: int
    body_port: int
    viewer_port: int

    @property
    def mcp_url(self) -> str:
        return f"http://127.0.0.1:{self.mcp_port}/mcp"

    @property
    def env_path(self) -> Path:
        return WORKSPACE / f"{self.id}.env"

    @property
    def agent_home(self) -> Path:
        return WORKSPACE / f"agent-{self.id}"

    @property
    def log_path(self) -> Path:
        return WORKSPACE / f"{self.id}.log"


BODIES = (
    BodyProfile(id="spamuel-1", player="spamuel_1", mcp_port=6777, body_port=6779, viewer_port=6781),
    BodyProfile(id="spamuel-alpha", player="spamuel_alpha", mcp_port=6778, body_port=6780, viewer_port=6782),
)


class BodyReceipt(BaseModel):
    """Where one started body answers and which process owns it."""

    id: str
    player: str
    mcp_url: str
    mcp_port: int
    mcp_pid: int
    env_path: str
    agent_home: str
    log_path: str


class StackReceipt(BaseModel):
    """Durable record of one started stack, written before any run uses it."""

    started_at: AwareDatetime
    server_dir: str
    server_pid: int
    server_log: str
    rcon_port: int
    bodies: list[BodyReceipt]


def read_receipt() -> StackReceipt:
    """Read the receipt, or raise with the command that creates one."""
    if not RECEIPT.is_file():
        raise FileNotFoundError(f"No Minecraft stack receipt at {RECEIPT}. Run `up` first.")
    return StackReceipt.model_validate_json(RECEIPT.read_text(encoding="utf-8"))


def port_open(host: str, port: int, timeout: float = 0.3) -> bool:
    """Report whether a TCP port accepts a connection now."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def require_free_port(port: int, label: str) -> None:
    """Refuse to start beside another listener; a shared port is not a body."""
    if port_open("127.0.0.1", port):
        raise RuntimeError(f"Port {port} is already in use, but {label} needs it.")


def property_lines(path: Path) -> list[str]:
    """Read one server properties file as editable lines."""
    return path.read_text(encoding="utf-8").splitlines()


def patch_properties(lines: list[str], values: dict[str, str]) -> list[str]:
    """Replace each named property, and append the ones the file does not hold."""
    remaining = dict(values)
    patched: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0] if "=" in line else ""
        if key in remaining:
            patched.append(f"{key}={remaining.pop(key)}")
        else:
            patched.append(line)
    patched.extend(f"{key}={value}" for key, value in remaining.items())
    return patched


def prepare_server() -> None:
    """Create the disposable server copy with this run's ports and game rules."""
    if SERVER_DIR.exists():
        shutil.rmtree(SERVER_DIR)
    shutil.copytree(SERVER_TEMPLATE, SERVER_DIR, ignore=shutil.ignore_patterns(*IGNORED))
    properties = SERVER_DIR / "server.properties"
    lines = patch_properties(property_lines(properties), {
        "server-port": str(SERVER_PORT),
        "rcon.port": str(RCON_PORT),
        "enable-rcon": "true",
        "rcon.password": RCON_PASSWORD,
        "online-mode": "false",
        "level-name": "world",
        "motd": "PM final run survival world",
        "max-players": "20",
        "pvp": "false",
    })
    properties.write_text("\n".join(lines) + "\n", encoding="utf-8")


def start_server() -> subprocess.Popen[str]:
    """Start the disposable server and keep its console pipe open."""
    log = SERVER_LOG.open("w", encoding="utf-8")
    process = OperatingSystem.start_text_process(
        [str(JAVA_EXE), f"-Xms{SERVER_MEMORY}", f"-Xmx{SERVER_MEMORY}", "-jar",
         str(SERVER_DIR / "fabric-server-launch.jar"), "nogui"],
        cwd=SERVER_DIR, stdin=OperatingSystem.pipe, stdout=log,
        stderr=OperatingSystem.merged_output, hidden=True,
    )
    print(f"Server started as pid {process.pid}; waiting for the world", flush=True)
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"The Minecraft server exited ({process.returncode}). Log: {SERVER_LOG}")
        if SERVER_LOG.is_file() and "Done (" in SERVER_LOG.read_text(encoding="utf-8", errors="replace"):
            print("Server is accepting players", flush=True)
            return process
        time.sleep(2.0)


def write_env(profile: BodyProfile) -> Path:
    """Write one body's configuration file from the setting definitions it needs."""
    settings = {
        "PYTHON_EXE": str(MINECRAFT_PYTHON),
        "JAVA_EXE": str(JAVA_EXE),
        "NODE_EXE": str(NODE_EXE),
        "MCMCP_BODY_ENTRYPOINT": str(MINECRAFT_REPO / "body" / "dist" / "main.js"),
        "MINECRAFT_HOST": "127.0.0.1",
        "MINECRAFT_PORT": str(SERVER_PORT),
        "MINECRAFT_VERSION": "1.19.4",
        "MINECRAFT_PLAYER": profile.player,
        "MINECRAFT_RCON_HOST": "127.0.0.1",
        "MINECRAFT_RCON_PORT": str(RCON_PORT),
        "MINECRAFT_RCON_PASSWORD": RCON_PASSWORD,
        "MCP_HOST": "127.0.0.1",
        "MCP_PORT": str(profile.mcp_port),
        "MCMCP_TEST_MODE": "false",
        "MCMCP_AGENT_HOME": str(profile.agent_home),
        "MCMCP_BODY_HOST": "127.0.0.1",
        "MCMCP_BODY_PORT": str(profile.body_port),
        "MCMCP_VIEWER_PORT": str(profile.viewer_port),
        "MCMCP_WALK_REACHED_DISTANCE_BLOCKS": "3",
        "BROWSER_EXECUTABLE_PATH": str(BROWSER_EXE),
        "MCMCP_LOG_LEVEL": "info",
        "MCMCP_INSPECTOR_ENABLED": "false",
        "MCMCP_INSPECTOR_HOST": "0.0.0.0",
        "MCMCP_INSPECTOR_PORT": "0",
    }
    profile.agent_home.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"{key}={value}" for key, value in settings.items()) + "\n"
    profile.env_path.write_text(text, encoding="utf-8")
    return profile.env_path


def start_body(profile: BodyProfile) -> subprocess.Popen[bytes]:
    """Start one pm-minecraft MCP process against the running server."""
    env_path = write_env(profile)
    log = profile.log_path.open("wb")
    return OperatingSystem.start_logged_process(
        [str(MINECRAFT_PYTHON), "main.py", str(env_path)],
        cwd=MINECRAFT_REPO, log=log, environment=OperatingSystem.copy_environment(),
    )


def wait_body(profile: BodyProfile, process: subprocess.Popen[bytes]) -> None:
    """Wait until the body answers one observation, then report it.

    There is no deadline: a fresh world takes as long as it takes, and a body
    that stays absent is a real failure to read, not a test to time out.
    """
    from workflows.minecraft_body.mcp_client import probe_body
    print(f"Waiting for body {profile.id} at {profile.mcp_url}", flush=True)
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"Body {profile.id} exited ({process.returncode}). Log: {profile.log_path}")
        if port_open("127.0.0.1", profile.mcp_port):
            try:
                tools = probe_body(profile.mcp_url, timeout_seconds=30)
            except Exception as error:
                print(f"Body {profile.id} MCP is up, body not ready yet: {error}", flush=True)
            else:
                if tools:
                    print(f"Body {profile.id} is live with {len(tools)} tools", flush=True)
                    return
        time.sleep(5.0)


class Rcon:
    """Minimal Minecraft RCON client: enough to ask the server to stop."""

    def __init__(self, host: str, port: int, password: str) -> None:
        self.host = host
        self.port = port
        self.password = password
        self.request_id = 0

    def _packet(self, request_id: int, kind: int, body: str) -> bytes:
        payload = struct.pack("<ii", request_id, kind) + body.encode("utf-8") + b"\x00\x00"
        return struct.pack("<i", len(payload)) + payload

    @staticmethod
    def _read(connection: socket.socket) -> tuple[int, int, str]:
        """Return one answer as (request id, packet type, body text)."""
        raw = connection.recv(4)
        if len(raw) < 4:
            raise RuntimeError("The RCON connection closed before it answered.")
        length = struct.unpack("<i", raw)[0]
        data = b""
        while len(data) < length:
            chunk = connection.recv(length - len(data))
            if not chunk:
                raise RuntimeError("The RCON connection closed during one packet.")
            data += chunk
        request_id, kind = struct.unpack("<ii", data[:8])
        return request_id, kind, data[8:].rstrip(b"\x00").decode("utf-8", errors="replace")

    def command(self, text: str) -> str:
        """Authenticate and run one command on the server console.

        A rejected password is answered with request id -1, not with text, so the
        id is what this client checks.
        """
        with socket.create_connection((self.host, self.port), timeout=15) as connection:
            self.request_id += 1
            connection.sendall(self._packet(self.request_id, 3, self.password))
            answer_id, _, _ = self._read(connection)
            if answer_id == -1:
                raise RuntimeError("The RCON password was rejected.")
            self.request_id += 1
            connection.sendall(self._packet(self.request_id, 2, text))
            return self._read(connection)[2]


def terminate(pid: int) -> None:
    """Stop one process and its children, and accept an already dead process."""
    if not OperatingSystem.is_windows():
        raise NotImplementedError("This tool is for the owner's Windows PC.")
    OperatingSystem.run_text(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, hidden=True)


def stop_stack() -> None:
    """Ask the server to save and stop, then stop both body process trees."""
    receipt = read_receipt()
    for body in receipt.bodies:
        terminate(body.mcp_pid)
    try:
        answer = Rcon("127.0.0.1", receipt.rcon_port, RCON_PASSWORD).command("stop")
        print(f"RCON stop accepted: {answer!r}", flush=True)
    except OSError as error:
        print(f"Server RCON is not reachable ({error}); terminating the server.", flush=True)
    terminate(receipt.server_pid)
    print("Stack stopped", flush=True)


def status() -> None:
    """Report each started process without changing anything."""
    receipt = read_receipt()
    server_alive = port_open("127.0.0.1", SERVER_PORT)
    print(f"server     port {SERVER_PORT} alive={server_alive} pid={receipt.server_pid}", flush=True)
    for body in receipt.bodies:
        alive = port_open("127.0.0.1", body.mcp_port)
        print(f"{body.id:14} {body.mcp_url} alive={alive} pid={body.mcp_pid} player={body.player}", flush=True)


def up() -> int:
    """Start the server and both bodies, then supervise them until they end."""
    for path in (SERVER_TEMPLATE, MINECRAFT_PYTHON, NODE_EXE, JAVA_EXE, BROWSER_EXE,
                 MINECRAFT_REPO / "body" / "dist" / "main.js"):
        if not path.exists():
            raise FileNotFoundError(f"Required path is missing: {path}")
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    require_free_port(SERVER_PORT, "the Minecraft server")
    require_free_port(RCON_PORT, "the Minecraft RCON console")
    for profile in BODIES:
        require_free_port(profile.mcp_port, f"body {profile.id}")
        require_free_port(profile.body_port, f"body {profile.id} internal HTTP")
        require_free_port(profile.viewer_port, f"body {profile.id} viewer")
    prepare_server()
    server = start_server()
    mcp: list[tuple[BodyProfile, subprocess.Popen[bytes]]] = []
    try:
        for profile in BODIES:
            process = start_body(profile)
            mcp.append((profile, process))
        for profile, process in mcp:
            wait_body(profile, process)
        receipt = StackReceipt(
            started_at=datetime.now(timezone.utc), server_dir=str(SERVER_DIR),
            server_pid=server.pid, server_log=str(SERVER_LOG), rcon_port=RCON_PORT,
            bodies=[BodyReceipt(id=profile.id, player=profile.player, mcp_url=profile.mcp_url,
                                mcp_port=profile.mcp_port, mcp_pid=process.pid,
                                env_path=str(profile.env_path), agent_home=str(profile.agent_home),
                                log_path=str(profile.log_path))
                    for profile, process in mcp],
        )
        atomic_json(RECEIPT, receipt.model_dump(mode="json"))
        print(f"Stack ready; receipt at {RECEIPT}", flush=True)
        while server.poll() is None:
            time.sleep(10.0)
        print(f"Server exited with code {server.returncode}", flush=True)
        return server.returncode
    finally:
        for _, process in mcp:
            if process.poll() is None:
                terminate(process.pid)


def main() -> int:
    """Run one command. Every failure keeps its traceback for the owner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("up", "stop", "status"))
    arguments = parser.parse_args()
    if arguments.command == "up":
        return up()
    if arguments.command == "stop":
        stop_stack()
        return 0
    status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
