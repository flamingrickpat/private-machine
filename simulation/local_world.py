"""Prepare the Fabric installation and own each process in one isolated simulation world."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
from tempfile import TemporaryDirectory
import time
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError
import yaml

from pm.paths import REPO_ROOT
from pm.utils.operating_system import OperatingSystem
from simulation.baseline_minecraft import WorldSetup
from workflows.minecraft_body.lease import BODY_LEASES, BodyEpochFloor


EXAMPLES = REPO_ROOT / "simulation/setup"


class SetupError(ValueError):
    """An installation or local configuration needs an owner correction."""


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def read(cls, path: Path) -> Self:
        try:
            return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8-sig")))
        except ValidationError as error:
            details = "; ".join(f"{'.'.join(map(str, issue['loc']))}: {issue['msg']}"
                                for issue in error.errors(include_input=False))
            raise SetupError(f"Edit {path}. Invalid {cls.__name__}: {details}") from error
        except yaml.YAMLError as error:
            raise SetupError(f"Edit {path}. The YAML could not be parsed.") from error


class BodySettings(Settings):
    browser: str
    node: str
    python: Path
    body_entrypoint: Path


class ServerSettings(Settings):
    java: str
    heap_mib: int = Field(ge=512)
    startup_timeout_seconds: int = Field(gt=0)


class PlayerService(BaseModel):
    participant: str
    username: str
    home: Path
    mcp_endpoint: str
    body_port: int
    viewer_port: int
    process_id: int


class ServiceReceipt(BaseModel):
    """Record the concrete world, server settings, installed mods and player endpoints."""

    world: Path
    seed: int
    game_port: int
    rcon_port: int
    server_pid: int
    mcp_revision: str | None
    mod_hashes: dict[str, str]  # pm-lint: allow PM004 -- digest by JAR filename
    players: list[PlayerService] = Field(default_factory=list)

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")  # pm-lint: allow PM007 -- service receipt serialization
        temporary.replace(path)


def seed_file(example: str, destination: Path) -> None:
    if not destination.exists():
        shutil.copyfile(EXAMPLES / example, destination)


def executable(value: str, field_name: str, config: Path) -> Path:
    selected = Path(value).expanduser()
    resolved = Path(shutil.which(str(selected)) or REPO_ROOT / selected)
    if not resolved.is_file():
        raise SetupError(f"Missing {field_name} executable: {resolved}. Set {field_name} in {config}.")
    return resolved.resolve()


@dataclass(slots=True)
class PortReservation:
    listener: socket.socket
    port: int

    @classmethod
    def available(cls) -> PortReservation:
        listener = socket.socket()
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind(("0.0.0.0", 0))
            return cls(listener, listener.getsockname()[1])
        except BaseException:
            listener.close()
            raise


@dataclass(slots=True)
class LocalWorld:
    root: Path
    work_root: Path
    installation: Path
    checkout: Path
    config: Path
    layout: WorldSetup
    reservations: list[PortReservation] = field(default_factory=list)
    children: list[subprocess.Popen[str]] = field(default_factory=list)
    logs: list[Path] = field(default_factory=list)

    def validate(self) -> None:
        """Create editable templates first so a failed check always has a repair destination."""
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("config_pm_minecraft.yaml", "config_minecraft_server.yaml", "server.properties",
                     "agentic_babymode.json", "pickup-range-server.json"):
            seed_file(name + ".example", self.root / name)
        body_path = self.root / "config_pm_minecraft.yaml"
        body = BodySettings.read(body_path)
        server_path = self.root / "config_minecraft_server.yaml"
        server = ServerSettings.read(server_path)
        executable(server.java, "java", server_path)
        executable(body.browser, "browser", body_path)
        node = executable(body.node, "node", body_path)
        version = subprocess.check_output([node, "--version"], text=True).strip()
        if int(version.lstrip("v").split(".")[0]) < 20:
            raise SetupError(f"Install Node 20 or later. Set node in {body_path}. Found {version}.")
        for path in (self.checkout / "main.py", self.checkout / body.python, self.checkout / body.body_entrypoint,
                     self.checkout / "body/node_modules"):
            if not path.exists():
                raise SetupError(f"Missing {path}. Run setup.ps1 in the pm-minecraft checkout. "
                                 f"Check mcp_checkout in {self.config} and the paths in {body_path}.")
        for name in ("fabric-server-launch.jar", "server.jar", "fabric-server-launcher.properties", "libraries", "eula.txt"):
            if not (self.installation / name).exists():
                raise SetupError(f"Missing {self.installation / name}. Set server_installation in {self.config} "
                                 "to an installed Fabric 1.19.4 server.")
        for pattern in ("agentic-babymode*.jar", "fabric-api*.jar"):
            if not list((self.installation / "mods").glob(pattern)):
                raise SetupError(f"Install {pattern} in {self.installation / 'mods'} or edit {self.config}.")
        if "eula=true" not in (self.installation / "eula.txt").read_text().splitlines():
            raise SetupError(f"Review and accept the Minecraft EULA in {self.installation / 'eula.txt'}.")
        for name in ("agentic_babymode.json", "pickup-range-server.json"):
            path = self.root / name
            try:
                document = json.loads(path.read_text(encoding="utf-8-sig"))
            except ValueError as error:
                raise SetupError(f"Invalid mod configuration in {path}: {error}") from error
            if not isinstance(document, dict):
                raise SetupError(f"Expected a JSON configuration object in {path}.")
        self.properties(1, 2, "preflight")

    def properties(self, game: int, rcon: int, password: str) -> str:
        path = self.root / "server.properties"
        template = path.read_text(encoding="utf-8-sig")
        for key in ("game_port", "rcon_port", "rcon_password", "world_seed"):
            if "{" + key + "}" not in template:
                raise SetupError(f"Restore the {{{key}}} placeholder in {path}.")
        rendered = template.replace("{game_port}", str(game)).replace("{rcon_port}", str(rcon))
        rendered = rendered.replace("{rcon_password}", password).replace("{world_seed}", str(self.layout.seed))
        required = {"level-name": "simulation-world", "server-port": str(game), "rcon.port": str(rcon),
                    "level-seed": str(self.layout.seed), "rcon.password": password, "enable-rcon": "true"}
        for key, expected in required.items():
            matches = [line.partition("=")[2] for line in rendered.splitlines() if line.startswith(key + "=")]
            if matches != [expected]:
                raise SetupError(f"Restore the {key} setting from {EXAMPLES / 'server.properties.example'} in {path}.")
        return rendered

    def stage_server(self) -> Path:
        self.work_root.mkdir(parents=True, exist_ok=True)
        server = self.work_root / "server"
        if not server.exists():
            with TemporaryDirectory(dir=self.work_root) as temporary:
                staged = Path(temporary) / "server"
                staged.mkdir()
                for name in ("fabric-server-launch.jar", "server.jar", "fabric-server-launcher.properties", "eula.txt"):
                    shutil.copy2(self.installation / name, staged / name)
                for name in ("mods", "libraries", "versions"):
                    source = self.installation / name
                    if source.is_dir():
                        shutil.copytree(source, staged / name)
                staged.rename(server)
        (server / "config").mkdir(exist_ok=True)
        for name in ("agentic_babymode.json", "pickup-range-server.json"):
            shutil.copyfile(self.root / name, server / "config" / name)
        return server

    def reserve(self) -> PortReservation:
        reservation = PortReservation.available()
        self.reservations.append(reservation)
        return reservation

    def spawn(self, command: list[str], directory: Path, log: Path, environment: dict[str, str] | None = None) -> None:  # pm-lint: allow PM004 -- OS environment by variable name
        with log.open("w", encoding="utf-8") as output:
            child = subprocess.Popen(command, cwd=directory, stdin=subprocess.PIPE, stdout=output,
                                     stderr=subprocess.STDOUT, env=environment, text=True,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        self.children.append(child)
        self.logs.append(log)

    def check_processes(self) -> None:
        for child, log in zip(self.children, self.logs, strict=True):
            if child.poll() is not None:
                raise SetupError(f"Service exited with code {child.returncode}. Open {log}. Edit settings in {self.root}.")

    def start(self) -> None:
        """Boot the copied server before starting each local MCP with its own body and viewer."""
        server = self.stage_server()
        server_path = self.root / "config_minecraft_server.yaml"
        settings = ServerSettings.read(server_path)
        game, rcon = self.reserve(), self.reserve()
        password = secrets.token_hex(24)
        (server / "server.properties").write_text(self.properties(game.port, rcon.port, password), encoding="ascii")
        game.listener.close()
        rcon.listener.close()
        log = self.work_root / "server.log"
        self.spawn([str(executable(settings.java, "java", server_path)), f"-Xmx{settings.heap_mib}M",
                    "-jar", "fabric-server-launch.jar", "nogui"], server, log)
        hashes: dict[str, str] = {}  # pm-lint: allow PM004 -- installed mod digest by JAR filename
        for jar in sorted((server / "mods").glob("*.jar")):
            with jar.open("rb") as stream:
                hashes[jar.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        revision = subprocess.run(["git", "-C", str(self.checkout), "rev-parse", "HEAD"],
                                  capture_output=True, text=True)
        receipt = ServiceReceipt(world=server / "simulation-world", seed=self.layout.seed, game_port=game.port,
                                 mcp_revision=revision.stdout.strip() if revision.returncode == 0 else None,
                                 rcon_port=rcon.port, server_pid=self.children[0].pid, mod_hashes=hashes)  # pm-lint: allow PM005 -- map of mod filenames to SHA-256
        receipt_path = self.work_root / "services.json"
        receipt.save(receipt_path)
        print(f"Waiting for Fabric on port {game.port}. Server log: {log}", flush=True)
        deadline = time.monotonic() + settings.startup_timeout_seconds
        while "Done (" not in log.read_text(encoding="utf-8", errors="replace"):
            self.check_processes()
            if time.monotonic() >= deadline:
                raise SetupError(f"Server startup timed out. Open {log}. Settings: {server_path}")
            time.sleep(0.5)
        body_path = self.root / "config_pm_minecraft.yaml"
        settings_body = BodySettings.read(body_path)
        for body in self.layout.bodies:
            mcp, http, viewer = self.reserve(), self.reserve(), self.reserve()
            home = self.work_root / "players" / body.participant
            home.mkdir(parents=True, exist_ok=True)
            env_path = home / "body.env"
            values = [
                f"NODE_EXE={executable(settings_body.node, 'node', body_path)}",
                f"BROWSER_EXECUTABLE_PATH={executable(settings_body.browser, 'browser', body_path)}",
                f"MCMCP_BODY_ENTRYPOINT={self.checkout / settings_body.body_entrypoint}",
                f"MCMCP_AGENT_HOME={home}", "MCP_HOST=0.0.0.0", f"MCP_PORT={mcp.port}",
                "MCMCP_BODY_HOST=127.0.0.1", f"MCMCP_BODY_PORT={http.port}", f"MCMCP_VIEWER_PORT={viewer.port}",
                "MINECRAFT_HOST=127.0.0.1", f"MINECRAFT_PORT={game.port}", f"MINECRAFT_PLAYER={body.username}",
                "MCMCP_CONNECTION_TOOLS=true", "MINECRAFT_RCON_HOST=127.0.0.1", f"MINECRAFT_RCON_PORT={rcon.port}", f"MINECRAFT_RCON_PASSWORD={password}",
            ]
            env_path.write_text("\n".join(values) + "\n", encoding="utf-8")
            environment = os.environ.copy()
            for key in list(environment):
                if key.startswith(("MINECRAFT_", "MCMCP_", "MCP_")) or key in {"NODE_EXE", "BROWSER_EXECUTABLE_PATH"}:
                    del environment[key]
            body.server = f"127.0.0.1:{game.port}"
            body.endpoint = f"http://127.0.0.1:{mcp.port}/mcp"
            fence = home / "body-epoch.json"
            if fence.is_file():
                BODY_LEASES.preserve_epoch(body.endpoint, BodyEpochFloor.model_validate_json(fence.read_bytes()))
            for reservation in (mcp, http, viewer):
                reservation.listener.close()
            self.spawn([str(self.checkout / settings_body.python), str(self.checkout / "main.py"), str(env_path)],
                       self.checkout, home / "mcp.log", environment)
            receipt.players.append(PlayerService(participant=body.participant, username=body.username, home=home,
                                   mcp_endpoint=body.endpoint, body_port=http.port, viewer_port=viewer.port,
                                   process_id=self.children[-1].pid))
            receipt.save(receipt_path)
            print(f"Waiting for {body.username} at {body.endpoint}. MCP log: {home / 'mcp.log'}", flush=True)
            deadline = time.monotonic() + settings.startup_timeout_seconds
            while True:
                self.check_processes()
                try:
                    with socket.create_connection(("127.0.0.1", mcp.port), timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise SetupError(f"MCP startup timed out. Open {home / 'mcp.log'}. Settings: {body_path}")
                    time.sleep(.5)

    def record_run_setup(self, run_root: Path) -> None:
        """Keep the effective rules and port receipt in exported evidence. The RCON password stays local."""
        destination = run_root / "minecraft-setup"
        destination.mkdir()
        shutil.copyfile(self.work_root / "services.json", destination / "services.json")
        shutil.copyfile(self.root / "server.properties", destination / "server.properties.example")
        for name in ("agentic_babymode.json", "pickup-range-server.json"):
            shutil.copyfile(self.work_root / "server/config" / name, destination / name)

    def close(self) -> None:
        """Stop owned player trees and let Java save the world before exiting."""
        for reservation in self.reservations:
            reservation.listener.close()
        for index in reversed(range(len(self.children))):
            child = self.children[index]
            if child.poll() is not None:
                continue
            if index == 0 and child.stdin is not None:
                try:
                    child.stdin.write("stop\n")
                    child.stdin.flush()
                    child.wait(timeout=45)
                    continue
                except (OSError, subprocess.TimeoutExpired) as error:
                    print(f"Java did not finish its save and stop command: {error}. Closing the owned process tree.", flush=True)
            OperatingSystem.kill_process_tree(child)
            child.wait()
