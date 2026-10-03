"""Run the existing narrow body gate on the user's real Babymode server."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.live.minecraft_stack import MinecraftStack, _wait_log, load_env_file


class RealServerStack(MinecraftStack):
    """Own test processes while retaining the actual server world and logs."""

    def __init__(self, server_root: Path, out: Path, env_file: Path):
        super().__init__(env_file=env_file, run_root=out)
        self.server_root = server_root
        self.run_dir = out / "services"
        self.run_dir.mkdir()

    def _start_server(self, timeout: float) -> None:
        log_path = self.run_dir / "server.log"
        with log_path.open("w", encoding="utf-8") as log:
            memory = self.values.get("SERVER_MEMORY", "2G")
            self.server = subprocess.Popen([self.values["JAVA_EXE"], f"-Xms{memory}", f"-Xmx{memory}",
                "-jar", str(self.server_root / "fabric-server-launch.jar"), "nogui"],
                cwd=self.server_root, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        _wait_log(log_path, "Done (", timeout, "The real Minecraft server")

    def stop(self) -> None:
        try:
            if self.mcp is not None and self.mcp.poll() is None:
                try:
                    self._control("/test/body/stop", timeout=30)
                finally:
                    self.mcp.terminate()
                    self._reap(self.mcp, 15)
        finally:
            if self.server is not None and self.server.poll() is None:
                self.server.stdin.write("stop\n")
                self.server.stdin.flush()
                self._reap(self.server, 60)
                self.server.stdin.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--server", type=Path, default=Path("C:/source/minecraft-server-real"))
    parser.add_argument("--reset-world", action="store_true", help="Run the server's wipe-world.ps1 before startup.")
    args = parser.parse_args()
    out = args.out.resolve()
    server = args.server.resolve()
    if not out.is_relative_to(ROOT / ".runtime"):
        raise ValueError("Keep the test character and receipts under .runtime.")
    out.mkdir(parents=True, exist_ok=False)
    properties = load_env_file(server / "server.properties")
    if properties["level-name"] != "world":
        raise ValueError("The wipe script only resets the world directory; inspect the selected level-name.")
    source_env = ROOT.parent / "pm-minecraft/.env_real"
    values = load_env_file(source_env)
    values.update({"MINECRAFT_PORT": properties["server-port"], "MINECRAFT_RCON_PORT": properties["rcon.port"],
        "MINECRAFT_RCON_PASSWORD": properties["rcon.password"], "MINECRAFT_PLAYER": "tdd_player",
        "MCMCP_TEST_MODE": "true", "MCMCP_AGENT_HOME": str(out / "character"),
        "MCMCP_INSPECTOR_ENABLED": "false",
        "JAVA_EXE": "C:/Users/rick/AppData/Roaming/PrismLauncher/java/java-runtime-gamma/bin/java.exe",
        "PYTHON_EXE": str(ROOT.parent / "pm-minecraft/.venv/Scripts/python.exe"),
        "TEST_RUN_ROOT": str(out), "SERVER_MEMORY": "2G", "SERVER_START_TIMEOUT_SECONDS": "180"})
    for key in ("MINECRAFT_PORT", "MINECRAFT_RCON_PORT", "MCP_PORT", "MCMCP_BODY_PORT", "MCMCP_VIEWER_PORT"):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", int(values.get(key, {"MCP_PORT": "6767", "MCMCP_BODY_PORT": "6768", "MCMCP_VIEWER_PORT": "6769"}.get(key)))))
    env_path = out / ".env.alpha"
    env_path.write_text("\n".join(f"{key}={value}" for key, value in values.items()) + "\n", encoding="utf-8")
    receipt = {"server": str(server), "reset_world": args.reset_world,
        "seed": properties["level-seed"], "level_type": properties["level-type"],
        "structures": properties["generate-structures"], "server_properties_sha256": hashlib.sha256((server / "server.properties").read_bytes()).hexdigest(),
        "wipe_script_sha256": hashlib.sha256((server / "wipe-world.ps1").read_bytes()).hexdigest(),
        "character_directory": str(out / "character"), "scope": "existing W8 one-action gate; no house or iron-pickaxe claim"}
    stack = RealServerStack(server, out, env_path)
    try:
        if args.reset_world:
            for name in ("world", "logs", ".fabric"):
                target = (server / name).resolve()
                if not target.is_relative_to(server) or target == server:
                    raise ValueError(f"The wipe target leaves the selected server directory: {target}")
            subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(server / "wipe-world.ps1")], check=True)
        stack.start()
        environment = os.environ.copy()
        environment.update({"PM_MODEL_CONFIG": str(args.config.resolve()), "MINECRAFT_MCP_URL": stack.mcp_url,
            "PM_MINECRAFT_AUTOSTART": "0"})
        result = subprocess.run([sys.executable, "-m", "unittest", "tests.live.test_minecraft_body", "-v", "-b"],
            cwd=ROOT, env=environment)
        receipt.update({"exit_code": result.returncode, "status": "passed" if result.returncode == 0 else "failed"})
        if result.returncode:
            raise RuntimeError("The existing Minecraft gate failed. Inspect the gate and service logs.")
    except BaseException as error:
        receipt.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        try:
            stack.stop()
        finally:
            (out / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            print(out / "receipt.json", flush=True)


if __name__ == "__main__":
    main()
