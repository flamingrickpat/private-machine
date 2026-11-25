"""Start and stop a real Fabric server alongside the live Minecraft MCP."""
from __future__ import annotations

from pathlib import Path

from pm.utils.operating_system import OperatingSystem
from tests.live.minecraft_stack import MinecraftStack, _wait_log


class RealServerStack(MinecraftStack):
    """Own test processes while retaining the actual server world and logs."""

    def __init__(self, server_root: Path, out: Path, env_file: Path) -> None:
        super().__init__(env_file=env_file, run_root=out)
        self.server_root = server_root
        self.run_dir = out / "services"
        self.run_dir.mkdir()

    def _start_server(self, timeout: float) -> None:
        log_path = self.run_dir / "server.log"
        with log_path.open("w", encoding="utf-8") as log:
            memory = self.values.get("SERVER_MEMORY", "2G")
            self.server = OperatingSystem.start_text_process(
                [self.values["JAVA_EXE"], f"-Xms{memory}", f"-Xmx{memory}",
                "-jar", str(self.server_root / "fabric-server-launch.jar"), "nogui"],
                cwd=self.server_root,
                stdin=OperatingSystem.pipe,
                stdout=log,
                stderr=OperatingSystem.merged_output,
                hidden=True
            )
        _wait_log(log_path, "Done (", timeout, "The real Minecraft server")

    def stop(self) -> None:
        try:
            if self.mcp is not None and self.mcp.poll() is None:
                try:
                    self._control("/test/body/stop", timeout=30)
                finally:
                    OperatingSystem.terminate_process(self.mcp)
                    self._reap(self.mcp, 15)
        finally:
            if self.server is not None and self.server.poll() is None:
                self.server.stdin.write("stop\n")
                self.server.stdin.flush()
                self._reap(self.server, 60)
                self.server.stdin.close()
