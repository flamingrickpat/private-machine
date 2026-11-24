"""Compact, deterministic orientation about this PM instance and its currently assigned work."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
import platform
from shutil import which

from pm.introspection.grants import git_revision
from pm.utils.operating_system import OperatingSystem


@dataclass(frozen=True, slots=True)
class MachineIdentity:
    host: str
    cpu: str
    ram_gib: int
    gpu: str
    revision: str


@cache
def machine_identity() -> MachineIdentity:
    """Read static host characteristics once per process."""
    ram = OperatingSystem.run_text(["powershell", "-NoProfile", "-Command",
                          "[int][math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)"],
                         hidden=True, check=True)
    root = Path(__file__).resolve().parents[2]
    revision = git_revision(root)[:7] or "unknown"
    gpu = "unknown"
    utility = which("nvidia-smi")
    if utility:
        found = OperatingSystem.run_text([utility, "--query-gpu=name", "--format=csv,noheader"], hidden=True)
        if found.returncode == 0:
            gpu = ", ".join(found.stdout.splitlines())
    return MachineIdentity(platform.node(), OperatingSystem.environment.get("PROCESSOR_IDENTIFIER") or platform.machine(),
                           int(ram.stdout.strip()), gpu, revision)


@dataclass(frozen=True, slots=True)
class RuntimeSelfState:
    machine: MachineIdentity
    capabilities: tuple[str, ...]
    workflows: tuple[str, ...]
    suites: tuple[str, ...]
    body_status: str
    active_work: tuple[str, ...]

    def render(self) -> str:
        m = self.machine
        return "\n".join((
            f"Your runtime: PM {m.revision} on {m.host}, {m.cpu}, {m.ram_gib} GiB RAM, GPU {m.gpu}.",
            "Capabilities: " + ", ".join(self.capabilities)[:700],
            "Available actions and workflows: " + ", ".join(self.workflows)[:700],
            "Configured external MCP suites: " + (", ".join(self.suites)[:400] or "none"),
            "Minecraft avatar: " + self.body_status,
            "Active work: " + ("; ".join(self.active_work)[:700] or "none recorded"),
        ))
