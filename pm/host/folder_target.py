"""Host-folder working target: one real directory per workflow invocation.

The target runs trusted host Bash in that directory. Use it only for
workflows that are allowed to touch host files.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


from pm_workflows.working_target import (
    AgentTarget,
    CommandResult,
    WorkingEnvironment,
    WorkingTarget,
    HostInputEnvironment,
)
from pm_workflows.inputs import FileMetadata


class FolderEnvironment(WorkingEnvironment):
    """A real directory used as one invocation-scoped working target."""

    def agent_target(self, artifact_workspace: Path) -> AgentTarget:
        return AgentTarget(self.root.resolve())

    def __init__(self, root: Path, bash_executable: str):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.bash_executable = bash_executable

    def _path(self, path: str) -> Path:
        return HostInputEnvironment(self.root)._path(path, folder=True)

    def read_text(self, path: str) -> str:
        return HostInputEnvironment(self.root).read_text(path)

    def file_metadata(self, path: str) -> FileMetadata:
        """Return actual host stat identity and an exact-byte digest."""
        return HostInputEnvironment(self.root).file_metadata(path)

    def check_file_boundary(self, path: str, folder: str) -> None:
        """Apply the declared folder boundary to the resolved host file."""
        HostInputEnvironment(self.root).check_file_boundary(path, folder)

    def canonical_relative_path(self, path: str) -> str:
        """Expose host symlink targets for the kernel's path grant checks."""
        return HostInputEnvironment(self.root).canonical_relative_path(path)

    def write_text(self, path: str, text: str) -> None:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="")

    def move(self, source: str, destination: str) -> None:
        source_path = self._path(source)
        destination_path = self._path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(destination_path))

    def write_binary(self, path: str, data: bytes) -> None:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    def list_files(self, directory: str) -> list[str]:
        root = self._path(directory)
        if not root.is_dir():
            return []
        prefix = directory.rstrip("/")
        if prefix:
            prefix += "/"
        return sorted(prefix + p.name for p in root.iterdir() if p.is_file())

    def exec(self, command: str) -> CommandResult:
        result = subprocess.run(
            [self.bash_executable, "--noprofile", "--norc", "-c", command],
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return CommandResult(
            stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode,
        )


class FolderTarget(WorkingTarget):
    """Opens one real folder under ``root`` for each invocation."""

    def __init__(self, root: Path, bash_executable: str):
        self.root = Path(root)
        self.bash_executable = bash_executable

    def open(self, invocation_id: str) -> FolderEnvironment:
        return FolderEnvironment(self.root / invocation_id, self.bash_executable)


__all__ = ["FolderEnvironment", "FolderTarget"]
