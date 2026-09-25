"""Working-target adapters: a virtual Ghost Bash environment and a real folder.

The virtual environment exposes a read-only projection of the Ghost as
``/work/ghost/knoxels/<id>.yaml`` files inside a fresh in-memory BashMachine,
plus writable ``/work/inbox``, ``/work/outbox``, and ``/work/scratch``. The
Ghost stays in RAM; the machine's content map resolves ``cat`` lazily from the
snapshot instead of copying every knoxel. The real-folder target runs trusted
host Bash in one temp directory per invocation.
"""
from __future__ import annotations

import posixpath
import shlex
import shutil
import subprocess
from pathlib import Path

import yaml

from pm_bash_machine import Access, BashMachine
from pm_workflows.working_target import (
    CommandResult,
    WorkingEnvironment,
    WorkingTarget,
)


class _LazyGhostFile:
    """A BashMachine content entry that renders YAML only on first read."""

    def __init__(self, data: dict):
        self._data = data
        self._rendered: str | None = None

    def content(self) -> str:
        if self._rendered is None:
            self._rendered = yaml.safe_dump(
                self._data, sort_keys=False, allow_unicode=True,
            )
        return self._rendered

def _index_entry(knoxel_id: int, record: dict) -> dict:
    data = record["data"]
    summary = str(data.get("content", ""))
    if len(summary) > 80:
        summary = summary[:77] + "..."
    return {
        "id": knoxel_id,
        "type": record["__type__"],
        "summary": summary,
    }


class GhostBashEnvironment(WorkingEnvironment):
    """One read-only Ghost projection inside a fresh in-memory BashMachine."""

    def __init__(
        self,
        visible_knoxels: dict[int, dict],
        visible_gwt: dict[int, dict],
        mode: str,
    ):
        self.visible_knoxels = visible_knoxels
        self.visible_gwt = visible_gwt
        self.machine = BashMachine()

        # Host-only skeleton setup in one bash call. The workflow user's first
        # ``cd /work`` needs /work to exist; inbox/outbox/scratch hold traffic.
        result = self.machine.exec("user",
            "mkdir -p /work/inbox /work/outbox /work/scratch "
            "/work/ghost/knoxels /work/ghost/gwt"
        )
        if result.exit_code != 0:
            raise RuntimeError(result.stderr)

        index = {"mode": mode, "knoxels": []}
        files = {}
        for knoxel_id in sorted(visible_knoxels):
            record = visible_knoxels[knoxel_id]
            path = f"/work/ghost/knoxels/{knoxel_id}.yaml"
            files[path] = _LazyGhostFile(record)
            index["knoxels"].append(_index_entry(knoxel_id, record))
        for tick in sorted(visible_gwt):
            path = f"/work/ghost/gwt/{tick}.yaml"
            files[path] = _LazyGhostFile(visible_gwt[tick])
        files["/work/ghost/index.yaml"] = _LazyGhostFile(index)
        self.machine.write_texts(files, access=Access.R)

        self.machine.add_user("workflow", cwd="/work")

    @staticmethod
    def _abs(path: str) -> str:
        return posixpath.join("/work", path)

    def read_text(self, path: str) -> str:
        result = self.exec(f"cat -- {shlex.quote(self._abs(path))}")
        if result.exit_code != 0:
            raise RuntimeError(result.stderr or f"read failed: {path}")
        return result.stdout

    def write_text(self, path: str, text: str) -> None:
        target = self._abs(path)
        parent = posixpath.dirname(target)
        command = (
            f"mkdir -p -- {shlex.quote(parent)} && "
            f"printf '%s' {shlex.quote(text)} > {shlex.quote(target)}"
        )
        result = self.exec(command)
        if result.exit_code != 0:
            raise RuntimeError(result.stderr or f"write failed: {path}")

    def move(self, source: str, destination: str) -> None:
        src = self._abs(source)
        dst = self._abs(destination)
        command = (
            f"mkdir -p -- {shlex.quote(posixpath.dirname(dst))} && "
            f"mv -- {shlex.quote(src)} {shlex.quote(dst)}"
        )
        result = self.exec(command)
        if result.exit_code != 0:
            raise RuntimeError(result.stderr or f"move failed: {source} -> {destination}")

    def list_files(self, directory: str) -> list[str]:
        target = self._abs(directory)
        command = f"find {shlex.quote(target)} -maxdepth 1 -type f"
        result = self.exec(command)
        if result.exit_code != 0:
            raise RuntimeError(result.stderr)
        out: list[str] = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line or not line.startswith("/work"):
                continue
            out.append(posixpath.relpath(line, "/work").replace("\\", "/"))
        return sorted(out)

    def exec(self, command: str) -> CommandResult:
        result = self.machine.exec("workflow", command, strip_output=False)
        return CommandResult(
            stdout=result.stdout, stderr=result.stderr, exit_code=result.exit_code,
        )


class GhostBashWrapper(WorkingTarget):
    """Opens a fresh read-only Ghost Bash environment per invocation."""

    def __init__(self, ghost, mode: str = "read", visible_ids=None):
        self.ghost = ghost
        self.mode = mode
        self.visible_ids = visible_ids

    def open(self, invocation_id: str) -> GhostBashEnvironment:
        view = self.ghost.snapshot(self.visible_ids)
        return GhostBashEnvironment(view.serialize_knoxels(), view.serialize_gwt(), self.mode)


class FolderEnvironment(WorkingEnvironment):
    """A real directory used as one invocation-scoped working target."""

    def __init__(self, root: Path, bash_executable: str):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.bash_executable = bash_executable

    def _path(self, path: str) -> Path:
        return self.root / path

    def read_text(self, path: str) -> str:
        return self._path(path).read_text(encoding="utf-8")

    def write_text(self, path: str, text: str) -> None:
        target = self._path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def move(self, source: str, destination: str) -> None:
        source_path = self._path(source)
        destination_path = self._path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(destination_path))

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


__all__ = [
    "GhostBashWrapper",
    "GhostBashEnvironment",
    "FolderTarget",
    "FolderEnvironment",
]
