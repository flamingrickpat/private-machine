"""Read-only memory view as a virtual Bash environment.

The view exposes the memory store as ``/work/ghost/knoxels/<id>.yaml`` files
inside a fresh in-memory BashMachine, plus writable ``/work/inbox``,
``/work/outbox``, and ``/work/scratch``. The store stays in RAM; ``cat``
resolves each record lazily from the snapshot instead of copying it.
"""
from __future__ import annotations

import posixpath
import shlex
from pathlib import Path

import yaml

from pm_bash_machine import Access, BashMachine
from pm_workflows.working_target import AgentTarget, CommandResult, WorkingEnvironment, WorkingTarget


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

    def agent_target(self, artifact_workspace: Path) -> AgentTarget:
        return AgentTarget(artifact_workspace, self.machine, "workflow")

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
        from pm_workflows.inputs import relative_path
        return posixpath.join("/work", relative_path(path, folder=True))

    def read_text(self, path: str) -> str:
        # Use the user's view so missing data keeps its distinct INVALID cause.
        if not self.machine.is_file_as("workflow", self._abs(path)):
            raise FileNotFoundError(path)
        result = self.exec(f"cat -- {shlex.quote(self._abs(path))}")
        if result.exit_code != 0:
            raise RuntimeError(result.stderr or f"read failed: {path}")
        return result.stdout

    def write_text(self, path: str, text: str) -> None:
        # Write through the user's filesystem view directly. Routing arbitrary
        # model text through a shell ``printf`` mangles backticks and nested
        # quotes; the direct write keeps the exact text and still enforces the
        # read-only projection through ``_need_write``.
        self.machine.write_text_as("workflow", self._abs(path), text)

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

    def write_binary(self, path: str, data: bytes) -> None:
        """Write exact bytes through the same read-only-projected view."""
        target = self._abs(path)
        self.machine.write_text_as("workflow", target, "")  # enforces write ACL
        self.machine.write_binary(target, data)

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
        environment = GhostBashEnvironment(view.serialize_knoxels(), view.serialize_gwt(), self.mode)
        environment.ghost_view = view
        return environment


__all__ = ["GhostBashEnvironment", "GhostBashWrapper"]
