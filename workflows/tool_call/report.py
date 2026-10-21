"""Read, retain, and re-check the terminal report of one tool goal.

The workflow writes the report once at its terminal boundary
(:mod:`workflows.tool_call.steps`). A restart retains an ``interrupted``
report for a run that no worker finished (:func:`retain_interrupted_report`).
Before a public reply, :func:`reply_evidence` reads the report again and
re-checks every local file it names, so a reply can name a path only when its
bytes are still the reported bytes.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from pm.utils import yaml_util

from workflows.tool_call.schemas import REPORT_PATH, PinnedDispatch, ToolCallReport


class FileCheck(BaseModel):
    """The state of one reported local file at reply time."""

    model_config = ConfigDict(extra="forbid")

    path: str
    exists: bool
    unchanged: bool
    sha256: str = ""


def report_path(artifact_root: str | Path) -> Path:
    """Return the host report file of a run folder."""
    return Path(artifact_root) / REPORT_PATH


def dump_report(report: ToolCallReport) -> str:
    """Return the report as YAML, with the ``schema`` key."""
    return yaml_util.safe_dump(report.model_dump(mode="json", by_alias=True), allow_unicode=True, sort_keys=False)


def load_report(text: str) -> ToolCallReport:
    """Parse report YAML. Malformed reports raise; they are our own files."""
    return ToolCallReport.model_validate(yaml_util.safe_load(text))


def retain_interrupted_report(receipt: Mapping[str, Any]) -> None:
    """Write an ``interrupted`` report for a run that a restart found without one.

    A report that exists stays unchanged: it is the session's own terminal
    record, and a second call changes nothing. The run is not resumed and no
    external effect repeats. Effects of the stopped session stay uncertain.
    """
    path = report_path(receipt["artifact_path"])
    if path.is_file():
        return
    pinned = PinnedDispatch.model_validate(receipt["arguments"])
    summary = ("The tool session stopped when the application stopped. It was not resumed. "
               "Its external effects remain unverified.")
    report = ToolCallReport(
        run_id=receipt["run_id"], instruction=receipt["instructions"], tool_goal=pinned.tool_goal,
        outcome="interrupted", summary=summary, rejection_reason=summary, action_id=receipt["action_id"],
        action_schema_id=pinned.action_schema_id, correlation_id=receipt["parent_correlation"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_report(report), encoding="utf-8")


def _check(root: Path, relative: str, sha256: str) -> FileCheck:
    path = root / relative
    if not path.is_file():
        return FileCheck(path=relative, exists=False, unchanged=False)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    return FileCheck(path=relative, exists=True, unchanged=actual == sha256, sha256=actual)


def reply_evidence(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Re-read the report and its local files before a public reply.

    Returns ``report`` (the typed report; it replaces the raw report of
    ``task_evidence``), ``file_checks`` (one entry for
    each reported local file and tool image), and ``task_verified: false``.
    Unchanged bytes allow a truthful statement about a local file. They do
    not verify the goal, the image content, or its quality. A run without a
    report (for example a rejected start) adds nothing.
    """
    root = Path(receipt["artifact_path"]) if receipt["artifact_path"] else None
    if root is None or not report_path(root).is_file():
        return {}
    report = load_report(report_path(root).read_text(encoding="utf-8"))
    checks = [_check(root, item.claim.path, item.sha256) for item in report.artifacts
              if item.claim.path is not None and item.file_verified and item.sha256 is not None]
    checks += [_check(root, image.path, image.sha256) for image in report.tool_images]
    return {"report": report.model_dump(mode="json", by_alias=True),
            "file_checks": [check.model_dump(mode="json") for check in checks],
            "task_verified": False}


__all__ = ["FileCheck", "dump_report", "load_report", "reply_evidence", "report_path", "retain_interrupted_report"]
