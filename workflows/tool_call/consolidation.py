"""The memory record of one terminal tool goal.

:class:`workflows.memory_consolidation.artifact_consolidation.Consolidator`
calls :func:`tool_call_outcome` for a run whose ``outbox/report.yaml`` is a
``pm.tool-call-report.v1`` report. The record replaces the generic outcome
record of that run. It keeps:

- the outcome, the summary, the agent's own quality judgment, and its
  uncertainty, in words that the companion reads later;
- the complete typed report in ``metadata.tool_call_report``;
- the links to the action that started the run (``action_id``,
  ``source_action_schema_id``, ``source_tick_id``, ``correlation_id``,
  ``stimulus_ids``);
- every image that a tool returned or that the agent kept as a verified
  local file, through :mod:`pm.records.media`, with ``provenance.source``
  ``tool``, the server, the tool, the run, and the artifact path. The record
  content holds their ``[image:<id>]`` tags, so the companion can see them
  later.

The content is deterministic for the same report and files. The consolidator
keys its memories by content and artifact digests, so a repeated terminal
hook or a restart creates no second record.
"""
from __future__ import annotations

from pathlib import Path

from pm.records.enums import FeatureType, InterlocusType
from pm.records.media import MediaProvenance, RecordImage, attach_images, image_from_bytes, with_tags
from pm.records.memory_records import Feature

from workflows.tool_call.report import load_report
from workflows.tool_call.schemas import REPORT_SCHEMA, ToolCallReport

#: Outcome words for the companion's later memory.
OUTCOME_WORDS = {
    "succeeded": "it worked",
    "failed": "it did not work out",
    "blocked": "it could not start",
    "uncertain": "the result is uncertain",
    "cancelled": "it was cancelled",
    "interrupted": "it was interrupted when the application stopped",
}


def is_tool_call_report(data: object) -> bool:
    """Return True when parsed report YAML is a tool-call report."""
    return isinstance(data, dict) and "schema" in data and data["schema"] == REPORT_SCHEMA


def _images(report: ToolCallReport, root: Path) -> tuple[list[RecordImage], list[str]]:
    """Return the images of a report and the notes about files that cannot become images.

    A local artifact becomes an image only when the host verified its bytes
    and the bytes still have the reported digest. A file that is not a
    supported image, or is too large, stays a file and gets a note.
    """
    images: list[RecordImage] = []
    notes: list[str] = []
    candidates = [(image.path, image.sha256, image.server, image.tool) for image in report.tool_images]
    candidates += [(item.claim.path, item.sha256, item.claim.server, "") for item in report.artifacts
                   if item.claim.path is not None and item.file_verified and item.sha256 is not None]
    for path, digest, server, tool in candidates:
        data = (root / path).read_bytes()
        try:
            image = image_from_bytes(data, provenance=MediaProvenance(
                source="tool", run_id=report.run_id, artifact_path=path, server=server, tool=tool))
        except ValueError as error:
            # A text, audio, or oversized file is an ordinary artifact, not a defect.
            notes.append(f"{path} is kept as a file, not as an image ({error})")
            continue
        if image.sha256 != digest:
            notes.append(f"{path} changed after the session; its image is not kept")
            continue
        # The same bytes from a tool result and a saved file are one image.
        if image.id not in {known.id for known in images}:
            images.append(image)
    return images, notes


def tool_call_outcome(run_id: str, report_text: str, root: Path) -> Feature:
    """Return the outcome record of one tool-call report. The caller commits it.

    Raises when the report is malformed or a reported file is missing:
    both are our own run files, so either one is a bug or a damaged run
    folder that must stay visible.
    """
    report = load_report(report_text)
    lines = [f"Tool work (run {run_id}) on the decision: {report.instruction}"]
    if report.tool_goal:
        lines.append(f"Tool goal: {report.tool_goal}")
    lines.append(f"Outcome: {OUTCOME_WORDS[report.outcome]} ({report.outcome}). {report.summary}")
    agent = report.agent_outcome
    if agent is not None and agent.quality_assessment:
        lines.append(f"Own judgment of the result: {agent.quality_assessment}")
    if agent is not None:
        lines += [f"Observed: {item}" for item in agent.observations]
        lines += [f"Uncertain: {item}" for item in agent.uncertainty]
    if report.tool_calls:
        lines.append("Tools used: " + ", ".join(dict.fromkeys(f"{call.server}.{call.tool}" for call in report.tool_calls)))
    for item in report.artifacts:
        where = item.claim.path or item.claim.uri
        lines.append(f"Artifact {where}: {item.reason}")
    images, notes = _images(report, root)
    lines += notes
    if not report.verified:
        lines.append("Nobody verified that the goal itself was met.")
    metadata = {
        "run_id": run_id, "kind": "outcome", "memory_kind": "artifact_report", "factual_promotion": False,
        "tool_call_report": report.model_dump(mode="json", by_alias=True),
        "action_id": report.action_id, "source_action_schema_id": report.action_schema_id,
        "source_tick_id": report.source_tick_id, "correlation_id": report.correlation_id,
        "stimulus_ids": list(report.stimulus_ids),
    }
    tags = attach_images(metadata, images)
    return Feature(
        content=with_tags("\n".join(lines), tags),
        feature_type=FeatureType.WorldEvent,
        interlocus=InterlocusType.PrivateInternal,
        causal=True,
        override_source="memory_consolidation",
        metadata=metadata,
    )


__all__ = ["OUTCOME_WORDS", "is_tool_call_report", "tool_call_outcome"]
