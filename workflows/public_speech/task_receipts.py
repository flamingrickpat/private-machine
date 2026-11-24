from __future__ import annotations

from datetime import datetime
from collections import deque
from pathlib import Path
from pm.context_render.budget import evidence_excerpt
from pydantic import BaseModel, ConfigDict, Field, field_validator, JsonValue, ValidationError
from workflows.minecraft_body.world_text import MinecraftToolResult
from workflows.minecraft.execution_contracts import ActorAcceptance
from workflows.tool_call.schemas import ToolCallReport
from workflows.tool_call.report import FileCheck


class TaskNoteEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    path: str
    text: str


class TaskVerificationEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    verified: bool
    reason: str = ""
    scope: str = ""
    checks: dict[str, bool] = Field(default_factory=dict[str, bool])
    accepted_nodes: tuple[str, ...] = ()
    goal_id: str = ""
    tree_revision: int | None = None
    tree_bases: tuple[tuple[int, int, int], ...] = ()
    trees: int | None = None
    initial_logs: int | None = None
    final_logs: int | None = None
    maximum_travel: float | None = None
    return_distance: float | None = None


class TaskReportHeader(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True, serialize_by_alias=True)
    schema_name: str = Field(default="", validation_alias="schema", serialization_alias="schema")


class TaskReportEvidence(TaskReportHeader):
    model_config = ConfigDict(extra="ignore", frozen=True)
    summary: str = ""
    goal: str = ""
    verified: bool = False
    plan_only: bool = False
    note_spec: TaskNoteEvidence | None = None
    original_request: str = ""
    journal: str = ""
    acceptance: ActorAcceptance | None = None
    expected_hex: str = ""
    actual_hex: str | None = None
    sha256: str | None = None
    original_request_id: int | None = None
    accepted: bool = False
    initial_observation: JsonValue = None
    final_observation: JsonValue = None


class AdmissionObservation(BaseModel):
    source_id: int
    observed_at: datetime
    text: str


class BodyJournalEntry(BaseModel):
    event: str


class BodyJournalReturn(BaseModel):
    seq: int
    tool: str
    result: JsonValue


def recent_body_effects(root: Path) -> tuple[str, ...]:
    """Recover individual server returns even when the project stopped before writing its final report."""
    effects: deque[str] = deque(maxlen=8)
    count = 0
    for journal in sorted(root.glob("attempts/actor-*/scratch/actor-journal.jsonl")):
        if not journal.resolve().is_relative_to(root):
            raise ValueError(f"Actor journal escapes its task folder: {journal}")
        with journal.open(encoding="utf-8") as stream:
            for line in stream:
                if BodyJournalEntry.model_validate_json(line).event != "result":
                    continue
                entry = BodyJournalReturn.model_validate_json(line)
                if entry.tool not in {"minecraft_mine_block", "minecraft_drop_item", "minecraft_build"}:
                    continue
                try:
                    facts = MinecraftToolResult.model_validate(entry.result).facts()
                except ValidationError:
                    facts = "Unsupported server response shape. Effect unverified."
                count += 1
                effects.append(f"Server return {journal.parent.parent.name} #{entry.seq} {entry.tool}: {facts}")
    if count > len(effects):
        effects_summary = f"Showing the latest {len(effects)} of {count} mining, dropping or building returns. Earlier returns remain in the actor journals."
        return (effects_summary, *effects)
    return tuple(effects)


from workflows.minecraft_body.world_text import StopObservation


class TaskReceipt(BaseModel):
    stop_source_id: int | None = None
    stop_observation: StopObservation | None = None
    awaiting_stop_observation: bool = False
    model_config = ConfigDict(extra="ignore")
    artifact_path: str = ""
    workflow_name: str
    state: str
    run_id: str = ""
    latest_event: str = ""
    blocked_by_run_id: str = ""
    instructions: str = ""
    summary: str = ""
    answer: str = ""
    task_verified: bool = False
    admission_only: bool = Field(default=False, description="Acknowledge that the worker accepted the request.")
    admission_observation: AdmissionObservation | None = None
    evidence: tuple[str, ...] = ()
    report: TaskReportEvidence | ToolCallReport | None = None
    file_checks: tuple[FileCheck, ...] = ()
    actual_hex: str | None = None
    verification_error: str = ""
    minecraft_verification: TaskVerificationEvidence | None = None


    def prompt_excerpt(self, characters: int) -> TaskReceipt:
        """Keep the worker and verifier verdicts when detailed evidence has to shrink."""
        evidence = [evidence_excerpt(render_task_receipt(self), characters)]
        verification = self.minecraft_verification
        if verification is not None:
            evidence.insert(0, f"Independent world verification: {verification.verified}. "
                            + evidence_excerpt(verification.reason, characters))
        observed = self.admission_observation
        return TaskReceipt(workflow_name=self.workflow_name, state=self.state, run_id=self.run_id,
            latest_event=evidence_excerpt(self.latest_event, characters), blocked_by_run_id=self.blocked_by_run_id,
            instructions=evidence_excerpt(self.instructions, characters), task_verified=self.task_verified,
            admission_only=self.admission_only, evidence=tuple(evidence), stop_observation=self.stop_observation,
            stop_source_id=self.stop_source_id, awaiting_stop_observation=self.awaiting_stop_observation,
            summary=evidence_excerpt(self.summary, characters),
            admission_observation=observed.model_copy(update={"text": evidence_excerpt(observed.text, characters)})
                if observed else None)


    @field_validator("report", mode="before")
    @classmethod
    def read_report(cls, value: object) -> TaskReportEvidence | ToolCallReport | None:
        if value is None or isinstance(value, (TaskReportEvidence, ToolCallReport)):
            return value
        header = TaskReportHeader.model_validate(value)
        if header.schema_name == "pm.tool-call-report.v1":
            return ToolCallReport.model_validate(value)
        return TaskReportEvidence.model_validate(value)



def read_optional_task_receipt(value: object) -> TaskReceipt | None:
    if value is None or (isinstance(value, dict) and not value):
        return None
    if isinstance(value, TaskReceipt):
        return value
    return TaskReceipt.model_validate(value)


def render_task_receipt(receipt: TaskReceipt) -> str:
    identity = f" (run {receipt.run_id})" if receipt.run_id else ""
    if receipt.blocked_by_run_id:
        return (f"Task {receipt.workflow_name}{identity} did not start. Resource owner: {receipt.blocked_by_run_id}. "
                f"Admission detail: {receipt.latest_event}. Requested work: {receipt.instructions}")
    if receipt.admission_only:
        text = (f"Task {receipt.workflow_name}{identity}: worker state {receipt.state}. "
                "This receipt confirms admission only. Original instructions: " + receipt.instructions)
        observation = receipt.admission_observation
        if observation:
            text += (f"\nEarlier body observation [{observation.source_id}, {observation.observed_at}]. "
                     "Coordinates, distance and compass bearings are relative to your own Minecraft body. "
                     "Speak from that viewpoint.\n" + observation.text)
        return text
    verified = "verified" if receipt.task_verified else "not verified"
    lines = [f"Task {receipt.workflow_name}{identity}: worker state {receipt.state}; requested effect {verified}."]
    if receipt.instructions:
        lines.append("Original instructions: " + receipt.instructions)
    if receipt.summary:
        lines.append("Receipt summary: " + receipt.summary)
    if receipt.answer:
        lines.append("Worker answer: " + receipt.answer)
    for evidence in receipt.evidence:
        lines.append("Retained evidence: " + evidence)
    report = receipt.report
    if isinstance(report, ToolCallReport):
        lines.append(f"Worker-assessed tool goal outcome: {report.outcome}; {report.summary}")
        lines.extend(f"Server return [{call.call_id}] {call.tool}: {call.returned_facts}"
                     for call in report.tool_calls if call.returned_facts)
        lines.append("Use returned facts for quantities and locations. Attribute worker interpretations and "
                     "describe unsupported measurements as unknown.")
        for artifact in report.artifacts:
            lines.append(f"Reported artifact: {artifact.claim.path or artifact.claim.uri}; bytes verified={artifact.file_verified}; {artifact.reason}")
        if report.agent_outcome is not None:
            if report.agent_outcome.quality_assessment:
                lines.append("Worker quality assessment: " + report.agent_outcome.quality_assessment)
            lines.extend("Worker uncertainty: " + uncertainty for uncertainty in report.agent_outcome.uncertainty)
        for check in receipt.file_checks:
            lines.append(f"Current artifact {check.path}: exists={check.exists}; unchanged={check.unchanged}")
    elif report is not None:
        if report.original_request and report.original_request != receipt.instructions:
            lines.append("Original request: " + report.original_request)
        if report.goal:
            lines.append("Goal: " + report.goal)
        lines.append(f"Worker report verified={report.verified}; plan only={report.plan_only}.")
        if report.summary:
            lines.append("Worker report: " + report.summary)
        if report.note_spec is not None:
            lines.append("Requested note path: " + report.note_spec.path)
            label = "Verified note content" if receipt.task_verified else "Requested note content; not verified"
            lines.append(label + ":\n" + report.note_spec.text)
    verification = receipt.minecraft_verification
    if verification is not None:
        lines.append(f"Independent world verification: {verification.verified}.")
        if verification.reason:
            lines.append("Verification reason: " + verification.reason)
        if verification.scope:
            lines.append("Verification scope: " + verification.scope)
        for check_id, passed in verification.checks.items():
            lines.append(f"Acceptance {check_id}: {passed}.")
        if verification.accepted_nodes:
            lines.append("Accepted task nodes: " + ", ".join(verification.accepted_nodes))
        if verification.trees is not None:
            lines.append(f"Distinct harvested trees: {verification.trees}.")
        if verification.initial_logs is not None and verification.final_logs is not None:
            lines.append(f"Retained logs: {verification.initial_logs} before; {verification.final_logs} after.")
        if verification.maximum_travel is not None:
            lines.append(f"Maximum observed travel: {verification.maximum_travel:.3f} blocks.")
        if verification.return_distance is not None:
            lines.append(f"Return distance: {verification.return_distance:.3f} blocks.")
    return "\n".join(lines)
