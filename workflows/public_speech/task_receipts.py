from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, JsonValue
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


class TaskReceipt(BaseModel):
    model_config = ConfigDict(extra="ignore")
    artifact_path: str = ""
    workflow_name: str
    state: str
    run_id: str = ""
    instructions: str = ""
    summary: str = ""
    answer: str = ""
    task_verified: bool = False
    evidence: tuple[str, ...] = ()
    report: TaskReportEvidence | ToolCallReport | None = None
    file_checks: tuple[FileCheck, ...] = ()
    actual_hex: str | None = None
    verification_error: str = ""
    minecraft_verification: TaskVerificationEvidence | None = None


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
