"""Typed request and report for the SelfManagement workflow.

Self-management is deterministic. No model call and no prompt are involved.
Model output never enters this boundary.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

class SelfManagementRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pm.self-management-request.v1"] = "pm.self-management-request.v1"
    operation: str
    action_id: str = ""
    run_id: str = ""
    workflow_name: str = ""
    instructions: str = ""
    parent_correlation: str = ""
    request_id: str = ""
    question_id: str = ""
    answer: str = ""
    arguments: dict = Field(default_factory=dict)


class SelfManagementReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pm.self-management-report.v1"] = "pm.self-management-report.v1"
    operation: str
    accepted: bool
    summary: str
    receipt: dict = Field(default_factory=dict)
    receipts: list[dict] = Field(default_factory=list)


__all__ = ["SelfManagementRequest", "SelfManagementReport"]
