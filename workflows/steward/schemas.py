"""Typed request and report for the steward workflow.

The steward performs no generation. Its report carries the evidence it read.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StewardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pm.steward-request.v1"] = "pm.steward-request.v1"
    operation: str
    run_id: str = ""
    question_id: str = ""
    answer: str = ""
    instructions: str = ""
    request_id: str = ""


class StewardReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pm.steward-report.v1"] = "pm.steward-report.v1"
    operation: str
    accepted: bool
    summary: str
    result: dict = Field(default_factory=dict)


__all__ = ["StewardReport", "StewardRequest"]
