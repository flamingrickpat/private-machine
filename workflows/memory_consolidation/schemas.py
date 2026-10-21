"""Typed request and report for memory consolidation."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ConsolidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pm.consolidation-request.v1"] = "pm.consolidation-request.v1"
    operation: str = "consolidate"
    run_id: str = ""
    artifact_path: str = ""
    source_event_ids: list[int] = Field(default_factory=list)


class ConsolidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pm.consolidation-report.v1"] = "pm.consolidation-report.v1"
    operation: str
    accepted: bool
    summary: str
    receipt: dict = Field(default_factory=dict)
    receipts: list[dict] = Field(default_factory=list)


__all__ = ["ConsolidationReport", "ConsolidationRequest"]
