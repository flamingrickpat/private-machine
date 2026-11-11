"""Frozen classification inputs and durable receipts for early triage.

The host overview explains acceptance and publication ordering.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from workflows.main_cognition.cycle_work import CycleRequest
from workflows.main_cognition.schemas import TriageResult


class InputSourcePayload(BaseModel):
    """The durable source identity carried by a user-input envelope."""

    knoxel_id: int = Field(strict=True, gt=0)


class TriageEvidence(BaseModel):
    """Exact source meanings and accepted commits supplied to classification."""

    model_config = ConfigDict(extra="forbid")
    record_fingerprints: dict[int, str]
    commit_ids: list[str]
    current_head_ids: list[int]


class InputTriageRequest(BaseModel):
    """A lightweight cycle view without retrieval, embeddings, or appraisal."""

    model_config = ConfigDict(extra="forbid")
    cycle: CycleRequest
    source_fingerprints: dict[int, str]
    guidance_tick: int | None
    evidence: TriageEvidence


class EarlyTriageAssessment(BaseModel):
    """One accepted classification and the evidence that its model observed.

    ``reconsideration_epoch`` advances only for a critical assessment. Main
    cognition acknowledges its sources in a later accepted cycle.
    """

    model_config = ConfigDict(extra="forbid")
    source_ids: list[int] = Field(min_length=1)
    source_fingerprints: dict[int, str]
    guidance_tick: int | None
    result: TriageResult
    evidence: TriageEvidence
    reconsideration_epoch: int = Field(default=0, ge=0)
