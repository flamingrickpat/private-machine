"""Typed request, ontology, and report boundary for Story dialogue.

Model output and the fixed situation enter Pydantic once, at the edge. Invalid
input raises immediately; no empty default conceals a missing field. These
models are plain data and perform no I/O.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from workflows.public_context import PublicContext
from workflows.public_claims import ClaimReview


class DialogueTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker: str = Field(min_length=1)
    text: str = Field(min_length=1)


class StoryRequest(BaseModel):
    """The fixed situation plus the selected context for one Story generation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pm.story-request.v1"] = "pm.story-request.v1"
    companion_name: str = Field(min_length=1)
    user_name: str = Field(min_length=1)
    character_card_story: str
    supported_capabilities: list[str] = Field(default_factory=list)
    unsupported_capabilities: list[str] = Field(default_factory=list)
    capability_notes: list[str] = Field(default_factory=list)
    temporal_context: str
    prior_dialogue: list[DialogueTurn] = Field(default_factory=list)
    current_user_message: str = Field(min_length=1)
    feelings: list[str] = Field(default_factory=list)
    memories: list[str] = Field(default_factory=list)
    workspace: str = ""
    selected_action: str = ""
    public_context: PublicContext | None = None


class OntologyViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    reason: str
    match: str


class OntologyVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    accepted: bool
    violations: list[OntologyViolation] = Field(default_factory=list)
    facts_checked: int = 0
    fact_conflicts: list[str] = Field(default_factory=list)


class StoryReport(BaseModel):
    """Everything a reviewer needs to audit one generation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pm.story-report.v1"] = "pm.story-report.v1"
    companion_name: str
    message_roles: list[str]
    reply_prefix: str
    raw_completion: str
    candidate: str
    accepted: bool
    ontology: OntologyVerdict
    claim_review: ClaimReview | None = None


class StoryQualityScore(BaseModel):
    """One evaluator result over a saved candidate. Reads artifacts only."""

    model_config = ConfigDict(extra="forbid")

    scale: Literal[-1, 0, 1]
    immersive: Literal[-1, 0, 1]
    voice: Literal[-1, 0, 1]
    grounded: Literal[-1, 0, 1]
    reason: str = Field(min_length=1)


class StoryTrialResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial: int
    run_id: str
    accepted: bool
    candidate: str
    ontology: OntologyVerdict
    report_path: str


__all__ = [
    "DialogueTurn",
    "StoryRequest",
    "OntologyViolation",
    "OntologyVerdict",
    "StoryReport",
    "StoryQualityScore",
    "StoryTrialResult",
]
