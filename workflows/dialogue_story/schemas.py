"""Typed request, ontology, and report boundary for Story dialogue.

Model output and the fixed situation enter Pydantic once, at the edge. Invalid
input raises immediately; no empty default conceals a missing field. These
models are plain data and perform no I/O.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from workflows.public_speech.public_context import DialogueTurn, PublicContext
from workflows.public_speech.speech_review import SpeechReview


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
    # ISO UTC time of this generation. The prompt renders it and the gap
    # since the last prior turn in plain words.
    now: str = ""
    prior_dialogue: list[DialogueTurn] = Field(default_factory=list)
    current_user_message: str = Field(min_length=1)
    current_message_id: int | None = None
    # Accepted feelings in words, for example "strong fear and mild guilt".
    feelings: str = ""
    regulation: str = ""
    goals: list[str] = Field(default_factory=list)
    # Plain memory lines. Used only when no public context is supplied.
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


class StoryAttempt(BaseModel):
    """One generation attempt and its hallucination review.

    ``avoid`` is the note that this attempt received from the previous
    review. ``blocked`` marks a review that still holds a blocking kind.
    """

    model_config = ConfigDict(extra="forbid")

    raw_completion: str
    candidate: str
    review: SpeechReview | None = None
    avoid: str = ""
    blocked: bool = False


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
    attempts: list[StoryAttempt] = Field(default_factory=list)
    # The one agentic review and the reason it recorded.
    speech_review: SpeechReview | None = None
    review_reason: str = ""


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
    "StoryAttempt",
    "StoryRequest",
    "OntologyViolation",
    "OntologyVerdict",
    "StoryReport",
    "StoryQualityScore",
    "StoryTrialResult",
]
