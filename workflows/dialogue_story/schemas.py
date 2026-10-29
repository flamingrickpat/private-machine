"""Typed request, ontology, and report boundary for Story dialogue.

Model output and the fixed situation enter Pydantic once, at the edge. Invalid
input raises immediately; no empty default conceals a missing field. These
models are plain data and perform no I/O.
Runtime ownership is described in [the Story host overview][workflows.dialogue_story.host].
"""
from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field, with_config
from pm.context_render.budget import SelectionReceipt
from workflows.public_speech.public_context import DialogueTurn, MemoryLaneTemplate, PublicContext, TimeTemplates
from workflows.public_speech.story_rendering import DialogueTemplates
from workflows.public_speech.speech_review import SpeechReview


@with_config(ConfigDict(extra="forbid"))
class StoryGenerationParameters(TypedDict, total=False):
    """Supported provider parameters in the Story prompt configuration."""

    temperature: float
    top_p: float
    enable_thinking: bool


@with_config(ConfigDict(extra="forbid"))
class StoryPrompts(TimeTemplates, DialogueTemplates):
    """The existing YAML prompt configuration, validated when it is loaded."""

    system_prompt: str
    context_start: str
    character_title: str
    architecture_title: str
    architecture_text: str
    abilities_title: str
    ontology_title: str
    time_title: str
    shards_title: str
    shards_note: str
    memory_lanes: list[MemoryLaneTemplate]
    memories_note: str
    goals_title: str
    inner_title: str
    inner_feelings: str
    inner_regulation: dict[str, str]
    inner_mind: str
    inner_intent: str
    presentation: dict[str, str]
    tasks_title: str
    context_end: str
    continue_directive: str
    story_instruction: str
    regeneration_title: str
    reply_prefix: str
    empty_completion_recovery: str
    rejected_reply_recovery: str
    generation: NotRequired[StoryGenerationParameters]


class StoryRequest(BaseModel):
    """The fixed situation plus the selected context for one Story generation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pm.story-request.v1"] = "pm.story-request.v1"
    companion_name: str = Field(min_length=1)
    partner_name: str = Field(min_length=1)
    character_card_story: str
    supported_capabilities: list[str] = Field(default_factory=list)
    unsupported_capabilities: list[str] = Field(default_factory=list)
    capability_notes: list[str] = Field(default_factory=list)
    # ISO UTC time of this generation. The prompt renders it and the gap
    # since the last prior turn in plain words.
    now: str = ""
    prior_dialogue: list[DialogueTurn] = Field(default_factory=list[DialogueTurn])
    current_user_message: str
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
    input_budget_tokens: int | None = Field(default=None, gt=0, description="Conservative ceiling for the complete rendered prompt, when supplied by the runtime.")


class OntologyViolation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    reason: str
    match: str


class OntologyVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    accepted: bool
    violations: list[OntologyViolation] = Field(default_factory=list[OntologyViolation])
    facts_checked: int = 0
    fact_conflicts: list[str] = Field(default_factory=list)


class StoryAttempt(BaseModel):
    """One completion layout, its output error, and its optional speech review."""

    model_config = ConfigDict(extra="forbid")

    raw_completion: str
    candidate: str
    review: SpeechReview | None = None
    avoid: str = ""
    blocked: bool = False
    message_roles: list[str] = Field(default_factory=list)
    generation_error: str = ""
    context_selection: SelectionReceipt | None = Field(default=None, description="Actual optional memory selection for this completion, including repair turns.")
    review_context_selection: SelectionReceipt | None = Field(default=None, description="Actual optional memory selection for the review layout and complete generated draft.")


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
    attempts: list[StoryAttempt] = Field(default_factory=list[StoryAttempt])
    # The accepted draft's review and its reason.
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
