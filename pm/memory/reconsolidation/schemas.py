"""Typed configuration, prompts, model answers, and reports of memory reconsolidation.

The answer models validate model output. A validator raises ValueError for
a wrong answer, and :class:`~pm.memory.reconsolidation.model_calls.ModelCalls`
sends the error back in a repair turn.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from pm.constants import (
    RECONSOLIDATION_CHUNK_EVENTS, RECONSOLIDATION_CONTEXT_WINDOW, RECONSOLIDATION_MIN_BLOCK,
    RECONSOLIDATION_MIN_EVENTS, RECONSOLIDATION_NARRATIVE_FIRST_TOKENS, RECONSOLIDATION_NARRATIVE_MAX_TOKENS,
    RECONSOLIDATION_NARRATIVE_MIN_TOKENS, RECONSOLIDATION_OPEN_TOPIC_MINUTES, RECONSOLIDATION_SUMMARY_TOKEN_LIMIT,
    RECONSOLIDATION_TRIGGER_EVENTS,
)
from pm.records.enums import FeatureType, NarrativeTypes
from pm.records.taxonomy import EpistemicStatus, PmCategory, RecordType
from pm.utils import yaml_util

PROMPTS_FILE = Path(__file__).with_name("prompts.yaml")


def naive_utc_now() -> datetime:
    """Return the current UTC time as a naive datetime.

    Report timestamps are stored naive UTC on purpose: the time-range logic of
    ``pm.memory.reconsolidation.temporal`` compares them with the naive world
    timestamps of the records.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ReconsolidationConfig(BaseModel):
    """The tunable values of one run. The defaults come from ``pm.constants``.

    ``cluster_feature_types`` are the events that topical clustering groups
    (old ``clustered_feature_types``). ``story_feature_types`` are the
    features that a summary reads. The old code read every causal feature;
    the live store also holds long appraisal and decision records
    (``StateUpdate``), so the port reads only the story kinds.
    """

    model_config = ConfigDict(extra="forbid")

    trigger_events: int = Field(default=RECONSOLIDATION_TRIGGER_EVENTS, ge=1)
    min_events: int = Field(default=RECONSOLIDATION_MIN_EVENTS, ge=1)
    context_window: int = Field(default=RECONSOLIDATION_CONTEXT_WINDOW, ge=1)
    min_block: int = Field(default=RECONSOLIDATION_MIN_BLOCK, ge=1)
    chunk_events: int = Field(default=RECONSOLIDATION_CHUNK_EVENTS, ge=1)
    open_topic_minutes: int = Field(default=RECONSOLIDATION_OPEN_TOPIC_MINUTES, ge=0)
    summary_token_limit: int = Field(default=RECONSOLIDATION_SUMMARY_TOKEN_LIMIT, ge=1)
    narrative_min_tokens: int = Field(default=RECONSOLIDATION_NARRATIVE_MIN_TOKENS, ge=1)
    narrative_max_tokens: int = Field(default=RECONSOLIDATION_NARRATIVE_MAX_TOKENS, ge=1)
    narrative_first_tokens: int = Field(default=RECONSOLIDATION_NARRATIVE_FIRST_TOKENS, ge=1)
    cluster_feature_types: list[FeatureType] = [FeatureType.Dialogue, FeatureType.WorldEvent]
    story_feature_types: list[FeatureType] = [
        FeatureType.Dialogue, FeatureType.WorldEvent, FeatureType.Thought, FeatureType.ExternalThought,
        FeatureType.InnerMonologue, FeatureType.SubjectiveExperience]


class NarrativeDefinition(BaseModel):
    """One kept narrative: its type, its target, its label, and its goal text."""

    type: NarrativeTypes = NarrativeTypes.SelfImage
    target: Literal["companion", "user"] = "companion"
    label: str = ""
    goal: str = ""


class ReconsolidationPrompts(BaseModel):
    """The texts of ``prompts.yaml``."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str = ""
    story_intro: str = ""
    topic_summary: str = ""
    temporal_raw: str = ""
    temporal_children: str = ""
    mental_state: str = ""
    facts_extract: str = ""
    facts_classify: str = ""
    narrative: str = ""
    narrative_card: str = ""
    no_previous_narrative: str = ""
    no_recorded_state: str = ""
    narratives: list[NarrativeDefinition] = Field(default_factory=list[NarrativeDefinition])
    evidence_types: dict[NarrativeTypes, list[FeatureType]] = Field(default_factory=dict[NarrativeTypes, list[FeatureType]])  # pm-lint: allow PM004 -- evidence types by narrative family


def load_prompts(path: Path = PROMPTS_FILE) -> ReconsolidationPrompts:
    """Read and validate the prompt file. An invalid file raises."""
    return ReconsolidationPrompts.model_validate(yaml_util.safe_load(path.read_text(encoding="utf-8")))


class FactExtractionValidation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    line_count: int


class FactClaimOption(BaseModel):
    """An entity-ledger interpretation citing the individual fact's source lines."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    claim_id: str
    about: str
    statement: str


class CanonicalFactOption(BaseModel):
    """A current semantic head with the same source authority in this reality."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    head_id: int
    about: str
    statement: str
    record_type: RecordType
    supported_through: datetime


class FactClassificationValidation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    fact_count: int
    allowed_subjects: tuple[str, ...]
    claim_options: tuple[tuple[FactClaimOption, ...], ...] = ()
    canonical_options: tuple[tuple[CanonicalFactOption, ...], ...] = ()


# ----------------------------------------------------------------- answers

class SummaryAnswer(BaseModel):
    """A topical or temporal summary."""

    summary: str = Field(min_length=20, max_length=8000)


class MentalStateAnswer(BaseModel):
    """The state description of one cluster."""

    description: str = Field(min_length=10, max_length=2000)


class ExtractedFact(BaseModel):
    """One extracted fact and the numbers of the scene lines that state it.

    The validation context holds ``lines``: the number of scene lines.
    """

    text: str = Field(min_length=11, max_length=600)
    lines: list[int] = Field(min_length=1)

    @field_validator("lines")
    @classmethod
    def lines_exist(cls, value: list[int], info: ValidationInfo) -> list[int]:
        context = info.context
        if not isinstance(context, FactExtractionValidation):
            raise TypeError("Fact extraction requires its named source-line validation context.")
        count = context.line_count
        wrong = [line for line in value if not 1 <= line <= count]
        if wrong:
            raise ValueError(f"Line numbers {wrong} do not exist. The scene has lines 1 to {count}.")
        return sorted(set(value))


class FactsAnswer(BaseModel):
    """The facts of one scene. An empty list is valid."""

    facts: list[ExtractedFact]


class FactFiling(BaseModel):
    """The classification of one fact. The validation context holds ``about``: the allowed subjects."""

    index: int = 0
    about: str
    entity_claim: str | None = None
    canonical_id: int | None = None
    record_type: RecordType = RecordType.Fact
    epistemic_status: EpistemicStatus = EpistemicStatus.Observed
    confidence: float = Field(default=0.5, ge=0, le=1)
    object_kind: str = ""
    object_value: str = ""
    topic_card: str = ""
    essence: str = Field(min_length=5)
    category: list[str] = Field(min_length=1, max_length=4)
    importance: float = Field(default=0.5, ge=0, le=1)
    time_dependent: float = Field(default=0.5, ge=0, le=1)
    reason: str = ""

    @field_validator("about")
    @classmethod
    def known_subject(cls, value: str, info: ValidationInfo) -> str:
        context = info.context
        if not isinstance(context, FactClassificationValidation):
            raise TypeError("Fact classification requires its named subject validation context.")
        allowed = context.allowed_subjects
        match = [name for name in allowed if name.casefold() == value.strip().casefold()]
        if not match:
            raise ValueError(f"about must be one of {allowed}, not {value!r}.")
        return match[0]

    @field_validator("record_type")
    @classmethod
    def classified(cls, value: RecordType) -> RecordType:
        if value == RecordType.Unclassified:
            raise ValueError("record_type must be a real kind, not unclassified.")
        return value

    @field_validator("category")
    @classmethod
    def palette(cls, value: list[str]) -> list[str]:
        wrong = [item for item in value if not PmCategory.is_valid(item)]
        if wrong:
            raise ValueError(f"Unknown categories {wrong}. Use only {list(PmCategory.all())}.")
        return value


class FilingAnswer(BaseModel):
    """The classifications of all facts of one scene. The context holds ``count``."""

    facts: list[FactFiling] = Field(default_factory=list[FactFiling])

    @model_validator(mode="after")
    def one_per_fact(self, info: ValidationInfo) -> "FilingAnswer":
        context = info.context
        if not isinstance(context, FactClassificationValidation):
            raise TypeError("Fact classification requires its named count validation context.")
        expected = list(range(1, context.fact_count + 1))
        got = sorted(item.index for item in self.facts)
        if got != expected:
            raise ValueError(f"Give exactly one entry for each fact index {expected}; got {got}.")
        indexed_facts = {item.index: item for item in self.facts}
        self.facts = [indexed_facts[index] for index in expected]
        for item in self.facts:
            if item.canonical_id is not None:
                candidates = context.canonical_options[item.index - 1] if context.canonical_options else ()
                if (item.entity_claim is not None or not any(candidate.head_id == item.canonical_id and
                        candidate.about == item.about for candidate in candidates)):
                    raise ValueError(f"Fact {item.index} must use a listed canonical ID for its subject alone, or null.")
            if item.entity_claim is None:
                continue
            options = context.claim_options[item.index - 1] if context.claim_options else ()
            if not any(choice.claim_id == item.entity_claim and choice.about == item.about for choice in options):
                raise ValueError(f"Fact {item.index} must select a listed claim for its subject or use null.")
        return self


class NarrativeAnswer(BaseModel):
    """One complete narrative version."""

    narrative: str = Field(min_length=50, max_length=6000)


# ----------------------------------------------------------------- report

class ReconsolidationReport(BaseModel):
    """What one run did. The store keeps it as a consolidation receipt.

    The id lists name the records that the run created or replaced.
    ``model_calls`` counts every model request, repairs included.
    ``notes`` explains skipped stages.
    """

    kind: Literal["memory_reconsolidation"] = "memory_reconsolidation"
    run_id: str = ""
    forced: bool = False
    started_at: datetime = Field(default_factory=naive_utc_now)  # pm-lint: allow PM003 -- Pydantic timestamps each report on creation
    finished_at: datetime | None = None
    unclustered_events: int = 0
    topical_ids: list[int] = Field(default_factory=list[int])
    temporal_ids: list[int] = Field(default_factory=list[int])
    fact_ids: list[int] = Field(default_factory=list[int])
    semantic_ids: list[int] = Field(default_factory=list[int])
    semantic_updated_ids: list[int] = Field(default_factory=list[int])
    narrative_ids: list[int] = Field(default_factory=list[int])
    model_calls: int = 0
    embedded_texts: int = 0
    notes: list[str] = Field(default_factory=list[str])

    @property
    def created_ids(self) -> list[int]:
        return [*self.topical_ids, *self.temporal_ids, *self.fact_ids, *self.semantic_ids, *self.narrative_ids]


__all__ = ["ExtractedFact", "FactFiling", "FactsAnswer", "FilingAnswer", "MentalStateAnswer", "NarrativeAnswer",
           "NarrativeDefinition", "PROMPTS_FILE", "ReconsolidationConfig", "ReconsolidationPrompts",
           "ReconsolidationReport", "SummaryAnswer", "load_prompts"]
