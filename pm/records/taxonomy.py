"""Holds the semantic record taxonomy shared by memory reconsolidation, canonical records,
and the ghost filesystem dump.

archetype: static taxonomy definitions
owns: RecordType / EpistemicStatus / MemoryVolatility / RecordStatus / SemanticRecordRelationType
  enums, the pm category palette constants, the MemoryTaxonomy static helper class with
  normalization/repair rules, and MemoryTaxonomyConfig defaults for batching and dedup.
not own: LLM classification prompts, consolidation scheduling, persistence, or embedding logic.
fails when: callers pass values outside the enums and rely on the defaulting behavior
  (unknown types are repaired to ``fact`` rather than raising).
domain: every semantic memory classification in pm uses these enums so that reconsolidation,
  canonical records, and filesystem dumps agree on vocabulary.
invariant: every taxonomy string round-trips through its enum unchanged; normalization never
  invents a value that is not a member of the corresponding enum.
"""
from __future__ import annotations

from enum import StrEnum

from pm.records.enums import RecordSubtypeBase


class RecordType(RecordSubtypeBase):
    """Claim-kind classification for a single piece of semantic memory.

    This is the agentic processing taxonomy: it says what KIND of claim a fact is,
    which drives routing, volatility, and retrieval behavior. It is orthogonal to
    the pm category palette (topical retrieval buckets) and to importance/time_dependent
    (ranking dimensions).
    """
    Fact = "fact"
    Preference = "preference"
    Goal = "goal"
    Belief = "belief"
    Trait = "trait"
    Relationship = "relationship"
    Capability = "capability"
    Procedure = "procedure"
    Rule = "rule"
    Location = "location"
    Resource = "resource"
    Hazard = "hazard"
    EventOutcome = "event_outcome"
    Unclassified = "unclassified"


class EpistemicStatus(StrEnum):
    """How a claim was obtained, i.e. its trust dimension.

    Orthogonal to importance: a low-importance fact can be fully asserted, and a
    high-importance one can be a mere hypothesis.
    """
    Asserted = "asserted"
    Observed = "observed"
    Inferred = "inferred"
    Hypothesis = "hypothesis"
    ReportedByOther = "reported_by_other"


class MemoryVolatility(StrEnum):
    """Coarse expected persistence of a claim."""
    Stable = "stable"
    Ephemeral = "ephemeral"


class RecordStatus(StrEnum):
    """Lifecycle state of a canonical semantic record."""
    Active = "active"
    Superseded = "superseded"
    Retracted = "retracted"


class TruthStatus(StrEnum):
    """Epistemic truth lifecycle of mechanically generated memory.

    Maintained by memory consolidation (agent sleep): extracted knowledge starts
    as ``Truth`` and is only moved by evidence. The three states distinguish
    different lifecycles:

    Truth
        Currently valid knowledge. Default for everything consolidation produces.
    ProvenNeverTrue
        Falsified: the claim was always wrong (mis-observation, mis-extraction,
        or a hypothesis that testing refuted).
    Invalidated
        Was true, but the world changed or newer knowledge superseded it
        (the superseding knoxels are listed in ``superseded_by_ids``).
    """
    Truth = "truth"
    ProvenNeverTrue = "proven_never_true"
    Invalidated = "invalidated"


class SemanticRecordRelationType(StrEnum):
    """Typed links between canonical semantic records."""
    Supersedes = "supersedes"
    ContradictedBy = "contradicted_by"
    Supports = "supports"
    RelatedTo = "related_to"


class PmCategory:
    """The pm fact-category palette used by CategorizeFacts.

    Static constants (not an enum) on purpose: the palette is part of the agent
    contract and new categories may be added without code changes to consumers
    that only validate membership via ``all()`` / ``is_valid()``.
    """
    PEOPLE_PERSONALITY = "people_personality"
    PEOPLE_NEEDS = "people_needs"
    PEOPLE_GOALS = "people_goals"
    PEOPLE_PREFERENCES = "people_preferences"
    PEOPLE_INTERACTIONS = "people_interactions"
    WORLD_PLACES = "world_places"
    WORLD_ENVIRONMENT = "world_environment"
    WORLD_WORLD_BUILDING = "world_world_building"
    WORLD_EVENTS = "world_events"
    RELATIONSHIPS_GOOD = "relationships_good"
    RELATIONSHIPS_BAD = "relationships_bad"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return (
            cls.PEOPLE_PERSONALITY,
            cls.PEOPLE_NEEDS,
            cls.PEOPLE_GOALS,
            cls.PEOPLE_PREFERENCES,
            cls.PEOPLE_INTERACTIONS,
            cls.WORLD_PLACES,
            cls.WORLD_ENVIRONMENT,
            cls.WORLD_WORLD_BUILDING,
            cls.WORLD_EVENTS,
            cls.RELATIONSHIPS_GOOD,
            cls.RELATIONSHIPS_BAD,
        )

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls.all()


class MemoryTaxonomy:
    """Static normalization/repair helpers for the semantic taxonomy.

    Consolidation and migration both route raw LLM output through these so the
    stored enums are always valid members and drift is repaired identically.
    """

    # LLM drift repair table, ported from the migration scripts.
    _TYPE_REPAIR = {
        "asserted": RecordType.Fact,
        "observed": RecordType.Fact,
        "inferred": RecordType.Fact,
        "hypothesis": RecordType.Fact,
        "reported_by_other": RecordType.Fact,
        "assertion": RecordType.Fact,
        "reference": RecordType.Fact,
        "behavior": RecordType.Trait,
        "activity": RecordType.Trait,
        "habit": RecordType.Trait,
        "action": RecordType.Trait,
        "routine": RecordType.Trait,
        "emotion": RecordType.Belief,
        "mood": RecordType.Belief,
        "state": RecordType.Fact,
        "question": RecordType.Fact,
        "instruction": RecordType.Rule,
        "proposal": RecordType.Goal,
        "plan": RecordType.Goal,
        "embedding": RecordType.Fact,
    }

    _EPISTEMIC_ALIASES = {
        "assertion": EpistemicStatus.Asserted,
        "assert": EpistemicStatus.Asserted,
        "observation": EpistemicStatus.Observed,
        "inference": EpistemicStatus.Inferred,
        "report": EpistemicStatus.ReportedByOther,
    }

    @staticmethod
    def normalize_text(text: str | None) -> str:
        """Lowercase alphanumeric normalization used for exact-match dedup keys."""
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in str(text or "").lower())
        return " ".join(cleaned.split())

    @staticmethod
    def normalize_record_type(value: str | None) -> RecordType:
        """Coerce a raw classifier type string into a valid RecordType member."""
        raw = str(value or "").strip().lower()
        if not raw:
            return RecordType.Unclassified
        try:
            return RecordType(raw)
        except ValueError:
            return MemoryTaxonomy._TYPE_REPAIR.get(raw, RecordType.Fact)

    @staticmethod
    def normalize_epistemic(value: str | None) -> EpistemicStatus:
        """Coerce a raw classifier epistemic string into a valid EpistemicStatus member."""
        raw = str(value or "").strip().lower()
        if not raw:
            return EpistemicStatus.Observed
        try:
            return EpistemicStatus(raw)
        except ValueError:
            return MemoryTaxonomy._EPISTEMIC_ALIASES.get(raw, EpistemicStatus.Observed)

    @staticmethod
    def dedup_key(subject_entity_id: int, essence: str) -> str:
        """Identity key for claim dedup: same subject + same canonical essence = same claim."""
        return f"{int(subject_entity_id)}:{MemoryTaxonomy.normalize_text(essence)}"
