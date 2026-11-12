"""Typed entity evidence, producer drafts, receipts, and compact views.

See :mod:`workflows.entity_context.owner` for acceptance ownership. Model drafts
contain semantic proposals and known source identities. The host supplies
dependency fingerprints, proposal identities, claim identities, and clocks.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pm.memory.dependencies import SourceDependency
from pm.records.base import RecordScope
from pm.records.enums import SourceKind


EntityClaimLayer = Literal["background", "relationship", "pattern", "situation", "perspective", "open_question"]
EntityEpistemicKind = Literal["observation", "report", "inference", "hypothetical"]
EntityProducer = Literal["immediate", "main", "thought", "tom", "migration"]
EntityApplicability = Literal["current", "due_for_review", "expired", "future"]
ENTITY_CONTEXT_FAMILY = "entity_context_family"
ENTITY_CONTEXT_RECEIPT = "entity_context_receipt"


class EntityContextKey(BaseModel):
    """One subject as understood by one observer in one reality and frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    subject_entity_id: int = Field(gt=0, description="Identity anchor of the entity this model describes.")
    observer_entity_id: int = Field(gt=0, description="Identity anchor of the observer holding this interpretation.")
    scope: RecordScope = Field(description="Reality and frame in which these claims apply.")


class EntityClaimEvidence(BaseModel):
    """Original source meaning and an optional verbatim report quotation."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source: SourceDependency
    exact_quote: str | None = Field(default=None, min_length=1, description="Exact source substring when this evidence is a report.")


class InteractionClaim(BaseModel):
    """An evidence-qualified ability estimate, never permission to contact."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    dimension: Literal["receive", "respond"]
    ability: Literal["can", "cannot", "unknown"]


class EntityClaimDraft(BaseModel):
    """Model-proposed interpretation using only identities supplied in its context."""

    model_config = ConfigDict(extra="forbid")
    subject_entity_id: int = Field(gt=0, description="Known subject identity copied from the supplied entity families.")
    source_ids: list[int] = Field(min_length=1, description="Original evidence record IDs copied from the supplied context.")
    exact_quote: str | None = Field(default=None, min_length=1, description="Verbatim source substring; required for a report.")
    layer: EntityClaimLayer
    claim_key: str = Field(min_length=1, description="Stable proposition key within the selected subject, observer, and scope.")
    statement: str = Field(min_length=1, description="Short attributed assertion that keeps reports separate from inferred external state.")
    epistemic_kind: EntityEpistemicKind
    perspective_entity_id: int | None = Field(default=None, gt=0)
    confidence: float = Field(ge=0.0, le=1.0, description="Support for this epistemic assertion, rather than proof of a report's truth.")
    applicable_from_utc: datetime | None = Field(default=None, description="Explicit applicable start in UTC; ISO offsets normalize to UTC, absent uses source world time.")
    review_after_utc: datetime | None = Field(default=None, description="Evidence-supported UTC review time; never an automatic opposite state.")
    applicable_until_utc: datetime | None = Field(default=None, description="Explicit UTC interval end, only when the evidence supports it.")
    interaction: InteractionClaim | None = None
    correction_target_ids: list[str] = Field(default_factory=list[str], description="Exact existing claim IDs copied from supplied context for explicit correction.")
    challenge_target_ids: list[str] = Field(default_factory=list[str], description="Exact existing claim IDs whose interpretation this evidence challenges.")

    @field_validator("applicable_from_utc", "review_after_utc", "applicable_until_utc")
    @classmethod
    def normalize_utc_time(cls, value: datetime | None) -> datetime | None:
        """Convert explicit model offsets to the runtime's naive UTC convention.

        This is the foreign-response boundary. Accepted internal claims keep
        their strict naive UTC invariant instead of accepting mixed clocks.
        An offset-free timestamp already denotes UTC through these fields.
        """
        if value is None or value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    @model_validator(mode="after")
    def coherent_epistemic_role(self) -> Self:
        """Reject malformed generation while the model response remains retryable."""
        if self.epistemic_kind == "report" and (self.exact_quote is None or not self.exact_quote.strip()):
            raise ValueError("An entity report draft requires its exact source quotation.")
        if self.epistemic_kind == "hypothetical" and self.interaction is not None:
            raise ValueError("A hypothetical claim cannot supply a current receive or respond estimate.")
        return self


class EntityExtractionDraft(BaseModel):
    """At most four local claims from one immediate extraction call."""

    model_config = ConfigDict(extra="forbid")
    claims: list[EntityClaimDraft] = Field(default_factory=list[EntityClaimDraft], max_length=4)


class EntityContextClaim(BaseModel):
    """One attributed claim whose historical truth and current applicability differ."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    claim_id: str = Field(default="", description="Owner-assigned immutable claim identity; empty only before staging.")
    layer: EntityClaimLayer
    claim_key: str = Field(min_length=1, description="Exact proposition key within this observer and subject family.")
    statement: str = Field(min_length=1, description="The attributed assertion, without promoting inference to observation.")
    epistemic_kind: EntityEpistemicKind
    perspective_entity_id: int | None = Field(default=None, gt=0, description="Entity whose knowledge is attributed, distinct from the observer.")
    confidence: float = Field(ge=0.0, le=1.0, description="Support for this epistemic assertion, not proof of a report's truth.")
    evidence: list[EntityClaimEvidence] = Field(min_length=1)
    observed_at_utc: datetime = Field(description="Latest cited source world time, as naive UTC.")
    applicable_from_utc: datetime = Field(description="Beginning of the asserted applicable interval, as naive UTC.")
    review_after_utc: datetime | None = Field(default=None, description="Time to weaken current applicability, never infer the opposite.")
    applicable_until_utc: datetime | None = Field(default=None, description="Explicit end of the asserted interval, if evidence supplies one.")
    interaction: InteractionClaim | None = None

    @model_validator(mode="after")
    def valid_times(self) -> Self:
        """Reject mixed clock conventions and impossible explicit intervals."""
        if not self.claim_key.strip() or not self.statement.strip():
            raise ValueError("Entity claims require a nonempty proposition key and assertion.")
        for timestamp in (self.observed_at_utc, self.applicable_from_utc, self.review_after_utc, self.applicable_until_utc):
            if timestamp is not None and timestamp.tzinfo is not None:
                raise ValueError("Entity-context timestamps must use naive UTC world time.")
        if self.applicable_until_utc is not None and self.applicable_until_utc < self.applicable_from_utc:
            raise ValueError("An entity claim cannot end before its applicable interval starts.")
        return self


class EntityContextProposal(BaseModel):
    """Frozen host-hydrated interpretation with an explicit expected family parent."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    proposal_id: str = Field(min_length=1)
    key: EntityContextKey
    parent_receipt_id: int | None
    producer: EntityProducer
    input_commit_ids: list[str] = Field(default_factory=list[str])
    claims: list[EntityContextClaim] = Field(default_factory=list[EntityContextClaim], max_length=4)
    retract_claim_ids: list[str] = Field(default_factory=list[str])
    challenge_claim_ids: list[str] = Field(default_factory=list[str])
    correction_evidence: list[EntityClaimEvidence] = Field(default_factory=list[EntityClaimEvidence])


class EntityContextReceipt(BaseModel):
    """Validated payload of one append-only private state-update feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["pm.entity-context.v1"] = "pm.entity-context.v1"
    proposal: EntityContextProposal


class EntityUpdateResult(BaseModel):
    """Explicit staging outcome that lets valid main appraisal survive stale proposals."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    proposal_id: str
    disposition: Literal["staged", "deferred", "rejected"]
    reason: Literal["accepted", "stale_parent", "changed_source", "unsupported_interpretation"]
    current_parent_receipt_id: int | None
    staged_receipt_id: int | None

    @model_validator(mode="after")
    def coherent_outcome(self) -> Self:
        """A rejected or deferred update cannot carry an accepted receipt."""
        if self.disposition == "staged":
            if self.reason != "accepted" or self.staged_receipt_id is None:
                raise ValueError("A staged entity update requires its accepted receipt identity.")
        elif self.reason == "accepted" or self.staged_receipt_id is not None:
            raise ValueError("A deferred or rejected entity update cannot carry an accepted receipt.")
        return self


class EntitySourceAttribution(BaseModel):
    """The actual source speaker and addressee, independent of the claim's subject."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    record_id: int
    source_kind: SourceKind | None
    speaker_entity_id: int | None
    speaker: str
    addressee_entity_id: int | None


class EntityClaimView(BaseModel):
    """A retained claim with time-sensitive applicability and competing-claim state."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    claim: EntityContextClaim
    applicability: EntityApplicability
    contested: bool
    attributions: list[EntitySourceAttribution] = Field(default_factory=list[EntitySourceAttribution])


class EntityProjectionQuery(BaseModel):
    """Exact relevant proposition keys frozen before a consumer calculates."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    key: EntityContextKey
    claim_keys: list[str] = Field(default_factory=list[str], description="Exact relevant proposition keys; an empty list selects no non-interaction claims.")
    include_interaction: bool = Field(default=True, description="Retain receive/respond estimates, including explicit unknown when support is absent.")
    token_budget: int = Field(gt=0, description="Caller-supplied text token allowance for complete claims and minimum grounding.")


class EntityContextView(BaseModel):
    """Compact supplied context and its original acceptance dependencies."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    key: EntityContextKey
    query: EntityProjectionQuery
    identity_source: SourceDependency
    head_receipt_id: int | None = Field(description="Family head at capture for provenance, never a blanket read-side acceptance veto.")
    projected_meaning_fingerprint: str = Field(description="Captured SHA-256 of relevant supplied meaning, excluding incidental receipt bookkeeping.")
    as_of_utc: datetime = Field(description="World time of this frozen projection, as naive UTC.")
    next_applicability_change_utc: datetime | None = Field(description="Earliest future review or interval boundary of the supplied claims, as naive UTC.")
    claims: list[EntityClaimView]
    source_dependencies: list[SourceDependency]
    omitted_claim_count: int = Field(ge=0)

    @model_validator(mode="after")
    def coherent_frontier(self) -> Self:
        """Keep the frozen query, identity, and clock in one explicit family."""
        if self.key != self.query.key or self.identity_source.record_id != self.key.subject_entity_id:
            raise ValueError("An entity view must retain its query family and subject identity.")
        if self.as_of_utc.tzinfo is not None:
            raise ValueError("Entity projection time must use naive UTC world time.")
        if self.next_applicability_change_utc is not None:
            if self.next_applicability_change_utc.tzinfo is not None or self.next_applicability_change_utc <= self.as_of_utc:
                raise ValueError("A captured applicability boundary must be a future naive UTC world time.")
        return self
