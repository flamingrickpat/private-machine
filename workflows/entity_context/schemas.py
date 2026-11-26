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


from pm.memory.entity_history import (
    ENTITY_CONTEXT_FAMILY as ENTITY_CONTEXT_FAMILY,
    ENTITY_CONTEXT_RECEIPT as ENTITY_CONTEXT_RECEIPT,
    EntityApplicability as EntityApplicability,
    EntityClaimLayer as EntityClaimLayer,
    EntityEpistemicKind as EntityEpistemicKind,
    EntityProducer as EntityProducer,
    EntityContextKey as EntityContextKey,
    EntityClaimEvidence as EntityClaimEvidence,
    InteractionClaim as InteractionClaim,
    EntityContextClaim as EntityContextClaim,
    EntityContextProposal as EntityContextProposal,
    EntityContextReceipt as EntityContextReceipt,
    EntitySourceAttribution as EntitySourceAttribution,
    EntityClaimView as EntityClaimView,
)


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


class PerspectiveRevision(BaseModel):
    """Entity-model entries that a perspective inquiry proposed. Main cognition stages them."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    subject_entity_id: int = Field(gt=0, description="Identity anchor of the entity the inquiry modelled.")
    claims: tuple[EntityClaimDraft, ...] = Field(min_length=1, max_length=4,
        description="Model drafts. The host supplies fingerprints, claim IDs, and the family parent later.")


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
