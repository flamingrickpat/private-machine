"""Project relevant entity meaning without turning receipt appends into vetoes.

See :mod:`workflows.entity_context.owner` for the authoritative receipt stream.
Projection is deterministic and performs no inference or memory mutation.
It preserves reports, uncertainty, conflicts, and source attribution within a
caller-supplied token budget. Time weakens applicability without inventing an
opposite state. Current direction validates this projection; historical main
appraisal does not acquire a latest-family requirement merely by reading it.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json

import tiktoken
from pydantic import BaseModel, ConfigDict

from pm.context_render.budget import RequiredContextTooLarge
from pm.memory.dependencies import SourceDependency, capture_source, source_fingerprint
from pm.memory.snapshot import Snapshot
from pm.memory.store import MemoryStore
from pm.memory.write_batch import WriteBatch
from pm.records.base import RecordBase
from pm.records.enums import SourceKind
from pm.records.memory_records import Entity, Feature
from workflows.entity_context.owner import model_state
from workflows.entity_context.schemas import (
    EntityApplicability, EntityClaimLayer, EntityClaimView, EntityContextClaim,
    EntityContextView, EntityEpistemicKind, EntityProjectionQuery,
    EntitySourceAttribution, InteractionClaim,
)


class EntityContextChanged(RuntimeError):
    """Relevant current meaning changed; refresh this direction before acceptance."""


class _AttributionMeaning(BaseModel):
    """Source identity semantics without incidental support-record identity."""

    model_config = ConfigDict(frozen=True)
    source_kind: SourceKind | None
    speaker_entity_id: int | None
    speaker: str
    addressee_entity_id: int | None


class _ClaimMeaning(BaseModel):
    """Fields whose change can alter the supplied assertion or its authority."""

    model_config = ConfigDict(frozen=True)
    claim_key: str
    layer: EntityClaimLayer
    statement: str
    epistemic_kind: EntityEpistemicKind
    perspective_entity_id: int | None
    confidence: float
    applicability: EntityApplicability
    applicable_from_utc: datetime | None
    applicable_until_utc: datetime | None
    interaction: InteractionClaim | None
    attributions: list[_AttributionMeaning]


def _digest(value: BaseModel) -> str:
    """Hash a named meaning contract at the serialization boundary."""
    serialized = json.dumps(value.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _applicability(claim: EntityContextClaim, as_of_utc: datetime) -> EntityApplicability:
    """Classify the stated interval without inferring an unobserved replacement."""
    if as_of_utc < claim.applicable_from_utc:
        return "future"
    if claim.applicable_until_utc is not None and as_of_utc >= claim.applicable_until_utc:
        return "expired"
    if claim.review_after_utc is not None and as_of_utc >= claim.review_after_utc:
        return "due_for_review"
    return "current"


def _attribution(view: Snapshot, record: RecordBase) -> EntitySourceAttribution:
    """Read known record shapes without treating the subject as the source speaker."""
    source_kind = record.provenance.source_kind if record.provenance is not None else None
    if isinstance(record, Feature):
        speaker = record.override_source or "unknown source"
        if record.entity_id:
            identity = view.get(record.entity_id)
            if not isinstance(identity, Entity):
                raise ValueError("An entity source speaker must refer to an Entity identity.")
            speaker = record.override_source or identity.content
        return EntitySourceAttribution(record_id=record.id, source_kind=source_kind,
            speaker_entity_id=record.entity_id or None, speaker=speaker,
            addressee_entity_id=record.addressee_entity_id or None)
    return EntitySourceAttribution(record_id=record.id, source_kind=source_kind,
        speaker_entity_id=None, speaker=source_kind.value if source_kind is not None else "unknown source",
        addressee_entity_id=None)


def _meaning(item: EntityClaimView) -> _ClaimMeaning:
    """Exclude repeat-support IDs and derived review clocks, while keeping attribution."""
    attributions: dict[str, _AttributionMeaning] = {}
    for source in item.attributions:
        attribution = _AttributionMeaning(source_kind=source.source_kind, speaker_entity_id=source.speaker_entity_id,
            speaker=source.speaker, addressee_entity_id=source.addressee_entity_id)
        attributions[_digest(attribution)] = attribution
    claim = item.claim
    return _ClaimMeaning(claim_key=claim.claim_key, layer=claim.layer,
        statement=" ".join(claim.statement.split()), epistemic_kind=claim.epistemic_kind,
        perspective_entity_id=claim.perspective_entity_id, confidence=claim.confidence,
        applicability=item.applicability,
        applicable_from_utc=claim.applicable_from_utc if item.applicability == "future" else None,
        applicable_until_utc=claim.applicable_until_utc, interaction=claim.interaction,
        attributions=[attributions[key] for key in sorted(attributions)])


def _claim_line(item: EntityClaimView) -> str:
    """Keep a complete assertion, its epistemic role, and its actual evidence sources."""
    claim = item.claim
    attribution = "; ".join(f"{source.speaker} ({source.source_kind.value if source.source_kind else 'unknown origin'}), "
                            f"record {source.record_id}" for source in item.attributions)
    perspective = f"; attributed perspective of entity {claim.perspective_entity_id}" if claim.perspective_entity_id is not None else ""
    conflict = "; contested" if item.contested else ""
    interaction = f"; {claim.interaction.dimension} estimate: {claim.interaction.ability}" if claim.interaction else ""
    return (f"[claim {claim.claim_id}; key {claim.claim_key}; {claim.epistemic_kind}; confidence {claim.confidence:.2f}; "
            f"{item.applicability}{conflict}{perspective}{interaction}; observed {claim.observed_at_utc.isoformat()} UTC; "
            f"source {attribution}] {claim.statement}")


def _tokens(text: str) -> int:
    """Use the existing local tokenizer dependency for a conservative text budget."""
    return len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))


def _header(query: EntityProjectionQuery) -> str:
    """Supply minimum grounding and distinguish estimates from external facts."""
    key = query.key
    return (f"Entity {key.subject_entity_id} as understood by observer {key.observer_entity_id}; "
            f"reality {key.scope.reality_id}/{key.scope.frame.value}. "
            "Reports and inferred ability remain attributed estimates. Missing, expired, and contested state remains uncertain.")


def _interaction_lines(claims: list[EntityClaimView], *, include_interaction: bool) -> list[str]:
    """Missing or stale support cannot establish either ability or inability."""
    if not include_interaction:
        return []
    lines: list[str] = []
    for dimension in ("receive", "respond"):
        abilities: set[str] = set()
        for item in claims:
            interaction = item.claim.interaction
            if (interaction is not None and interaction.dimension == dimension
                    and item.applicability == "current" and not item.contested
                    and item.claim.epistemic_kind != "hypothetical"):
                abilities.add(interaction.ability)
        estimate = next(iter(abilities)) if len(abilities) == 1 else "unknown"
        lines.append(f"Current ability to {dimension}: {estimate} (estimate, not contact permission).")
    return lines


def project_entity_context(view: Snapshot, *, query: EntityProjectionQuery,
                           as_of_utc: datetime) -> EntityContextView:
    """Freeze compact relevant meaning and original sources without a model call.

    Exact claim keys and interaction dimensions form the read frontier.
    Unrelated family appends do not become read-side cancellation guards.
    Count limits omit whole claims and report that omission.
    """
    if as_of_utc.tzinfo is not None:
        raise ValueError("Entity projection time must use naive UTC world time.")
    identity = view.get(query.key.subject_entity_id)
    observer = view.get(query.key.observer_entity_id)
    if not isinstance(identity, Entity) or not isinstance(observer, Entity):
        raise ValueError("Entity projection requires subject and observer identity anchors.")
    state = model_state(view, query.key)
    equivalent: dict[str, EntityClaimView] = {}
    for claim in state.claims.values():
        if claim.claim_key not in query.claim_keys and not (query.include_interaction and claim.interaction is not None):
            continue
        item = EntityClaimView(claim=claim, applicability=_applicability(claim, as_of_utc),
            contested=claim.claim_id in state.challenged_claim_ids,
            attributions=[_attribution(view, view.get(source.source.record_id)) for source in claim.evidence])
        meaning_id = _digest(_meaning(item))
        previous = equivalent.get(meaning_id)
        # Retain the earliest equivalent current support. A repeat report's
        # new IDs and timestamp cannot change admission at a tight budget.
        if previous is None:
            equivalent[meaning_id] = item
        elif item.contested and not previous.contested:
            equivalent[meaning_id] = previous.model_copy(update={"contested": True})
    relevant = list(equivalent.values())
    # Different assertions of the same proposition remain visibly contested.
    for index, item in enumerate(relevant):
        alternatives = [other for other in relevant
                        if other.claim.claim_key == item.claim.claim_key and _digest(_meaning(other)) != _digest(_meaning(item))
                        and other.applicability == "current" and item.applicability == "current"]
        if alternatives:
            relevant[index] = item.model_copy(update={"contested": True})
    relevant.sort(key=lambda item: (item.claim.interaction is None, item.claim.layer not in {"relationship", "background"},
                                    item.claim.claim_key, _digest(_meaning(item))))
    minimum = _header(query) + "\n" + "\n".join(_interaction_lines([], include_interaction=query.include_interaction))
    if _tokens(minimum) > query.token_budget:
        raise RequiredContextTooLarge("The entity context budget cannot hold required identity and uncertainty grounding.")
    selected: list[EntityClaimView] = []
    omission_notice = f"{len(relevant)} further relevant claims omitted by the context budget."
    used_tokens = _tokens(minimum) + (_tokens(omission_notice) + 1 if relevant else 0)
    for item in relevant:
        cost = _tokens(_claim_line(item)) + 1
        if used_tokens + cost <= query.token_budget:
            selected.append(item)
            used_tokens += cost
    deadlines: list[datetime] = []
    dependencies: dict[int, SourceDependency] = {}
    for item in selected:
        claim = item.claim
        for deadline in (claim.applicable_from_utc, claim.review_after_utc, claim.applicable_until_utc):
            if deadline is not None and deadline > as_of_utc:
                deadlines.append(deadline)
        for evidence in claim.evidence:
            dependencies[evidence.source.record_id] = evidence.source
        for attribution in item.attributions:
            if attribution.speaker_entity_id is not None:
                speaker_id = attribution.speaker_entity_id
                dependencies.setdefault(speaker_id, capture_source(view, speaker_id, current_head=True))
    dependencies.setdefault(observer.id, capture_source(view, observer.id, current_head=True))
    # Meaning hashes deliberately exclude record IDs, clocks of repeated
    # observations, family-head IDs, and the count of unrelated receipts.
    meaning_hashes = sorted(_digest(_meaning(item)) + (":contested" if item.contested else "") for item in selected)
    meaning_text = json.dumps([query.key.model_dump(mode="json"), meaning_hashes,
                              _interaction_lines(selected, include_interaction=query.include_interaction)],
                             sort_keys=True, ensure_ascii=False)
    selected_keys = sorted({item.claim.claim_key for item in selected if item.claim.claim_key in query.claim_keys})
    frozen_query = query.model_copy(update={"claim_keys": selected_keys})
    return EntityContextView(key=query.key, query=frozen_query,
        identity_source=capture_source(view, identity.id, current_head=True),
        head_receipt_id=state.receipts[-1].record.id if state.receipts else None,
        projected_meaning_fingerprint=hashlib.sha256(meaning_text.encode()).hexdigest(),
        as_of_utc=as_of_utc, next_applicability_change_utc=min(deadlines) if deadlines else None,
        claims=selected, source_dependencies=list(dependencies.values()),
        omitted_claim_count=len(relevant) - len(selected))


def render_entity_context(context: EntityContextView) -> str:
    """Render the captured view with attribution, uncertainty, and original claim IDs."""
    lines = [_header(context.query)]
    lines.extend(_interaction_lines(context.claims, include_interaction=context.query.include_interaction))
    lines.extend(_claim_line(item) for item in context.claims)
    if context.omitted_claim_count:
        lines.append(f"{context.omitted_claim_count} further relevant claims omitted by the context budget.")
    return "\n".join(lines)


def validate_entity_context(store: MemoryStore, captured: EntityContextView, *, as_of_utc: datetime) -> None:
    """Reject changed relevant direction while retaining the original evidence.

    The first slice conservatively rejects a crossed captured freshness
    boundary, even if unseen equivalent evidence could extend that estimate.
    The host keeps the store lock through validation and its acceptance commit.
    """
    with store.lock:
        if captured.next_applicability_change_utc is not None and as_of_utc >= captured.next_applicability_change_utc:
            raise EntityContextChanged("The captured entity estimate reached its review or applicability boundary.")
        current = store.snapshot(capture=False)
        for source in [captured.identity_source, *captured.source_dependencies]:
            record = current.get_knoxel_by_id(source.record_id)
            if (record is None or source.current_head and source.record_id not in current.all_knoxels
                    or source_fingerprint(record) != source.fingerprint):
                raise EntityContextChanged("An original entity-context source changed or lost required current authority.")
        projection = project_entity_context(current, query=captured.query, as_of_utc=as_of_utc)
        if projection.projected_meaning_fingerprint != captured.projected_meaning_fingerprint:
            raise EntityContextChanged("The relevant projected entity meaning changed during calculation.")


def expect_entity_context(batch: WriteBatch, captured: EntityContextView, *, as_of_utc: datetime) -> None:
    """Compare current relevant meaning and keep original source commit guards."""
    validate_entity_context(batch.store, captured, as_of_utc=as_of_utc)
    batch.expect_source(captured.identity_source)
    for source in captured.source_dependencies:
        batch.expect_source(source)
