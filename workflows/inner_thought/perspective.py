"""The theory-of-mind specialization of a shared private inquiry.

See :mod:`workflows.inner_thought.host` for the canonical overview of the
private loop and the inquiry graph. This module gives one deliberate
perspective inquiry (the ``theory_of_mind`` mental action) what it may read
about its subject: the companion's accepted entity model of that subject and
the original messages behind it. It renders that material for the thought
prompt and states the epistemic rules that keep inference, report, attributed
knowledge, and unknown separate.

The subject is the entity that the action's partner named. The material is
captured without a model call from the accepted entity-context family of the
same observer, reality, and frame, so a perspective step never reads another
character's model or another conversation. Thinking proposes entries; only
main cognition writes them (decision D2 in the B06 plan).
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from pydantic import Field

from pm.constants import WORKSPACE_TAIL_TOKENS
from pm.memory.snapshot import Snapshot
from pm.records.memory_records import Entity
from workflows.entity_context.owner import active_claims
from workflows.entity_context.projection import project_entity_context, render_entity_context
from workflows.entity_context.schemas import EntityContextKey, EntityContextView, EntityProjectionQuery
from workflows.entity_context.schemas import EntityExtractionDraft
from workflows.inner_thought.inquiry import outside_source
from workflows.inner_thought.pipeline import IntegrationError
from workflows.inner_thought.schemas import InquiryContract, InquiryGrant


class PerspectiveContext(InquiryContract):
    """What one perspective-inquiry step reads about its subject.

    ``evidence_source_ids`` are the original messages behind the accepted
    entries, filtered to the current view, the accepted commits, and this
    inquiry's reality and frame. They become citable candidates and extend
    the step grant; the subject and observer identity anchors do not.
    """

    subject_entity_id: int = Field(gt=0, description="Identity anchor of the entity this inquiry models.")
    subject_name: str = Field(min_length=1, description="Canonical name of the subject entity.")
    subject_class: str = Field(min_length=1, description="EntityClass value, so a non-human subject is visible.")
    entity_view: EntityContextView = Field(description="Accepted view of the subject, captured for this step.")
    evidence_source_ids: tuple[int, ...] = Field(
        description="Original sources behind that view: current, accepted, same scope, not identity anchors.")


def capture_perspective_context(view: Snapshot, *, subject_entity_id: int, grant: InquiryGrant,
                                accepted_ids: frozenset[int], as_of_utc: datetime) -> PerspectiveContext:
    """Freeze the accepted entity view of one subject for one perspective step.

    The claim keys come from the active family, so the whole accepted model
    of the subject is readable, and the projection budget still bounds it.
    Only the observer's own reality and frame is projected: the grant already
    carries that scope, and a subject outside it cannot appear.
    """
    if as_of_utc.tzinfo is not None:
        raise ValueError("Perspective context requires naive UTC world time.")
    subject = view.get(subject_entity_id)
    if not isinstance(subject, Entity):
        raise ValueError("A perspective inquiry requires an entity subject.")
    key = EntityContextKey(subject_entity_id=subject_entity_id,
                           observer_entity_id=grant.observer_entity_id, scope=grant.scope)
    claim_keys = list(dict.fromkeys(claim.claim_key for claim in active_claims(view, key=key)))
    entity_view = project_entity_context(view,
        query=EntityProjectionQuery(key=key, claim_keys=claim_keys, token_budget=WORKSPACE_TAIL_TOKENS),
        as_of_utc=as_of_utc)
    evidence_source_ids: list[int] = []
    for dependency in entity_view.source_dependencies:
        record_id = dependency.record_id
        if record_id not in view.all_knoxels or record_id not in accepted_ids:
            continue
        record = view.get(record_id)
        if record.scope != grant.scope or isinstance(record, Entity):
            # The identity anchors are header material, not evidence. A source
            # in another reality or frame stays out of this inquiry's grant.
            continue
        evidence_source_ids.append(record_id)
    return PerspectiveContext(subject_entity_id=subject.id, subject_name=subject.content,
        subject_class=subject.entity_class.value, entity_view=entity_view,
        evidence_source_ids=tuple(evidence_source_ids))


def render_perspective_context(context: PerspectiveContext) -> str:
    """Write the subject line and the accepted entity view for one perspective step."""
    lines = [f"Subject {context.subject_name} (entity {context.subject_entity_id}, "
             f"class {context.subject_class})."]
    if not context.entity_view.claims:
        lines.append("No accepted entry exists for this subject.")
    else:
        lines.append(render_entity_context(context.entity_view))
    return "\n".join(lines)


def perspective_rule(evidence: str, *, companion_name: str, subject_name: str) -> str:
    """State how a perspective step reads and uses its subject material.

    The evidence is the rendered accepted view of the subject. The rules keep
    the four epistemic roles apart, forbid contact, and stop repeated thinking
    from masquerading as support. Main cognition decides anything that asks
    the subject directly.
    """
    return (
        f"This inquiry models {subject_name} as {companion_name} understands them.\n"
        "The lines below are accepted entries. A report is what was said. An inference is an estimate. "
        "Entries marked contested, expired, or due for review are uncertain.\n"
        "Keep four things apart: what the subject said or did (cite the record), what the companion infers, "
        "what the subject may know, believe, or want (attributed and fallible), and what is unknown.\n"
        "The companion's knowledge is not the subject's knowledge.\n"
        "Consider one alternative explanation and one exception when the evidence allows it.\n"
        "Do not give a non-human entity human sleep, work, motives, or feelings without evidence.\n"
        "A routine is an estimate. It is never an observation of today.\n"
        "Repeating an earlier thought adds no support.\n"
        "This inquiry cannot contact the subject. If asking is the best way to learn something, state it as "
        "an information gap. Main cognition decides whether to ask.\n"
        "Perspective material:\n" + evidence)


class ExistingClaimRef(InquiryContract):
    """One accepted claim the integrate stage may use for correction or challenge."""

    claim_id: str = Field(min_length=1, description="Owner-assigned immutable identity of the accepted claim.")
    claim_key: str = Field(min_length=1, description="Exact proposition key within the subject family.")
    statement: str = Field(min_length=1, description="The accepted attributed assertion.")
    source_ids: tuple[int, ...] = Field(description="Original sources the accepted claim already cites.")


class PerspectiveRevisionRequest(InquiryContract):
    """Ask the integrate stage for optional entity-model entries at the end of a perspective inquiry."""

    subject_entity_id: int = Field(gt=0, description="Identity anchor of the entity this inquiry modelled.")
    subject_name: str = Field(min_length=1, description="Canonical name of the subject entity.")
    citable_source_ids: tuple[int, ...] = Field(
        description="Original outside sources only; never a private thought or an identity anchor.")
    existing_claims: tuple[ExistingClaimRef, ...] = Field(
        description="Accepted claims of the subject that a revision may cite, correct, or challenge.")


def revision_request(view: Snapshot, context: PerspectiveContext,
                     grant: InquiryGrant) -> PerspectiveRevisionRequest:
    """Build the bounded revision question from the captured context and grant.

    ``citable_source_ids`` are only the grant sources that are independently
    supplied outside material (decision D5): a private thought that the step
    itself produced is not evidence about the subject. ``existing_claims`` are
    the accepted claims shown to the model, with their original sources.
    """
    citable_source_ids = tuple(source.record_id for source in grant.sources
                               if outside_source(view, source.record_id, grant=grant))
    existing_claims = tuple(
        ExistingClaimRef(claim_id=item.claim.claim_id, claim_key=item.claim.claim_key,
            statement=item.claim.statement,
            source_ids=tuple(evidence.source.record_id for evidence in item.claim.evidence))
        for item in context.entity_view.claims)
    return PerspectiveRevisionRequest(subject_entity_id=context.subject_entity_id,
        subject_name=context.subject_name, citable_source_ids=citable_source_ids,
        existing_claims=existing_claims)


def _normalized(text: str) -> str:
    """Collapse whitespace so rewritten wording cannot dodge the repeat rule."""
    return " ".join(text.split())


def validate_perspective_revision(draft: EntityExtractionDraft, request: PerspectiveRevisionRequest,
                                  source_texts: Mapping[int, str]) -> None:
    """Reject a stored perspective draft that breaks the entity model or its evidence rules.

    Each message tells the model what to do instead, in one sentence. A stale
    draft is a model-quality result against a changed world, not an internal
    bug: the owner records the rejection and the episode continues.
    """
    citable = frozenset(request.citable_source_ids)
    existing_by_id = {claim.claim_id: claim for claim in request.existing_claims}
    for claim in draft.claims:
        if claim.subject_entity_id != request.subject_entity_id:
            raise IntegrationError(
                f"A perspective claim names another subject; name entity {request.subject_entity_id}.")
        if claim.epistemic_kind == "observation":
            raise IntegrationError("Thinking is not observing; use inference, hypothetical, or report.")
        if claim.interaction is not None or claim.layer == "situation":
            raise IntegrationError(
                "A perspective revision cannot change the receive/respond estimate or the current-situation layer.")
        outside = [source_id for source_id in claim.source_ids if source_id not in citable]
        if outside:
            raise IntegrationError(
                f"A perspective claim cites a source this inquiry cannot read: {sorted(outside)}.")
        if claim.epistemic_kind == "report":
            quote = claim.exact_quote or ""
            if not any(quote in source_texts.get(source_id, "") for source_id in claim.source_ids):
                raise IntegrationError(
                    "A report needs its exact quote inside one of the message texts it cites.")
        targets = set(claim.correction_target_ids + claim.challenge_target_ids)
        if not targets.issubset(existing_by_id):
            raise IntegrationError(
                "Correct or challenge only the existing claim IDs supplied; write an estimate for anything new.")
        if claim.perspective_entity_id is not None and claim.perspective_entity_id != request.subject_entity_id:
            raise IntegrationError(
                "Attributed knowledge belongs to the subject of this inquiry, not to another entity.")
        for existing in request.existing_claims:
            if (existing.claim_key == claim.claim_key
                    and _normalized(existing.statement) == _normalized(claim.statement)):
                if not set(claim.source_ids).difference(existing.source_ids):
                    raise IntegrationError(
                        "Repeating an accepted claim with no new source adds no support; "
                        "cite a new source or drop the claim.")


__all__ = [
    "ExistingClaimRef", "PerspectiveContext", "PerspectiveRevisionRequest", "capture_perspective_context",
    "perspective_rule", "render_perspective_context", "revision_request", "validate_perspective_revision",
]
