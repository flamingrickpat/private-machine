"""Hydrate semantic entity drafts from the input worker's original evidence.

See :mod:`workflows.input_triage.host` for atomic acceptance. The model supplies
interpretation and existing citation IDs. This adapter supplies all hashes,
proposal identities, source clocks, observer identity, scope, and parent guards.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import hashlib

from pm.llm.structured import ModelFormatError
from pm.memory.dependencies import capture_source
from pm.memory.snapshot import Snapshot
from workflows.entity_context.owner import EntityContextOwner, active_claims
from workflows.entity_context.projection import project_entity_context, render_entity_context
from workflows.entity_context.schemas import (
    EntityClaimDraft, EntityClaimEvidence, EntityContextClaim, EntityContextKey, EntityContextProposal,
    EntityExtractionDraft, EntityProducer, EntityProjectionQuery,
)
from workflows.input_triage.schemas import (
    EntityExtractionFamily, EntityExtractionInput, EntityExtractionSource,
)


def capture_entity_extraction(owner: EntityContextOwner, view: Snapshot, *, subject_ids: Sequence[int],
                              observer_entity_id: int, source_ids: Sequence[int], evidence_ids: Sequence[int],
                              as_of_utc: datetime, token_budget: int) -> EntityExtractionInput:
    """Freeze permitted subjects, accepted parents, and actual supplied source meanings."""
    sources = [EntityExtractionSource(dependency=capture_source(view, key, current_head=key in view.all_knoxels),
                                     content=view.get(key).content, occurred_at_utc=view.get(key).timestamp_world_begin,
                                     scope=view.get(key).scope) for key in dict.fromkeys(evidence_ids)]
    scopes = list(dict.fromkeys(view.get(key).scope for key in source_ids))
    families: list[EntityExtractionFamily] = []
    for subject_id in dict.fromkeys(subject_ids):
        for scope in scopes:
            key = EntityContextKey(subject_entity_id=subject_id, observer_entity_id=observer_entity_id, scope=scope)
            claim_keys = list(dict.fromkeys(claim.claim_key for claim in active_claims(view, key=key)))
            context = project_entity_context(view, query=EntityProjectionQuery(key=key, claim_keys=claim_keys, token_budget=token_budget),
                                             as_of_utc=as_of_utc)
            families.append(EntityExtractionFamily(key=key, parent_receipt_id=owner.current_head(view, key), context=context))
    return EntityExtractionInput(source_ids=list(source_ids), sources=sources, families=families,
                                 input_commit_ids=sorted(view.commit_ids))


def render_extraction_context(extraction: EntityExtractionInput) -> str:
    """Show compact accepted claims without exposing host-generated answer fields."""
    lines = ["Accepted entity context for this extraction:"]
    for family in extraction.families:
        lines.append(render_entity_context(family.context))
    lines.append("Only these new source IDs require maintenance: " + ", ".join(map(str, extraction.source_ids)))
    return "\n".join(lines)


def hydrate_entity_extraction(extraction: EntityExtractionInput, draft: EntityExtractionDraft, *,
                              producer: EntityProducer) -> list[EntityContextProposal]:
    """Validate model references, then derive immutable proposals from frozen evidence."""
    sources = {source.dependency.record_id: source for source in extraction.sources}
    grouped: dict[EntityContextKey, list[EntityClaimDraft]] = {}
    for claim in draft.claims:
        if any(source_id not in sources for source_id in claim.source_ids):
            raise ModelFormatError("An entity draft cites a source outside its frozen evidence.")
        if not set(claim.source_ids).intersection(extraction.source_ids):
            raise ModelFormatError("An immediate entity draft must cite at least one supplied new input.")
        claim_scopes = list(dict.fromkeys(sources[source_id].scope for source_id in claim.source_ids))
        if len(claim_scopes) != 1:
            raise ModelFormatError("An entity draft cannot merge evidence across reality or frame.")
        scope = claim_scopes[0]
        family = next((item for item in extraction.families
                       if item.key.subject_entity_id == claim.subject_entity_id and item.key.scope == scope), None)
        if family is None:
            raise ModelFormatError("An entity draft names a subject outside the supplied observer families.")
        targets = set(claim.correction_target_ids + claim.challenge_target_ids)
        supplied_claims = {item.claim.claim_id for item in family.context.claims}
        if not targets.issubset(supplied_claims):
            raise ModelFormatError("A correction must cite an exact existing claim ID supplied in this family.")
        if claim.exact_quote is not None and not any(claim.exact_quote in sources[key].content for key in claim.source_ids):
            raise ModelFormatError("An entity report quote must occur verbatim in its supplied source.")
        grouped.setdefault(family.key, []).append(claim)
    proposals: list[EntityContextProposal] = []
    for key, drafts in grouped.items():
        family = next(item for item in extraction.families if item.key == key)
        material = producer + key.model_dump_json() + str(family.parent_receipt_id)
        material += "".join(extraction.input_commit_ids) + "".join(item.model_dump_json() for item in drafts)
        for draft_claim in drafts:
            for source_id in draft_claim.source_ids:
                material += sources[source_id].dependency.fingerprint
        proposal_id = "entity-" + hashlib.sha256(material.encode("utf-8")).hexdigest()
        claims: list[EntityContextClaim] = []
        retract_ids: list[str] = []
        challenge_ids: list[str] = []
        correction_evidence: list[EntityClaimEvidence] = []
        for draft_claim in drafts:
            evidence = [EntityClaimEvidence(source=sources[source_id].dependency,
                exact_quote=draft_claim.exact_quote if draft_claim.exact_quote is not None
                and draft_claim.exact_quote in sources[source_id].content else None) for source_id in draft_claim.source_ids]
            observed = max(sources[source_id].occurred_at_utc for source_id in draft_claim.source_ids)
            claims.append(EntityContextClaim(layer=draft_claim.layer, claim_key=draft_claim.claim_key,
                statement=draft_claim.statement, epistemic_kind=draft_claim.epistemic_kind,
                perspective_entity_id=draft_claim.perspective_entity_id, confidence=draft_claim.confidence,
                evidence=evidence, observed_at_utc=observed,
                applicable_from_utc=draft_claim.applicable_from_utc or observed,
                review_after_utc=draft_claim.review_after_utc, applicable_until_utc=draft_claim.applicable_until_utc,
                interaction=draft_claim.interaction))
            retract_ids.extend(draft_claim.correction_target_ids)
            challenge_ids.extend(draft_claim.challenge_target_ids)
            if draft_claim.correction_target_ids or draft_claim.challenge_target_ids:
                correction_evidence.extend(evidence)
        proposals.append(EntityContextProposal(proposal_id=proposal_id, key=key,
            parent_receipt_id=family.parent_receipt_id, producer=producer, input_commit_ids=extraction.input_commit_ids,
            claims=claims, retract_claim_ids=list(dict.fromkeys(retract_ids)),
            challenge_claim_ids=list(dict.fromkeys(challenge_ids)), correction_evidence=correction_evidence))
    return proposals
