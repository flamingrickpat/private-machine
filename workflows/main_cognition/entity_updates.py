"""Delegate entity reports without creating a competing current belief store.

The main host stages these proposals at its final locked acceptance boundary.
Generic and self-model belief updates retain the existing belief owner.
See :mod:`workflows.main_cognition.host` for the subsystem overview.
"""
from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from pm.constants import WORKSPACE_TAIL_TOKENS
from pm.llm.structured import ModelFormatError
from pm.memory.snapshot import Snapshot
from pm.records.enums import FeatureType, SourceKind
from pm.records.memory_records import Entity, Feature
from workflows.entity_context.owner import EntityContextOwner
from workflows.entity_context.schemas import (
    EntityClaimDraft, EntityContextProposal, EntityExtractionDraft, PerspectiveRevision,
)
from workflows.input_triage.entity_updates import capture_entity_extraction, hydrate_entity_extraction
from workflows.main_cognition.experience import RejectedBeliefUpdate
from workflows.main_cognition.schemas import ActionRequest, BeliefUpdate


class EntityActionReconsideration(BaseModel):
    """An actual selected action withheld because its current context changed."""

    action_record_id: int
    request: ActionRequest
    reason: str
    correlation_id: str


class EntityActionMetadata(BaseModel):
    """Validated storage boundary for a durable reconsideration cause."""

    entity_action_reconsideration: EntityActionReconsideration


class TerminalReconsideration(BaseModel):
    """Exact terminal authority or an explicit refusal to combine different task replies."""

    request: ActionRequest | None = None
    only_terminal_cause: bool = False
    rejection_reason: str | None = None


class MainEntityUpdates(BaseModel):
    """Prepared proposals and explicit raw-report refusals for one main result."""

    proposals: list[EntityContextProposal] = Field(default_factory=list[EntityContextProposal])
    generic_updates: list[BeliefUpdate] = Field(default_factory=list[BeliefUpdate])
    rejected: list[RejectedBeliefUpdate] = Field(default_factory=list[RejectedBeliefUpdate])


def prepare_perspective_revisions(owner: EntityContextOwner, view: Snapshot, stimulus_ids: Sequence[int], *,
                                  observer_entity_id: int) -> MainEntityUpdates:
    """Stage the entity entries that private perspective inquiries proposed.

    A perspective finding (a ``theory_of_mind`` outcome) keeps its proposed
    revision on the operation record. Main cognition accepts it here, exactly
    like its own belief updates: the private loop proposes, this boundary
    writes, with ``producer="tom"``. The revision cites the original messages
    it read, so they become the proposal's evidence and source guards.

    A stored draft that no longer fits the model is a recorded rejection, not
    an error: the entity family can change between thinking and appraisal
    (risk R2 in the B06 plan), and hydrating a stale draft against a changed
    world is a model-quality result. Nothing else is caught here. The cited
    messages are retained guards of the finding, so a cited message that the
    view does not hold is a broken invariant and raises.
    """
    result = MainEntityUpdates()
    for finding_id in stimulus_ids:
        record = view.get_knoxel_by_id(finding_id)
        if record is None or not isinstance(record, Feature) or "perspective_revision" not in record.metadata:
            continue
        revision = PerspectiveRevisionMetadata.model_validate(record.metadata).perspective_revision
        # The stored contract requires at least one claim, and each claim at
        # least one source, so the cited list is never empty.
        cited_ids: list[int] = []
        for claim in revision.claims:
            for source_id in claim.source_ids:
                if source_id not in cited_ids:
                    cited_ids.append(source_id)
        cited_times = [view.get(source_id).timestamp_world_begin for source_id in cited_ids]
        as_of_utc = max(cited_times)
        extraction = capture_entity_extraction(owner, view, subject_ids=[revision.subject_entity_id],
            observer_entity_id=observer_entity_id, source_ids=cited_ids, evidence_ids=cited_ids,
            as_of_utc=as_of_utc, token_budget=WORKSPACE_TAIL_TOKENS)
        try:
            proposals = hydrate_entity_extraction(extraction,
                EntityExtractionDraft(claims=list(revision.claims)), producer="tom")
        except ModelFormatError as error:
            # The stored draft is checked against the world as it is now. A
            # changed family, a retracted target, or a removed source is a
            # recorded rejection; the finding keeps its evidence for review.
            result.rejected.append({"source_id": finding_id, "reason": str(error)})
            continue
        result.proposals.extend(proposals)
    return result


class PerspectiveRevisionMetadata(BaseModel):
    """Read the perspective revision a private finding kept on its operation record."""

    model_config = ConfigDict(extra="ignore")
    perspective_revision: PerspectiveRevision


def prepare_entity_beliefs(owner: EntityContextOwner, view: Snapshot, updates: list[BeliefUpdate], *,
                           observer_entity_id: int, permitted_source_ids: set[int]) -> MainEntityUpdates:
    """Keep exact entity reports as proposals over their originally observed parents."""
    result = MainEntityUpdates()
    drafts: list[EntityClaimDraft] = []
    subject_ids: list[int] = []
    source_ids: list[int] = []
    for update in updates:
        if update.source_id not in permitted_source_ids:
            result.rejected.append({"source_id": update.source_id, "reason": "source is outside the current stimulus grant"})
            continue
        source = view.get_knoxel_by_id(update.source_id)
        if (not isinstance(source, Feature) or source.feature_type != FeatureType.Dialogue
                or source.provenance is None or source.provenance.source_kind not in {SourceKind.User, SourceKind.Agent}
                or update.quote not in source.content):
            result.rejected.append({"source_id": update.source_id, "reason": "entity reports require exact attributed dialogue"})
            continue
        subject_id = update.subject_entity_id or source.entity_id
        if not subject_id or subject_id == observer_entity_id:
            result.generic_updates.append(update)
            continue
        if not isinstance(view.get_knoxel_by_id(subject_id), Entity):
            result.rejected.append({"source_id": update.source_id, "reason": "the proposed subject has no supplied identity anchor"})
            continue
        subject_ids.append(subject_id)
        source_ids.append(source.id)
        drafts.append(EntityClaimDraft(subject_entity_id=subject_id, source_ids=[source.id], exact_quote=update.quote,
            layer=update.entity_layer, claim_key=update.belief_key, statement=update.quote,
            epistemic_kind="report", confidence=1.0,
            correction_target_ids=update.correction_target_ids if update.correction else []))
    if drafts:
        extraction = capture_entity_extraction(owner, view, subject_ids=subject_ids,
            observer_entity_id=observer_entity_id, source_ids=source_ids, evidence_ids=source_ids,
            as_of_utc=max(view.get(key).timestamp_world_begin for key in source_ids), token_budget=WORKSPACE_TAIL_TOKENS)
        result.proposals = hydrate_entity_extraction(extraction, EntityExtractionDraft(claims=drafts), producer="main")
    return result
