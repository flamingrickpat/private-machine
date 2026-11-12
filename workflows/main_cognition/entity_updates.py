"""Delegate entity reports without creating a competing current belief store.

The main host stages these proposals at its final locked acceptance boundary.
Generic and self-model belief updates retain the existing belief owner.
See :mod:`workflows.main_cognition.host` for the subsystem overview.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from pm.constants import WORKSPACE_TAIL_TOKENS
from pm.memory.snapshot import Snapshot
from pm.records.enums import FeatureType, SourceKind
from pm.records.memory_records import Entity, Feature
from workflows.entity_context.owner import EntityContextOwner
from workflows.entity_context.schemas import EntityClaimDraft, EntityContextProposal, EntityExtractionDraft
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
