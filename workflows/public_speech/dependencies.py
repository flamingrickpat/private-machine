"""Freeze evidence and partner frontiers shared by selection and speech.

The publication overview is :mod:`workflows.public_speech.public_dialogue`.
The memory store owns source meaning and atomic source guards. This library
adds the observed partner frontier and governing main tick without owning
intentions or treating every later event as grounds for cancellation.
"""
from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from pm.memory.dependencies import SourceDependency, capture_source, source_fingerprint
from pm.memory.snapshot import GhostConflict, Snapshot
from pm.memory.store import MemoryStore
from pm.records.base import RecordScope
from pm.records.enums import FeatureType, InterlocusType, SourceKind
from pm.records.memory_records import Entity, Feature


class PartnerFrontierChanged(GhostConflict):
    """A partner added input after this exchange context was frozen."""


class WorkDependencies(BaseModel):
    """Evidence observed by one local choice, including its partner frontier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sources: list[SourceDependency]
    input_commit_ids: list[str]
    guidance_tick: int = Field(description="Accepted main tick used as guidance, not a global freshness veto.")
    critical_epoch: int = Field(ge=0, description="Critical assessment epoch at capture time.")
    partner_entity_id: int | None
    partner_name: str | None = None
    scopes: list[RecordScope] = Field(min_length=1)
    source_ids: list[int]
    latest_partner_input_id: int | None


def _partner_input(record: Feature, *, partner_entity_id: int | None,
                   partner_name: str | None, scopes: Sequence[RecordScope]) -> bool:
    """Identify actual attributed input in this partner's reality and frame."""
    if (record.feature_type != FeatureType.Dialogue or not record.causal
            or record.interlocus != InterlocusType.Public or record.scope not in scopes
            or record.provenance is None
            or record.provenance.source_kind not in {SourceKind.User, SourceKind.Agent}):
        return False
    if partner_entity_id is not None and record.entity_id:
        return record.entity_id == partner_entity_id
    return partner_name is not None and record.source == partner_name


def latest_partner_input(view: Snapshot, *, partner_entity_id: int | None,
                         partner_name: str | None, scopes: Sequence[RecordScope]) -> int | None:
    """Return the newest committed input identity for this exact partner."""
    latest: int | None = None
    for record in view.all_knoxels.values():
        if isinstance(record, Feature) and _partner_input(record, partner_entity_id=partner_entity_id,
                                                        partner_name=partner_name, scopes=scopes):
            if latest is None or record.id > latest:
                latest = record.id
    return latest


def capture_dependencies(view: Snapshot, source_ids: Sequence[int], *, guidance_tick: int,
                         critical_epoch: int, partner_entity_id: int | None,
                         scopes: Sequence[RecordScope]) -> WorkDependencies:
    """Keep supplied source meanings and the observed partner frontier.

    Historical evidence remains historical. Caller-selected current members
    must remain heads. This operation performs no inference or mutation.
    """
    selected_ids = list(dict.fromkeys(source_ids))
    selected_scopes = list(dict.fromkeys(scopes))
    partner_name: str | None = None
    if partner_entity_id is not None:
        identity = view.get(partner_entity_id)
        if not isinstance(identity, Entity):
            raise TypeError("A dependency partner must identify an Entity record.")
        partner_name = identity.content
    else:
        for record_id in sorted(selected_ids, reverse=True):
            record = view.get(record_id)
            if (isinstance(record, Feature) and record.feature_type == FeatureType.Dialogue
                    and record.provenance is not None
                    and record.provenance.source_kind in {SourceKind.User, SourceKind.Agent}):
                partner_name = record.source
                break
    return WorkDependencies(
        sources=[capture_source(view, key, current_head=key in view.all_knoxels) for key in selected_ids],
        input_commit_ids=sorted(view.commit_ids), guidance_tick=guidance_tick,
        critical_epoch=critical_epoch, partner_entity_id=partner_entity_id, partner_name=partner_name,
        scopes=selected_scopes, source_ids=selected_ids,
        latest_partner_input_id=latest_partner_input(view, partner_entity_id=partner_entity_id,
            partner_name=partner_name, scopes=selected_scopes))


def validate_dependencies(ghost: MemoryStore, dependencies: WorkDependencies, *,
                          current_partner_input: bool) -> None:
    """Reject changed source meaning and optionally a newer partner frontier.

    The caller holds the store lock through its subsequent commit. Critical
    epoch validation remains with cognitive control. An unrelated partner's
    new input does not invalidate this exchange.
    """
    current = ghost.snapshot(capture=False)
    for dependency in dependencies.sources:
        if dependency.current_head and dependency.record_id not in current.all_knoxels:
            raise GhostConflict(f"Source {dependency.record_id} acquired a successor.")
        record = current.get_knoxel_by_id(dependency.record_id)
        if record is None or source_fingerprint(record) != dependency.fingerprint:
            raise GhostConflict(f"Source {dependency.record_id} changed meaning.")
    if current_partner_input:
        newest = latest_partner_input(current, partner_entity_id=dependencies.partner_entity_id,
            partner_name=dependencies.partner_name, scopes=dependencies.scopes)
        if newest != dependencies.latest_partner_input_id:
            raise PartnerFrontierChanged("New partner input requires a refreshed interaction context.")
