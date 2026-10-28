"""Classify typed source evidence without guessing an internal record's shape."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from pm.memory.references import referenced_ids
from pm.memory.snapshot import Snapshot
from pm.records.base import Provenance, RecordBase, RecordScope
from pm.records.enums import FeatureType, FrameEnum, SourceKind
from pm.records.memory_records import Entity, Feature


class EvidenceMetadata(BaseModel):
    """The declared classification fields in heterogeneous stored metadata."""

    model_config = ConfigDict(extra="ignore")
    memory_kind: str = ""
    affect_summary: JsonValue = None


class EvidenceData(BaseModel):
    """Fields needed to classify a stored record envelope."""

    model_config = ConfigDict(extra="ignore")
    id: int
    scope: RecordScope = Field(default_factory=RecordScope)
    metadata: EvidenceMetadata = Field(default_factory=EvidenceMetadata)
    feature_type: FeatureType | Literal["Feeling"] | None = Field(default=None, description="Current feature type or the legacy Feeling label.")
    provenance: Provenance | None = None
    entity_id: int | None = None


class EvidenceEnvelope(BaseModel):
    """The named wire boundary for existing serialized record callers."""

    model_config = ConfigDict(extra="ignore")
    record_kind: str = Field(alias="__type__")
    data: EvidenceData


def project(raw: RecordBase | Mapping[str, object]) -> EvidenceEnvelope:
    """Read known record fields directly. Validate serialized callers once."""
    if isinstance(raw, RecordBase):
        data = EvidenceData(id=raw.id, scope=raw.scope, metadata=EvidenceMetadata.model_validate(raw.metadata),
            feature_type=raw.feature_type if isinstance(raw, Feature) else None, provenance=raw.provenance,
            entity_id=raw.entity_id if isinstance(raw, Feature) else None)
        return EvidenceEnvelope(__type__=raw.type.value, data=data)
    return EvidenceEnvelope.model_validate(raw)


def memory_kind(raw: RecordBase | Mapping[str, object]) -> str:
    """Classify attribution and occurrence without treating generated prose as fact."""
    envelope = project(raw)
    data = envelope.data
    if data.metadata.memory_kind:
        return data.metadata.memory_kind
    if data.feature_type == FeatureType.StateUpdate and data.metadata.affect_summary:
        return "internal_state"
    if envelope.record_kind in {"Narrative", "CauseEffectKnoxel"}:
        return "interpretation"
    if data.feature_type == FeatureType.Dialogue:
        return "utterance"
    if data.feature_type in {FeatureType.Thought, FeatureType.InnerMonologue}:
        return "speculation"
    if data.feature_type in {"Feeling", FeatureType.SubjectiveExperience}:
        return "interpretation"
    if envelope.record_kind in {"WorldPlaceKnoxel", "WorldObjectKnoxel", "BodyEpisodeKnoxel"}:
        return "observation"
    return "attributed_record"


def evidence_role(view: Snapshot, raw: RecordBase | Mapping[str, object]) -> str:
    """Distinguish reports, observed events, generated speech, and derived records."""
    envelope = project(raw)
    data = envelope.data
    if data.scope.frame != FrameEnum.InWorld:
        return "imagined"
    if envelope.record_kind == "ActionSchemaKnoxel":
        return "intent"
    kind = memory_kind(raw)
    if kind == "executed_event":
        return "runtime_event"
    if kind == "internal_state":
        return "internal_state"
    if data.feature_type == FeatureType.Dialogue:
        entity = view.get_knoxel_by_id(data.entity_id) if data.entity_id is not None else None
        human = isinstance(entity, Entity) and str(entity.entity_class) == "Human"
        if data.provenance is not None and data.provenance.source_kind == SourceKind.User or human:
            return "user_report"
        return "companion_utterance"
    if envelope.record_kind in {"WorldPlaceKnoxel", "WorldObjectKnoxel", "BodyEpisodeKnoxel"}:
        return "world_observation"
    if kind in {"speculation", "interpretation"}:
        return "interpretation"
    return "derived" if referenced_ids(view.get(data.id)) else "background"
