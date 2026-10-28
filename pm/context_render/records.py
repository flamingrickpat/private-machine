"""Typed prompt evidence and reusable collections of knoxels.

Stored metadata belongs to the memory owner. These explicit projections carry
only fields that a model needs to read or cite. Renderers never inspect a
record's arbitrary metadata dictionary.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from typing import cast, TypedDict
import json

from pm.context_render import ContextKind, render_context
from pm.context_render.turns import Content
from pm.records.base import Provenance, RecordBase, persisted_type_name


class CitedMemory(BaseModel):
    """The actual memory content supplied to an earlier decision."""

    model_config = ConfigDict(extra="ignore")
    knoxel_id: int
    revision: int
    text: str
    source_ids: list[int] = Field(default_factory=list[int])
    score: float = 0
    reasons: list[str] = Field(default_factory=list[str])


class MemorySelection(BaseModel):
    """Named retrieval facts useful for architectural reflection."""

    model_config = ConfigDict(extra="ignore")
    query: str = ""
    original_query: str = ""
    requested_method: str = ""
    effective_method: str = ""
    selected_ids: list[int] = Field(default_factory=list[int])
    context_selected_ids: list[int] = Field(default_factory=list[int])


class RecordArchitecture(BaseModel):
    """Explicit architecture fields projected from heterogeneous record metadata.

    Other stored metadata remains on disk. In particular, source-revision maps,
    dispatch payloads, full memory traces, and embeddings never enter this projection.
    """

    model_config = ConfigDict(extra="ignore")
    schema_version: str = Field(default="", alias="schema")
    reflection_versions: dict[int, int] = Field(default_factory=dict[int, int], description="Experiment IDs and effective versions consumed by this decision.")
    operation: str = ""
    origin: str = ""
    loop: str = ""
    control_kind: str = ""
    status: str = ""
    memory_evidence: list[CitedMemory] = Field(default_factory=list[CitedMemory])
    memory_trace: MemorySelection = Field(default_factory=MemorySelection)


class RecordEvidence(BaseModel):
    """Citable content from one exact knoxel in a frozen memory view."""

    model_config = ConfigDict(extra="forbid")
    knoxel_id: int = Field(description="The immutable source knoxel ID.")
    revision: int = Field(description="The captured source revision, including legacy imports.")
    tick_id: int = Field(description="The source's creation tick. This is not an admission decision.")
    text: str = Field(description="Original source content, without execution metadata.")
    kind: str = Field(description="The concrete stored record kind.")
    reference_ids: list[int] = Field(default_factory=list[int], description="Explicit links to other source knoxels.")
    evidence_role: str = Field(default="", description="The source's epistemic role, such as user_report.")
    source: str = Field(default="", description="The named speaker or source when the record has one.")
    provenance: Provenance | None = None


class EvidenceMode(StrEnum):
    """The purpose of a prompt collection, independent of persisted payloads."""

    Decision = "decision"
    ArchitecturalReflection = "architectural_reflection"


@dataclass(frozen=True)
class KnoxelCollection(ABC):
    """Share record rendering while allowing a use case to add a typed header."""

    records: tuple[RecordEvidence, ...]

    @property
    @abstractmethod
    def mode(self) -> EvidenceMode:
        """Name the collection's purpose for the model and execution receipt."""

    def render_header(self) -> str:
        """Describe the collection without inventing evidence or source links."""
        return f"Evidence collection: {self.mode.value}"

    @property
    def schema_path(self) -> Path | None:
        """Use the ordinary profile unless this collection has a specialized one."""
        return None

    def render(self, kinds: ContextKind = ContextKind.All, *, schema_path: str | Path | None = None,
               dedupe: bool = True) -> Content:
        """Render each source through the same central record renderer."""
        result = Content(self.render_header())
        profile = self.schema_path if schema_path is None else schema_path
        seen: dict[int, RecordEvidence] = {}
        for record in self.records:
            result += "\n"
            if dedupe and record.knoxel_id in seen and record == seen[record.knoxel_id]:
                result += render_context(RecordReference(record_ref=record.knoxel_id), kinds, schema_path=profile)
            else:
                seen[record.knoxel_id] = record
                result += render_context(record, kinds, schema_path=profile, dedupe=dedupe)
        return result


class RecordReference(BaseModel):
    """Refer to an identical source already rendered earlier in this collection."""

    record_ref: int


@dataclass(frozen=True)
class DecisionEvidence(KnoxelCollection):
    """Original source content for an ordinary cognitive decision."""

    @property
    def mode(self) -> EvidenceMode:
        return EvidenceMode.Decision


@dataclass(frozen=True)
class ArchitecturalEvidence(KnoxelCollection):
    """Sources with explicit architecture fields supplied by the owning workflow."""

    @property
    def mode(self) -> EvidenceMode:
        return EvidenceMode.ArchitecturalReflection

    @property
    def schema_path(self) -> Path:
        return Path(__file__).with_name("architectural.json")


@render_context.register(RecordEvidence)
def render_record(value: RecordEvidence, kinds: ContextKind = ContextKind.All, *,
                  schema_path: str | Path | None = None, dedupe: bool = True) -> Content:
    """Render the named evidence projection. Omit unused optional model fields."""
    projection = value.model_dump(mode="python", exclude_defaults=True)
    return render_context(projection, kinds, schema_path=schema_path, dedupe=dedupe)


@render_context.register(KnoxelCollection)
def render_collection(value: KnoxelCollection, kinds: ContextKind = ContextKind.All, *,
                      schema_path: str | Path | None = None, dedupe: bool = True) -> Content:
    """Dispatch collections through their typed rendering contract."""
    return value.render(kinds, schema_path=schema_path, dedupe=dedupe)


class PromptMetadata(RecordArchitecture):
    """Named cognitive metadata available in ordinary model-facing record files."""

    memory_kind: str = ""
    reality: str = ""
    uncertainty: JsonValue = None
    reason: str = ""
    affect_summary: JsonValue = None
    evidence_ids: list[int] = Field(default_factory=list[int])
    source_ids: list[int] = Field(default_factory=list[int])
    execution_status: str = ""
    exchange_status: str = ""
    experiment_id: int | None = None
    effective_record_id: int | None = None
    state: str = ""
    result: str = ""
    plan: JsonValue = None
    previous: JsonValue = None
    history: list[JsonValue] = Field(default_factory=list[JsonValue])
    observation_ids: list[int] = Field(default_factory=list[int])


class StoredRecordContext(BaseModel):
    """The record-file wire format after deliberate prompt projection.

    Only this serializer handles heterogeneous semantic fields. Internal users
    keep their concrete record types. Full runtime metadata stays on disk.
    """

    record_kind: str = Field(serialization_alias="__type__")
    data: dict[str, JsonValue]


def _stored_projection(value: RecordBase) -> StoredRecordContext:
    """Project one concrete record at the model-facing serialization boundary."""
    fields = cast(dict[str, JsonValue], value.model_dump(mode="json", serialize_as_any=True,
        exclude={"metadata", "embedding", "embedding_model", "embedding_query_hint", "last_accessed"}))
    metadata = PromptMetadata.model_validate(value.metadata)
    fields["metadata"] = metadata.model_dump(mode="json", by_alias=True, exclude_defaults=True)
    return StoredRecordContext(record_kind=persisted_type_name(type(value)), data=fields)


@render_context.register(RecordBase)
def render_stored_record(value: RecordBase, kinds: ContextKind = ContextKind.All, *,
                         schema_path: str | Path | None = None, dedupe: bool = True) -> Content:
    """Render semantic record fields and an explicit metadata projection."""
    return render_context(_stored_projection(value).data, kinds, schema_path=schema_path, dedupe=dedupe)


def render_record_file(value: RecordBase) -> str:
    """Keep virtual record files valid JSON while applying central prompt policy."""
    projected = _stored_projection(value)
    return render_context(projected.model_dump(by_alias=True), ContextKind.Facts, schema_path=record_file_schema()).text


def record_file_schema() -> Path:
    """Locate the precise, machine-readable profile shared by model-facing files."""
    return Path(__file__).with_name("record_file.json")


class RecordFile(TypedDict):
    """The model-facing JSON envelope at legacy tool transport boundaries."""

    __type__: str
    data: dict[str, JsonValue]


def record_file_payload(value: RecordBase) -> RecordFile:
    """Reuse the exact central file projection for object-valued tool responses."""
    return cast(RecordFile, json.loads(render_record_file(value)))
