"""Typed prompt evidence and reusable collections of knoxels.

Stored metadata belongs to the memory owner. These explicit projections carry
only fields that a model needs to read or cite. Renderers never inspect a
record's arbitrary metadata dictionary.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from pm.context_render import ContextKind, render_context
from pm.context_render.turns import Content
from pm.records.base import Provenance, RecordBase


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

    def render(self, kinds: ContextKind = ContextKind.All, *, dedupe: bool = True) -> Content:
        """Render each source through the same central record renderer."""
        result = Content(self.render_header())
        seen: dict[int, RecordEvidence] = {}
        for record in self.records:
            result += "\n"
            if dedupe and record.knoxel_id in seen and record == seen[record.knoxel_id]:
                result += f"[source {record.knoxel_id}; already shown]"
            else:
                seen[record.knoxel_id] = record
                result += render_context(record, kinds, dedupe=dedupe)
        return result


class RecordReference(BaseModel):
    """Refer to an identical source already rendered earlier in this collection."""

    record_ref: int


@dataclass(frozen=True)
class DecisionEvidence(KnoxelCollection):
    """Original source content for an ordinary cognitive decision."""

    @property
    def mode(self) -> EvidenceMode:
        """Return the projection that an ordinary cognitive decision uses."""
        return EvidenceMode.Decision


@dataclass(frozen=True)
class ArchitecturalEvidence(KnoxelCollection):
    """Sources with explicit architecture fields supplied by the owning workflow."""

    @property
    def mode(self) -> EvidenceMode:
        """Return the projection that keeps explicit architecture fields."""
        return EvidenceMode.ArchitecturalReflection


def render_record(value: RecordEvidence, kinds: ContextKind = ContextKind.All, *, dedupe: bool = True) -> Content:
    """Render the named evidence projection. Omit unused optional model fields."""
    header = f"[source {value.knoxel_id}; revision {value.revision}; tick {value.tick_id}; {value.kind}]"
    if value.source:
        header += " " + value.source
    if value.evidence_role:
        header += " (" + value.evidence_role + ")"
    text = header + "\n" + value.text
    if value.reference_ids:
        text += "\nSupporting sources: " + ", ".join(str(key) for key in value.reference_ids)
    return Content(text)


def render_collection(value: KnoxelCollection, kinds: ContextKind = ContextKind.All, *, dedupe: bool = True) -> Content:
    """Dispatch collections through their typed rendering contract."""
    return value.render(kinds, dedupe=dedupe)



def render_stored_record(value: RecordBase, kinds: ContextKind = ContextKind.All, *, dedupe: bool = True) -> Content:
    from pm.context_render.memory_tool import render_memory_record
    return Content(render_memory_record(value))


def render_record_file(value: RecordBase) -> str:
    from pm.context_render.memory_tool import render_memory_record, MemoryRecordRenderSettings, RecordPurpose
    return render_memory_record(value, MemoryRecordRenderSettings(purpose=RecordPurpose.Inspection))
