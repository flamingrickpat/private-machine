from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pm.records.base import RecordBase
from pm.records.memory_records import Feature, SemanticRecord
from pm.records.global_workspace import GwtTurnRecord
from pm.records.graph_records import GraphNode, GraphEdge, ConceptNode


class RecordPurpose(StrEnum):
    Recall = "recall"
    Inspection = "inspection"


@dataclass(frozen=True, slots=True)
class MemoryRecordRenderSettings:
    purpose: RecordPurpose = RecordPurpose.Recall
    include_identity: bool = True


def render_memory_record(record: RecordBase, settings: MemoryRecordRenderSettings = MemoryRecordRenderSettings()) -> str:
    lines: list[str] = []
    if settings.include_identity:
        when = record.timestamp_world_begin.isoformat(timespec="minutes")
        lines.append(f"Source {record.id} | {record.type.value} | {when}")
    if isinstance(record, Feature):
        source = record.override_source or f"entity {record.entity_id}"
        lines.append(f"{source} ({record.feature_type.value}): {record.content}")
    elif isinstance(record, GraphEdge):
        lines.append(f"Relation: source {record.source_id} --{record.label}--> source {record.target_id}")
        lines.append(record.fact_text)
        if record.content and record.content != record.fact_text:
            lines.append(record.content)
        if record.valid_at is not None:
            lines.append("Valid from: " + record.valid_at.isoformat(timespec="minutes"))
        if record.invalid_at is not None:
            lines.append("Invalid from: " + record.invalid_at.isoformat(timespec="minutes"))
    elif isinstance(record, GraphNode):
        lines.append("Entity: " + record.name)
        if record.content and record.content != record.name:
            lines.append(record.content)
    elif isinstance(record, ConceptNode):
        lines.append(record.description)
        if record.content and record.content != record.description:
            lines.append(record.content)
    else:
        lines.append(record.content)
    if record.provenance is not None:
        lines.append(f"Evidence origin: {record.provenance.source_kind.value}.")
    if record.scope.reality_id != -1 or record.scope.frame.value != "in_world":
        lines.append(f"Reality {record.scope.reality_id}; frame {record.scope.frame.value}.")
    if isinstance(record, SemanticRecord):
        lines.append(f"Epistemic status: {record.epistemic_status}.")
        if record.confidence is not None:
            lines.append(f"Confidence: {record.confidence:.3f}.")
        if record.contradicted_by_record_ids:
            lines.append("Contradicting sources: " + ", ".join(str(key) for key in record.contradicted_by_record_ids))
        if record.superseded_by_ids:
            lines.append("Superseded by: " + ", ".join(str(key) for key in record.superseded_by_ids))
    if settings.purpose is RecordPurpose.Inspection:
        lines.append(f"Created in tick {record.tick_id}; scope {record.scope.reality_id}/{record.scope.frame.value}.")
        if record.lineage is not None:
            lineage = record.lineage
            lines.append(f"Operation {lineage.operation}; episode {lineage.episode_id}; step {lineage.sequence}.")
            lines.append(f"Original sources: {lineage.source_event_ids}; parent records: {lineage.parent_record_ids}.")
    return "\n".join(lines)


def render_workspace_turn(turn: GwtTurnRecord) -> str:
    lines = [f"Workspace tick {turn.tick_id} ({turn.slot_id}) | {turn.timestamp.isoformat(timespec='minutes')}",
             turn.summary, f"Trigger sources: {turn.stimulus_ids}; selected sources: {turn.knoxel_ids}."]
    if turn.experience_id is not None:
        lines.append(f"Subjective experience source: {turn.experience_id}.")
    if turn.decision_id is not None:
        lines.append(f"Decision source: {turn.decision_id}.")
    for change in turn.shard_changes:
        lines.append(f"{change.key.kind}/{change.key.ref}: entered {[item.record_id for item in change.entered]}; "
                     f"retained {change.kept_ids}; history {change.tail_ids}.")
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class MemoryIndexEntry:
    record_id: int
    kind: str
    summary: str


@dataclass(frozen=True, slots=True)
class MemoryIndexInput:
    mode: str
    entries: tuple[MemoryIndexEntry, ...]


@dataclass(frozen=True, slots=True)
class MemoryIndexSettings:
    estimated_token_budget: int = 2048


def render_memory_index(input: MemoryIndexInput, settings: MemoryIndexSettings = MemoryIndexSettings()) -> str:
    from pm.context_render.budget import TokenEstimator
    from pm.context_render.turns import Role, Turn
    header = (f"Memory view: {input.mode}; {len(input.entries)} visible sources.\n"
              "Recent source index. Use ghost_search to find older relevant sources; ghost_get fetches complete evidence.\n")
    lines: list[str] = []
    estimator = TokenEstimator()
    for entry in reversed(input.entries):
        line = f"Source {entry.record_id} ({entry.kind}): {entry.summary}"
        trial = header + "\n".join([*lines, line])
        if estimator.estimate((Turn(Role.User, trial),)) > settings.estimated_token_budget:
            break
        lines.append(line)
    return header + "\n".join(lines)
