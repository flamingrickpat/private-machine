"""GWT cycle records: the attention frame factory and the turn record.

archetype: information-holder
owns: the GWT turn record model (spec v2 §7), its YAML serialization, and
  the attention frame factory the personality cycle uses once per tick.
not own: GWT slot publishing (the loop engine writes gwt/ files), the
  attention frame lifecycle after the cycle stores it, markdown shards.
fails when: a turn record YAML file does not validate as a turn record.
domain: one accepted cycle produces one AttentionFrameRecord (a knoxel in
  the ghost) plus one GWT turn record (a projection file in gwt/). The
  turn record references the frame and the causal knoxel ids of the cycle,
  so readers can rebuild the broadcast from memory alone (decision D-M4-4).
invariant: a turn record always names exactly one slot, one tick, and one
  attention frame id.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from pm.utils import yaml_util
from pydantic import BaseModel, ConfigDict, Field

from pm.records.base import RecordScope, Provenance
from pm.records.enums import SourceKind
from pm.records.cognitive_records import AttentionFrameRecord
from pm.records.workspace import ShardKey, WorkspaceItem

GWT_TURN_SCHEMA = "pm.gwt-turn.v2"


class LeftItem(BaseModel):
    """One record that left a shard in this cycle, with the reason."""

    record_id: int = 0
    reason: str = ""


class ShardChange(BaseModel):
    """The change of one workspace shard in one cycle.

    ``entered`` holds the items that entered. ``left`` holds the records that
    left, with reasons. ``kept_ids`` names the incumbents that stayed.
    ``tail_ids`` is the tail of the shard after the change.
    """

    key: ShardKey = Field(default_factory=ShardKey)
    entered: List[WorkspaceItem] = Field(default_factory=list)
    left: List[LeftItem] = Field(default_factory=list)
    kept_ids: List[int] = Field(default_factory=list)
    tail_ids: List[int] = Field(default_factory=list)

    def rekey(self, mapping: dict[int, int]) -> None:
        """Replace pending record and structure ids by store ids.

        The attention phase records the change before the commit assigns
        record ids to new percepts and structures. Ids that are not in
        ``mapping`` stay unchanged.
        """
        for item in self.entered:
            item.record_id = mapping.get(item.record_id, item.record_id)
            if item.structure_id is not None:
                item.structure_id = mapping.get(item.structure_id, item.structure_id)
        for item in self.left:
            item.record_id = mapping.get(item.record_id, item.record_id)
        self.kept_ids = [mapping.get(record_id, record_id) for record_id in self.kept_ids]
        self.tail_ids = [mapping.get(record_id, record_id) for record_id in self.tail_ids]


class GwtTurnRecord(BaseModel):
    """One published GWT broadcast for one slot and one cycle (spec §7).

    Cycle publishes always name one attention frame. Hidden slot publishes
    (dialogue, M5) carry no frame; the field stays None there.

    Schema v2 (A19F) adds ``shard_changes`` and ``experience_id``. Old v1
    turns keep their schema id and load with empty v2 fields. They are not
    migrated. ``timestamp`` is naive UTC, like the record timestamps in
    ``pm.records.base``.
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_id: str = Field(default=GWT_TURN_SCHEMA, alias="schema")
    tick_id: int
    slot_id: str = "main"
    attention_frame_id: Optional[int] = None
    knoxel_ids: List[int] = Field(default_factory=list)
    summary: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    shard_changes: List[ShardChange] = Field(default_factory=list)
    experience_id: Optional[int] = None

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json", by_alias=True)
        return yaml_util.safe_dump(data, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> "GwtTurnRecord":
        return cls.model_validate(yaml_util.safe_load(text))


def load_turn_record(path: str | Path) -> GwtTurnRecord:
    return GwtTurnRecord.from_yaml(Path(path).read_text(encoding="utf-8"))


def build_attention_frame(
    *,
    tick_id: int,
    content: str,
    focus_knoxel_ids: List[int],
    active_goal_ids: List[int],
    frame_reason: str,
    reality_id: int,
    control_mode: str = "reactive",
    winning_attention_ids: Optional[List[int]] = None,
) -> AttentionFrameRecord:
    """Build the one attention frame a cycle publishes (decision D-M4-4).

    The caller assigns the knoxel id through the ghost service and stores
    the frame together with the other cycle knoxels in one batch.
    """
    frame = AttentionFrameRecord(
        content=content,
        tick_id=tick_id,
        focus_knoxel_ids=list(focus_knoxel_ids),
        winning_attention_ids=list(winning_attention_ids or []),
        active_goal_ids=list(active_goal_ids),
        control_mode=control_mode,
        frame_reason=frame_reason,
    )
    frame.scope = RecordScope(reality_id=reality_id)
    frame.provenance = Provenance(
        source_kind=SourceKind.Workflow, source_ref="personality-cycle"
    )
    return frame
