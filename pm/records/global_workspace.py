"""GWT cycle records: the turn record of one accepted cycle.

archetype: information-holder
owns: the GWT turn record model (``pm.gwt-turn.v2``) and its YAML
  serialization.
not own: the workspace shards (``pm.records.workspace``), the cycle state
  (``pm.records.cycle_state``), and the records that the turn names.
domain: one accepted cycle of a loop produces one turn record. A main cycle
  lists the shard changes, the claimed stimuli, the subjective experience,
  and the trace of its codelets. An inner cycle (``slot_id="inner"``) lists
  its thought and its qualia. Readers rebuild the broadcast from the store
  alone.
invariant: a turn record names exactly one slot and one tick.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pm.utils import yaml_util
from pydantic import BaseModel, ConfigDict, Field

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


class CodeletRunRef(BaseModel):
    """One codelet run of a cycle, as the turn record keeps it.

    ``score`` is the activation score that chose the codelet in its wave.
    ``calls`` counts model calls; an agent session counts as one call.
    ``status`` is ``ran`` or the reason that the run produced nothing.
    """

    codelet: str = ""
    mode: str = ""
    wave: int = 0
    status: str = ""
    reason: str = ""
    score: float = 0.0
    calls: int = 0


class PathwayBoostRef(BaseModel):
    """One pathway boost of a cycle: ``source`` ran, and ``target`` got ``strength``."""

    pathway: str = ""
    source: str = ""
    target: str = ""
    strength: float = 0.0
    skipped: List[str] = Field(default_factory=list)


class CycleTrace(BaseModel):
    """The codelet trace of one main cycle.

    ``percept_ids`` and ``structure_ids`` are the records that the cycle
    created. ``causal_percept_ids`` are the percepts that entered a shard in
    this cycle. The attention scores of each structure are in the metadata
    of its ``Structure`` record. ``input_tokens_estimate`` and
    ``output_tokens_estimate`` are tiktoken estimates of the codelet calls.
    ``selector_patterns`` is the selector's note on the patterns of the
    recent story. ``echo_ids`` are the earlier lines that echo a current
    stimulus; the selector read them.
    """

    codelet_budget: int = 0
    codelet_runs: List[CodeletRunRef] = Field(default_factory=list)
    codelet_ratings: dict[str, float] = Field(default_factory=dict)
    selector_patterns: str = ""
    echo_ids: List[int] = Field(default_factory=list)
    pathway_boosts: List[PathwayBoostRef] = Field(default_factory=list)
    definitions_version: str = ""
    definitions_digest: str = ""
    percept_ids: List[int] = Field(default_factory=list)
    structure_ids: List[int] = Field(default_factory=list)
    causal_percept_ids: List[int] = Field(default_factory=list)
    llm_calls: int = 0
    input_tokens_estimate: int = 0
    output_tokens_estimate: int = 0


class GwtTurnRecord(BaseModel):
    """One published GWT broadcast for one slot and one cycle.

    ``slot_id`` names the loop: ``main`` for main cognition and ``inner`` for
    the private loop. ``stimulus_ids`` are the stimuli that the cycle
    claimed. ``knoxel_ids`` lists the stimuli, the created records, the
    selected memories, and every workspace item and tail record.
    ``shard_changes`` holds one change for each updated shard.
    ``experience_id`` is the subjective experience of the cycle.
    ``decision_id`` is the ``StateUpdate`` record that holds the decisions
    and routes of the cycle. ``trace`` holds the codelet trace.

    Old v1 turns keep their schema id and load with empty v2 fields. Their
    ``attention_frame_id`` key is ignored. They are not migrated.
    ``timestamp`` is naive UTC, like the record timestamps in
    ``pm.records.base``.
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_id: str = Field(default=GWT_TURN_SCHEMA, alias="schema")
    tick_id: int
    slot_id: str = "main"
    stimulus_ids: List[int] = Field(default_factory=list)
    knoxel_ids: List[int] = Field(default_factory=list)
    summary: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    shard_changes: List[ShardChange] = Field(default_factory=list)
    experience_id: Optional[int] = None
    decision_id: Optional[int] = None
    trace: CycleTrace = Field(default_factory=CycleTrace)

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json", by_alias=True)
        return yaml_util.safe_dump(data, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> "GwtTurnRecord":
        return cls.model_validate(yaml_util.safe_load(text))
