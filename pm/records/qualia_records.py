"""Qualia schema (knoxel v2, spec §2.3).

Qualia channels are deterministic projections of the existing mental-state
vector ranges and knoxel-type statistics - no parallel embedding store.
Per cycle the ghost records channel position/velocity/acceleration; notable
movements become QualeKnoxels. A quale is reportable only when its driver
features were causal / won attention (causal gating, decision D9).

The full machinery lands in M10; the schema lands now so cycles, canon
converters and the persistence layer are stable from the start.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from pm.records.base import RecordBase


class QualeChannelState(BaseModel):
    """Per-cycle state of one qualia channel (stored in CycleState)."""

    channel_id: str = Field(description="Derived channel name (mental-state range or knoxel-type collection).")
    position: float = Field(default=0.0)
    velocity: float = Field(default=0.0)
    acceleration: float = Field(default=0.0)
    last_driver_ids: list[int] = Field(default_factory=list, description="Knoxel ids that last moved this channel.")


class QualeRecord(RecordBase):
    """Snapshot of a qualia channel written on notable movement.

    ``content`` = human readable description of the experience snapshot.
    ``reportable`` implements the causal gate: only qualia whose drivers
    won attention / entered the GWT may enter introspection or answers.
    """

    channel_id: str = Field(description="Channel this snapshot belongs to.")
    position: float = Field(default=0.0)
    velocity: float = Field(default=0.0)
    acceleration: float = Field(default=0.0)
    driver_ids: list[int] = Field(default_factory=list, description="Features that moved the channel.")
    reportable: bool = Field(default=False, description="True only if drivers were causal / won attention.")

    def get_story_element(self) -> str:
        return f"[Quale:{self.channel_id}] {self.content}"
