"""Retain episode membership without assigning a permanent memory topic."""
from pydantic import BaseModel, Field


class RecordLineage(BaseModel):
    """Link an executed operation to its episode, causes, and goals.

    Topic hints are advisory labels. Later consolidation can place one record
    in several clusters without changing its identity or these causal links.
    Reality labels describe the content, not whether the operation ran.
    """

    episode_id: str
    operation: str
    sequence: int = Field(default=0, ge=0)
    parent_record_ids: list[int] = Field(default_factory=list)
    source_event_ids: list[int] = Field(default_factory=list)
    intention_ids: list[int] = Field(default_factory=list)
    topic_hints: list[str] = Field(default_factory=list)
    reality: str = "internal"
