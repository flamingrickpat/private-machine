"""What the body learned about a world it acts in, and one obsolete skill record.

owns: ``WorldPlaceRecord`` and ``WorldObjectRecord`` (known places and
objects of an embodied body), ``BodyEpisodeRecord`` (one embodied experience
with its evidence), and the unused ``ExecutableSkillRecord`` that old
checkpoints still contain. See ``pm.records.base`` for the package overview.

Keys are per body: ``body_id`` plus ``place_key`` or ``object_key`` identifies
the thing, and ``position`` stays the free-form JSON shape that the body
reports. ``confidence`` is the belief strength of the entry, and
``source_event_ids`` link it to the world events that taught it.
"""
from __future__ import annotations

from typing import Any

from pydantic import Field

from pm.records.base import RecordBase


class ExecutableSkillRecord(RecordBase):
    """
    OBSOLETE: DO NEVER USE!

    Kept only so that checkpoints and seeds that still contain the record
    class load. No workflow creates or runs it.
    """
    name: str
    body_kinds: list[str] = Field(default_factory=list)
    description: str = ""
    source_code: str
    entrypoint: str = "run"
    inputs_schema: dict[str, Any] = Field(default_factory=dict)
    success_criteria: str = ""
    failure_patterns: list[str] = Field(default_factory=list)
    usage_notes: str = ""
    version: int = 1
    enabled: bool = True
    created_by: str = "seed"
    last_result: dict[str, Any] = Field(default_factory=dict)

    def get_story_element(self) -> str:
        status = "enabled" if self.enabled else "disabled"
        return f"Executable skill {self.name} v{self.version} ({status}): {self.description}"


class WorldPlaceRecord(RecordBase):
    """A place that a body has been to or been told about.

    ``place_key`` is the body's own name for the place and ``position`` its
    reported coordinates, so two bodies can name the same spot differently.
    """

    body_id: str
    place_key: str
    position: dict[str, Any] = Field(default_factory=dict)
    dimension: str = "unknown"
    landmarks: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    source_event_ids: list[int] = Field(default_factory=list)

    def get_story_element(self) -> str:
        return f"Known place {self.place_key} in {self.dimension}: {self.content}"


class WorldObjectRecord(RecordBase):
    """One object that a body noticed, with where it was seen and how sure the body is."""

    body_id: str
    object_key: str
    object_kind: str
    position: dict[str, Any] = Field(default_factory=dict)
    dimension: str = "unknown"
    confidence: float = 0.0
    source_event_ids: list[int] = Field(default_factory=list)

    def get_story_element(self) -> str:
        return f"Known {self.object_kind} {self.object_key}: {self.content}"


class BodyEpisodeRecord(RecordBase):
    """One embodied experience, recorded at appraisal time.

    The raw episode arrives as a personality stimulus; the feature appears
    only when the cycle appraises it. ``evidence`` links the durable raw
    material (screenshot path, state snapshot reference, source stimulus)
    so any later process can inspect what the body experienced.
    """

    body_id: str
    goal: str | None = Field(default=None, description="The goal of the episode, when the body had one.")
    outcome: str = Field(default="unknown", description="What came of the episode, in the body's words.")
    action_count: int = Field(default=0, description="Number of actions the body took during the episode.")
    evidence: dict[str, Any] = Field(default_factory=dict)

    def get_story_element(self) -> str:
        goal_part = f" while pursuing: {self.goal}" if self.goal else ""
        return (
            f"Body episode for {self.body_id}{goal_part} -> "
            f"{self.outcome}. {self.content}"
        )


__all__ = [
    "BodyEpisodeRecord",
    "ExecutableSkillRecord",
    "WorldObjectRecord",
    "WorldPlaceRecord",
]
