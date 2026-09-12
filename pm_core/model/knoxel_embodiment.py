from __future__ import annotations

from typing import Any

from pydantic import Field

from pm_core.model.knoxel_core import KnoxelBase


class ExecutableSkillKnoxel(KnoxelBase):
    """
    OBSOLETE: DO NEVER USE!
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


class WorldPlaceKnoxel(KnoxelBase):
    body_id: str
    place_key: str
    position: dict[str, Any] = Field(default_factory=dict)
    dimension: str = "unknown"
    landmarks: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    source_event_ids: list[int] = Field(default_factory=list)

    def get_story_element(self) -> str:
        return f"Known place {self.place_key} in {self.dimension}: {self.content}"


class WorldObjectKnoxel(KnoxelBase):
    body_id: str
    object_key: str
    object_kind: str
    position: dict[str, Any] = Field(default_factory=dict)
    dimension: str = "unknown"
    confidence: float = 0.0
    source_event_ids: list[int] = Field(default_factory=list)

    def get_story_element(self) -> str:
        return f"Known {self.object_kind} {self.object_key}: {self.content}"


class BodyEpisodeKnoxel(KnoxelBase):
    """One embodied experience, recorded at appraisal time.

    The raw episode arrives as a personality stimulus; the feature appears
    only when the cycle appraises it. ``evidence`` links the durable raw
    material (screenshot path, state snapshot reference, source stimulus)
    so any later process can inspect what the body experienced.
    """

    body_id: str
    goal: str | None = None
    outcome: str = "unknown"
    action_count: int = 0
    evidence: dict[str, Any] = Field(default_factory=dict)

    def get_story_element(self) -> str:
        goal_part = f" while pursuing: {self.goal}" if self.goal else ""
        return (
            f"Body episode for {self.body_id}{goal_part} -> "
            f"{self.outcome}. {self.content}"
        )


__all__ = [
    "BodyEpisodeKnoxel",
    "ExecutableSkillKnoxel",
    "WorldObjectKnoxel",
    "WorldPlaceKnoxel",
]
