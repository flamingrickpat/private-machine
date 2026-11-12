"""Named read-side evidence for private requests.

See :mod:`workflows.inner_thought.host` for the private loop overview. These
views preserve the host's selected records and inspection evidence. They do
not establish new memory, judgments, or execution facts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from pm.introspection.episodes import EpisodeAnswer, EpisodeRow


class PrivateSource(BaseModel):
    """One original record supplied as evidence, with its captured revision.

    Retained expectation artifacts used ``content``. The boundary reads that
    historical spelling; the current host always serializes ``text``.
    """

    model_config = ConfigDict(extra="ignore")
    knoxel_id: int
    text: str = Field(validation_alias=AliasChoices("text", "content"))
    revision: int = 1
    tick_id: int = 0
    operation: str = ""
    source_ids: list[int] = Field(default_factory=list[int])


class GoalView(PrivateSource):
    """Accepted intention content and its existing progress readout."""

    fulfilment: float = 0.0
    affective_valence: float = 0.0


class ExpectationView(PrivateSource):
    """An unresolved anticipation, without an invented observed outcome."""

    anticipation: str = ""
    expected_outcome: str = ""


class RecentThoughtView(PrivateSource):
    """A retained private operation, with its exact cited sources."""

    trigger: str = ""
    continuation: str = ""


class MainFrameView(BaseModel):
    """The last accepted main turn and its actual source classification."""

    model_config = ConfigDict(extra="forbid")
    tick: int = 0
    frame_id: int | None = None
    summary: str = ""
    surprise: float = 0.0
    gap_seconds: float = 0.0
    self_feedback: bool = False


class OperationSuggestion(BaseModel):
    """One rendered entry from the delivered operation catalogue."""

    model_config = ConfigDict(extra="forbid")
    name: str


class InspectedCapability(BaseModel):
    """The selected capability finding and its implementation revision."""

    model_config = ConfigDict(extra="forbid")
    name: str
    state: str
    summary: str = ""
    revision: str = ""


class CapabilityInspection(BaseModel):
    """The actual public projection of an inspected capability overlay."""

    model_config = ConfigDict(extra="forbid")
    available: bool = False
    checked_at: datetime | None = None
    stale: list[str] = Field(default_factory=list[str])
    capabilities: list[InspectedCapability] = Field(default_factory=list[InspectedCapability])


class PrivateInspection(BaseModel):
    """Selected implementation evidence or the provider's explicit limit.

    Episode rows and answers retain the existing named introspection boundary
    contracts. The question-name map is a real lookup, not an anonymous record.
    """

    model_config = ConfigDict(extra="forbid")
    status: Literal["ok", "unavailable"]
    reason: str = ""
    capability_overlay: CapabilityInspection = Field(default_factory=CapabilityInspection)
    episodes: list[EpisodeRow] = Field(default_factory=list[EpisodeRow])
    newest_episode_answers: dict[str, EpisodeAnswer] = Field(default_factory=dict[str, EpisodeAnswer])
    observed_at: datetime | None = None
