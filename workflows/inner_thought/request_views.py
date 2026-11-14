"""Named read-side evidence for private requests.

See :mod:`workflows.inner_thought.host` for the private loop overview. These
views preserve the host's selected records and inspection evidence. They do
not establish new memory, judgments, or execution facts.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

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


class InquiryThoughtView(BaseModel):
    """The executed inquiry step behind one recent private thought.

    The private loop reads these fields from the operation's own accepted
    metadata. They describe what the step did, not a new judgment about it.
    """

    model_config = ConfigDict(extra="forbid")
    episode_id: str = Field(description="The private inquiry episode that owns this executed step.")
    outcome: Literal["continue", "finding", "information_gap", "wait", "pause", "completed", "abandoned"] = Field(
        description="How the step left its question: continue, a finding, a stated gap, a wait, a pause, or an end.")
    reason: str = Field(description="Why the step continued or stopped, in its own recorded words.")
    information_gap: str = Field(
        description="The specific missing evidence the step states; empty when it states none.")
    progress_kind: Literal["new_evidence", "discrimination", "none"] = Field(
        description="Accepted progress: newly inspected evidence, a meaningful comparison, or none.")


class RecentThoughtView(PrivateSource):
    """A retained private operation, with its exact cited sources.

    ``loop`` names the path that produced the record. A linear thought carries
    no inquiry step, so ``inquiry`` is null rather than an invented episode.
    """

    loop: Literal["inner", "inquiry"] = Field(
        description="The private path that produced the thought: the legacy linear loop or the inquiry graph.")
    trigger: str = ""
    continuation: str = ""
    inquiry: InquiryThoughtView | None = Field(
        default=None,
        description="The executed graph step for an inquiry thought; null for a linear thought.")

    @model_validator(mode="after")
    def loop_matches_inquiry(self) -> Self:
        """Keep the loop label and the step view from contradicting each other."""
        if (self.loop == "inquiry") != (self.inquiry is not None):
            raise ValueError("An inquiry thought requires its step view, and a linear thought has none.")
        return self


class RepeatedSourceActivity(BaseModel):
    """One source that several recent private steps cited, with no judgment."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    source_id: int = Field(description="A source that several recent private steps cite.")
    step_ids: tuple[int, ...] = Field(description="Recent private steps that cite it, newest first.")
    stalled_step_ids: tuple[int, ...] = Field(description="Those of them that accepted no progress.")
    episode_ids: tuple[str, ...] = Field(description="Inquiry episodes those steps belong to.")


class ThoughtActivityView(BaseModel):
    """Recorded facts about recent private thinking. No judgment and no cause."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    window_step_ids: tuple[int, ...] = Field(description="The recent private steps counted, newest first.")
    repeated: tuple[RepeatedSourceActivity, ...] = Field(
        description="Every source cited by at least two of those steps, with the steps that cite it.")
    stalled_step_count: int = Field(ge=0, description="Inquiry steps that accepted no progress; linear thoughts carry no such judgment.")
    new_evidence_step_count: int = Field(ge=0, description="Inquiry steps that accepted newly inspected original evidence.")


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
