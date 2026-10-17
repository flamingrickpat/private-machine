"""Validated limits and advisory requests for deliberate mental control."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class ControlConfig(BaseModel):
    """Operator limits. Durations use seconds of elapsed real time."""

    model_config = ConfigDict(extra="forbid")
    wait_seconds: float = Field(default=1800, ge=1, le=86400)
    maximum_wait_seconds: float = Field(default=86400, ge=1, le=86400)
    focus_seconds: float = Field(default=900, ge=1, le=86400)
    maximum_focus_bias: float = Field(default=0.35, ge=0, le=0.5)
    episode_steps: int = Field(default=6, ge=1, le=32)
    episode_seconds: float = Field(default=600, ge=1, le=3600)
    stalled_steps: int = Field(default=2, ge=1, le=8)
    parked_seconds: float = Field(default=900, ge=1, le=86400)
    mental_timeout_seconds: float = Field(default=600, ge=1, le=3600)


class WaitRequest(BaseModel):
    """A deliberate inference pause. External events can end it early."""

    seconds: FiniteFloat = Field(default=1800, ge=1, le=86400, strict=True)
    reason: str = "Wait for a new stimulus."


class FocusProposal(BaseModel):
    """Temporary influence on attention. This proposal never erases a record."""

    instruction: str
    promote_ids: list[int] = Field(default_factory=list)
    reduce_ids: list[int] = Field(default_factory=list)
    strength: float = Field(default=0.3, ge=0, le=0.5)
    seconds: float = Field(default=900, ge=1, le=86400)
    topic_hints: list[str] = Field(default_factory=list)
    reason: str = ""


class MentalPlan(BaseModel):
    """A private agent's proposal. Main cognition owns its acceptance."""

    model_config = ConfigDict(extra="forbid")
    instruction: str
    focus: FocusProposal | None = None
    goal_changes: list[dict] = Field(default_factory=list)
    evidence_ids: list[int] = Field(default_factory=list)
    summary: str
    disposition: Literal["propose", "no_change"] = "propose"
