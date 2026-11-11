"""Frozen requests and durable outcomes for ordinary character choices.

See :mod:`workflows.interaction.host` for selection authority and recovery.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from workflows.main_cognition.cycle_work import CycleRequest
from workflows.main_cognition.schemas import ActionDispatch, ActionRequest, DecisionAnswer, RouteChoice
from workflows.public_speech.dependencies import WorkDependencies


class GuidanceDisposition(StrEnum):
    """How this exchange uses the accepted direction, without revising it."""

    Follow = "follow"
    Decline = "decline"
    Suspend = "suspend"


class InteractionOutcome(StrEnum):
    """The local outcome or explicit request for main reconsideration."""

    Reply = "reply"
    Quiet = "quiet"
    Reconsider = "reconsider"


class InteractionDecision(DecisionAnswer):
    """The existing character decision plus its use of accepted guidance."""

    guidance_disposition: GuidanceDisposition = GuidanceDisposition.Follow
    guidance_reason: str = Field(default="", description="Why the local exchange declines or suspends prior direction.")

    @model_validator(mode="after")
    def explain_mismatch(self) -> Self:
        """Require an explicit reason when the character declines accepted direction."""
        if self.guidance_disposition != GuidanceDisposition.Follow and not self.guidance_reason.strip():
            raise ValueError("Declining or suspending guidance requires the situational reason.")
        return self


class InteractionRequest(BaseModel):
    """The exact evidence supplied to one ordinary selection kernel."""

    model_config = ConfigDict(extra="forbid")
    selection_id: str
    cycle: CycleRequest
    dependencies: WorkDependencies
    intention_ids: list[int] = Field(default_factory=list[int])


class InteractionReceipt(BaseModel):
    """An accepted local choice, with evidence and a replayable speech action."""

    model_config = ConfigDict(extra="forbid")
    selection_id: str
    source_ids: list[int] = Field(min_length=1)
    outcome: InteractionOutcome
    decision: InteractionDecision
    route: RouteChoice | None = None
    dispatch: ActionDispatch | None = None
    dependencies: WorkDependencies
    action_request: ActionRequest | None = None

    @model_validator(mode="after")
    def coherent_outcome(self) -> Self:
        """Reject contradictory internal results before the host writes them."""
        if self.outcome == InteractionOutcome.Reply:
            if self.dispatch is None or self.route is None:
                raise ValueError("An ordinary reply requires its selected route and dispatch.")
            if self.dispatch.action_type != "Reply" or self.dispatch.workflow_name != "dialogue_story":
                raise ValueError("An ordinary reply must use the existing Story speech owner.")
        elif self.dispatch is not None:
            raise ValueError("Only an ordinary reply can directly dispatch an action.")
        return self


class InteractionMetadata(BaseModel):
    """Validated view of the open record metadata at its storage boundary."""

    model_config = ConfigDict(extra="ignore")
    interaction_receipt: InteractionReceipt
