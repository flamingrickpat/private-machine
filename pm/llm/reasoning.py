"""Compute a reasoning allowance without capping the visible completion.

The caller supplies serving limits and a labelled input estimate. A budget
reserves answer space first. Invalid limits fail instead of removing context.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pm.llm.http_client import ModelError


class ReasoningOptions(BaseModel):
    """Describe thinking separately from the total completion limit.

    Omitted fields retain provider defaults. A token allowance and an effort
    level are alternative controls. Invalid combinations raise ValueError.
    Exclusion hides returned reasoning; it does not reduce generation work.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    enabled: bool | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None
    exclude: bool | None = None

    @model_validator(mode="after")
    def compatible_controls(self):
        """Reject contradictory controls instead of silently discarding one."""
        if self.max_tokens is not None and self.effort is not None:
            raise ValueError("Choose a reasoning token allowance or an effort level, not both.")
        if self.enabled is False and (self.max_tokens is not None or self.effort not in {None, "none"}):
            raise ValueError("Disabled thinking cannot have a positive reasoning allowance or effort.")
        if self.enabled is True and self.effort == "none":
            raise ValueError("Enabled thinking cannot use effort none.")
        return self


class ReasoningBudget(BaseModel):
    """Allocate a fraction of output headroom, with explicit token bounds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fraction: float = Field(ge=0, le=1)
    minimum_tokens: int = Field(default=128, gt=0)
    ceiling_tokens: int = Field(default=1024, gt=0)
    answer_reserve_tokens: int = Field(default=2048, gt=0)
    input_margin_fraction: float = Field(default=0.1, ge=0, le=1)

    def generation(self, *, context_window: int, input_tokens: int, max_output_tokens: int) -> dict:
        """Return neutral controls. Insufficient answer headroom raises ModelError."""
        if self.fraction == 0:
            return {"enable_thinking": False}
        remaining = min(max_output_tokens, context_window - int(input_tokens * (1 + self.input_margin_fraction)))
        headroom = remaining - self.answer_reserve_tokens
        if headroom < self.minimum_tokens:
            raise ModelError("The reasoning request has insufficient output headroom. Reduce input or adjust its budget.")
        tokens = min(headroom, self.ceiling_tokens, max(self.minimum_tokens, int(headroom * self.fraction)))
        return {"enable_thinking": True, "reasoning_tokens": tokens}
