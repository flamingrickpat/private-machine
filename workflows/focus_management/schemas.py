"""Typed request of one deliberate mental work session."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MentalWorkRequest(BaseModel):
    """The input of one mental work session.

    ``instruction`` is the private mental task. ``workspace`` is the shard
    story of the cycle that chose the task: the self shard, the partner
    shard, and a workflow shard with their tails. ``experience`` is the
    subjective experience of that cycle and ``affect`` the feeling in words.
    ``timeout_seconds`` bounds the agent session.
    """

    model_config = ConfigDict(extra="forbid")

    instruction: str = ""
    companion: str = ""
    timeout_seconds: float = 600
    character: str = ""
    workspace: str = ""
    experience: str = ""
    affect: str = ""


__all__ = ["MentalWorkRequest"]
