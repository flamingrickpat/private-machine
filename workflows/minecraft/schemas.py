"""Typed request and report for the Minecraft body workflow."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MinecraftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pm.minecraft-request.v1"] = "pm.minecraft-request.v1"
    run_id: str = ""
    instruction: str = ""
    artifact_path: str = ""
    answer: str = ""
    goal: str = ""
    body_id: str = "body"
    action_tool: str = ""
    action_arguments: dict = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    mode: Literal["actor", "boundary_check"] = "actor"
    specialist: Literal["gatherer", "navigator", "recovery", "builder", "miner"] = "gatherer"
    persona_context: dict = Field(default_factory=dict)
    acceptance: dict = Field(default_factory=dict)
    include_image: bool = True
    wait_for_cancel: bool = False
    wait_for_update: bool = False


class MinecraftReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["pm.minecraft-report.v1"] = "pm.minecraft-report.v1"
    goal: str
    body_id: str
    observations: list[str] = Field(default_factory=list)
    action: dict | None = None
    images: list[dict] = Field(default_factory=list)
    world: dict = Field(default_factory=dict)
    accepted: bool
    rejection_reason: str = ""


__all__ = ["MinecraftReport", "MinecraftRequest"]
