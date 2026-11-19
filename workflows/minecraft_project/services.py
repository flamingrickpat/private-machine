from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from pydantic import JsonValue
from workflows.minecraft_project.schemas import TaskTree
from workflows.minecraft_project.planning.context_contracts import CallerAuditResult


@dataclass(frozen=True, slots=True)
class CallerAuditEnvironment:
    initial_observation: JsonValue
    character: Path


class ProjectAuditOwner(Protocol):
    def bind(self, environment: CallerAuditEnvironment) -> ProjectAuditOwner: ...
    def plan_constraints(self, tree: TaskTree) -> CallerAuditResult: ...
    def check(self, root: Path, tree: TaskTree) -> CallerAuditResult: ...
