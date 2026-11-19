from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from workflows.minecraft_project.services import ProjectAuditOwner
from pydantic import BaseModel, ConfigDict
from typing import Generator
from pm_workflows.runtime import WorkflowRuntime
from pm.runs.role_runtime import PMWorkflowRuntime
from pm.runs.managed_boundary import BoundaryReadSettings, BoundaryInstructions
from workflows.minecraft_body.mcp_client import MinecraftTransport
from workflows.minecraft_body.lease import BODY_LEASES


class BodyLeaseEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    endpoint: str
    owner: str
    epoch: int
    session: str
    pid: int
    active: bool
    acquired_at: float


@dataclass(frozen=True, slots=True)
class MinecraftBodyLease:
    endpoint: str
    owner: str

    @contextmanager
    def acquire(self) -> Generator[BodyLeaseEvidence, None, None]:
        with BODY_LEASES.acquire(self.endpoint, self.owner) as token:
            yield BodyLeaseEvidence.model_validate(token)


@dataclass(kw_only=True)
class MinecraftWorkflowRuntime(PMWorkflowRuntime):
    minecraft_transport: MinecraftTransport
    project_audit: ProjectAuditOwner | None = None
    evidence_sources: dict[str, Path] = field(default_factory=dict[str, Path])
    body_lease: MinecraftBodyLease

    def instructions(self, settings: BoundaryReadSettings = BoundaryReadSettings()) -> BoundaryInstructions:
        if self.managed_boundary is None:
            raise RuntimeError("Minecraft instructions require their managed-run boundary.")
        return self.managed_boundary.read(settings)


def require_minecraft_runtime(runtime: WorkflowRuntime | None) -> MinecraftWorkflowRuntime:
    if not isinstance(runtime, MinecraftWorkflowRuntime):
        raise TypeError("Minecraft operations require MinecraftWorkflowRuntime with its concrete body transport.")
    return runtime
