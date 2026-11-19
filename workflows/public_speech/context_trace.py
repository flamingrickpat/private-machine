from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict
from pm.memory.context_plan import BudgetRedistribution, CompetingRecord
from pm.records.workspace import ShardKey


class SemanticContextTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal["disabled", "ready"] = "disabled"
    model: str = ""


class ContextLaneTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    budget: int
    used: int
    selected: int
    candidates: int
    exhausted: bool
    competing: tuple[CompetingRecord, ...] = ()


class ContextShardTrace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str
    key: ShardKey
    items: int
    tail: int


class ContextPlanningTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Literal["retrieval", "shards"] = "retrieval"
    plan: str = ""
    semantic: SemanticContextTrace = SemanticContextTrace()
    new_embeddings: int = 0
    rounds: tuple[BudgetRedistribution, ...] = ()
    lanes: tuple[ContextLaneTrace, ...] = ()
    shards: tuple[ContextShardTrace, ...] = ()
    transcript: int = 0
    empty_transcript_ids: tuple[int, ...] = ()
    late_ids: tuple[int, ...] = ()
    planned_lanes: tuple[ContextLaneTrace, ...] = ()
