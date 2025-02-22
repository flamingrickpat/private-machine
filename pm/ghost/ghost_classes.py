from enum import StrEnum
from enum import StrEnum
from typing import List, Optional

from pydantic import BaseModel, Field

from pm.database.tables import Event
from pm.system_classes import ActionType, Impulse


class PipelineStage(StrEnum):
    Null = "Null"
    CreateAction = "CreateAction"
    AddImpulse = "AddImpulse"
    EvalImpulse = "EvalImpulse"
    ChooseAction = "ChooseAction"
    PlanAction = "PlanAction"
    VerifyAction = "VerifyAction"
    PublishOutput = "PublishOutput"


class GhostAction(BaseModel):
    action_type: ActionType
    event: Event


class GhostState(BaseModel):
    tick_id: int
    sensation: Impulse
    buffer_add_impulse: List[Event] = Field(default_factory=list)
    buffer_sensation_evaluation: List[Event] = Field(default_factory=list)
    action: Optional[GhostAction] = Field(default=None)
    buffer_plan_action: List[Event] = Field(default_factory=list)
    buffer_create_action: List[Event] = Field(default_factory=list)
    buffer_verify_action: List[Event] = Field(default_factory=list)
    output: Impulse | None = Field(default=None)