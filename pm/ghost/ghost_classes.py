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
    buffer_sensation_preprocessing: List[Event] = Field(default_factory=list)
    action: Optional[GhostAction] = Field(default=None)
    buffer_response_planning: List[Event] = Field(default_factory=list)
    buffer_output_verification: List[Event] = Field(default_factory=list)
    output: Impulse | None = Field(default=None)