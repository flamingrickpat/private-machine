from datetime import datetime
from enum import StrEnum
from typing import List, Optional

from pydantic import BaseModel, Field

from pm.actions import generate_action_selection
from pm.character import companion_name
from pm.database.db_utils import update_database_item
from pm.database.tables import CognitiveTick, ImpulseRegister, WorldInteractionType, Event
from pm.sensation_evaluation import generate_sensation_evaluation
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