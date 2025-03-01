from enum import StrEnum
from enum import StrEnum
from typing import List, Optional, Any

from pydantic import BaseModel, Field

from pm.common_prompts.get_emotional_state import EmotionalState
from pm.database.tables import Event
from pm.ghost.mental_state import EmotionalAxesModel, NeedsAxesModel
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
    prev_tick_id: int | None
    tick_id: int
    ghost: Optional[Any] = Field(default=None)
    sensation: Impulse
    buffer_add_impulse: List[Event] = Field(default_factory=list)
    buffer_sensation_evaluation: List[Event] = Field(default_factory=list)
    action: Optional[GhostAction] = Field(default=None)
    buffer_plan_action: List[Event] = Field(default_factory=list)
    buffer_create_action: List[Event] = Field(default_factory=list)
    buffer_verify_action: List[Event] = Field(default_factory=list)
    output: Impulse | None = Field(default=None)
    emotional_state: EmotionalAxesModel
    needs_state: NeedsAxesModel

    @property
    def subsystem_description(self) -> str:
        ghost = self.ghost
        return "\n- ".join([x.get_subsystem_name() + ": " + x.get_subsystem_description() for x in ghost.subsystems])


class InvalidActionException(Exception):
    pass
