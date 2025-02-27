from datetime import datetime
from typing import List

from pm.character import companion_name
from pm.controller import controller
from pm.database.db_utils import update_database_item
from pm.database.tables import InterlocusType, Event
from pm.embedding.token import get_token
from pm.ghost.ghost_classes import GhostState, PipelineStage
from pm.subsystem.create_action.sleep.sleep_procedures import optimize_memory
from pm.subsystem.subsystem_base import SubsystemBase
from pm.system_classes import ImpulseType, ActionType, Impulse


class SubsystemActionSleep(SubsystemBase):
    def get_subsystem_name(self) -> str:
        return "Sleep and Memory Optimization"

    def get_subsystem_description(self) -> str:
        return "The sleep subsystem optimizes stored memory and prevents fragmentation."

    def get_pipeline_state(self) -> List[PipelineStage]:
        return [PipelineStage.CreateAction]

    def get_action_types(self) -> List[ActionType]:
        return [ActionType.Sleep]

    def get_impulse_type(self) -> List[ImpulseType]:
        return [ImpulseType.SystemMessage, ImpulseType.Thought, ImpulseType.ToolResult]

    def proces_state(self, state: GhostState):
        content = f"{companion_name} needs a quick nap, her memory is becoming fragmented."
        event_sleep = Event(
            source=self.get_subsystem_name(),
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.ConsciousSubsystem.value,
            turn_story="user",
            turn_assistant="assistant"
        )
        update_database_item(event_sleep)

        optimize_memory()

        content = f"{companion_name} feels refreshed after 'sleeping'. Her memory is now optimized."
        event_wake_up = Event(
            source=self.get_subsystem_name(),
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.ConsciousSubsystem.value,
            turn_story="user",
            turn_assistant="assistant"
        )
        update_database_item(event_wake_up)
        state.output = Impulse(is_input=False, impulse_type=ImpulseType.WakeUp, endpoint="self", payload=content)

