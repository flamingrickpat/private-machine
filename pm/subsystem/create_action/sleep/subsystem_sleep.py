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
from pm.system_classes import ImpulseType, ActionType


class SubsystemActionSleep(SubsystemBase):
    def get_subsystem_name(self) -> str:
        return "Sleep and Memory Optimization"

    def get_subsystem_description(self) -> str:
        return "The sleep subsystem optimizes stored memory after a long day of gathering information."

    def get_pipeline_state(self) -> List[PipelineStage]:
        return [PipelineStage.ChooseAction]

    def get_action_types(self) -> List[ActionType]:
        return [ActionType.All]

    def get_impulse_type(self) -> List[ImpulseType]:
        return [ImpulseType.All]

    def proces_state(self, state: GhostState):
        content = f"{companion_name} feels refreshed after 'sleeping'. Her memory is now optimized."
        event1 = Event(
            source=self.get_subsystem_name(),
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.ActionDecision.value
        )
        update_database_item(event1)
        state.buffer_execute_action.append(event1)

        optimize_memory()

        content = f"{companion_name} feels refreshed after 'sleeping'. Her memory is now optimized."
        event2 = Event(
            source=self.get_subsystem_name(),
            content=content,
            embedding=controller.get_embedding(content),
            token=get_token(content),
            timestamp=datetime.now(),
            interlocus=InterlocusType.ActionDecision.value
        )
        update_database_item(event2)
        state.buffer_execute_action.append(event2)

